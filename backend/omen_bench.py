"""OMEN benchmark runner. Measures a running OpenAI-compatible server (vLLM, llama.cpp server, ...) and writes a result JSON.

  python omen_bench.py --url http://localhost:8000 --served-model my-model --model Llama-3.1-8B --quant "Q4_K_M" \
      --gpu "RTX 4090 24GB" --engine "llama.cpp b4567" --users 1 --output-tokens 256

Nothing is sent anywhere: the result is written to a local JSON file.

Tokens are counted, best method first, and the result records which one was used:
  usage      the server's own completion token count (stream_options.include_usage). Exact.
  tokenizer  the generated text re-tokenized with --tokenizer (a Hugging Face tokenizer name or path). Exact only if it is the served model's tokenizer.
  chunks     one streamed chunk = one token. An ESTIMATE: a chunk can hold several tokens or part of one, so tokens/s and TPOT can be off.

Repeated measurements: --requests per user are sent back to back, and --rounds repeats the whole measured window, so the result carries a median, mean,
p95, min, max and standard deviation for TTFT, TPOT and per-user tokens/s, each round's aggregate throughput, and warnings when the numbers are noisy.

GPU memory is sampled every 0.5 s through the whole run (warm-up included) with nvidia-smi: the result carries the peak of the summed memory of the
GPUs you name and each GPU's own peak. It is what was in use on those GPUs, including other processes and anything the engine pre-allocated.
"""
import argparse
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

SCHEMA_VERSION = 2

PROMPT_WORD = "benchmark "
EXACTNESS = ["usage", "tokenizer", "chunks"]  # best to worst; a run is only as exact as its weakest request


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("--tokenizer needs the transformers package: pip install transformers")
    return AutoTokenizer.from_pretrained(name)


def one_request(client: httpx.Client, url: str, served_model: str, input_tokens: int, output_tokens: int, tokenizer=None) -> dict:
    body = {"model": served_model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": output_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT_WORD * input_tokens}]}
    t0 = time.perf_counter()
    first = None
    chunks, usage, text = 0, None, []
    with client.stream("POST", url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            try:
                obj = json.loads(line[5:])
            except ValueError:
                continue
            usage = obj.get("usage") or usage  # the final chunk carries it; its choices list is empty
            try:
                delta = obj["choices"][0]["delta"]
            except (KeyError, IndexError, TypeError):
                continue
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                chunks += 1
                text.append(piece)
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    if usage and usage.get("completion_tokens"):
        method, n, prompt = "usage", int(usage["completion_tokens"]), usage.get("prompt_tokens")
    elif tokenizer is not None:
        method, n, prompt = "tokenizer", len(tokenizer.encode("".join(text), add_special_tokens=False)), len(tokenizer.encode(PROMPT_WORD * input_tokens, add_special_tokens=False))
    else:
        method, n, prompt = "chunks", chunks, None
    if first is None or n < 2:
        raise RuntimeError("server returned fewer than 2 tokens")
    # the first chunk is assumed to hold one token, so decode time covers the other n - 1
    return {"ttft_ms": (first - t0) * 1000, "tpot_ms": (end - first) / (n - 1) * 1000, "tokens": n, "chunks": chunks, "prompt_tokens": prompt, "method": method, "seconds": end - t0}


def read_gpu_mem() -> dict[int, float] | None:
    """{GPU index: MiB in use} from nvidia-smi, or None when it is missing or fails."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return {int(i): float(m) for i, m in (ln.split(",") for ln in out.strip().splitlines())}
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


class VramSampler:
    """Polls GPU memory in the background and keeps the peak of the summed memory of `gpu_ids` and each GPU's own peak."""

    def __init__(self, gpu_ids: list[int], interval: float = 0.5):
        self.gpu_ids, self.interval, self.samples, self.missing = gpu_ids, interval, 0, False
        self.peak_total, self.peak_each = 0.0, {i: 0.0 for i in gpu_ids}
        self._stop, self._thread = threading.Event(), None

    def sample(self) -> None:
        mem = read_gpu_mem()
        if mem is None or any(i not in mem for i in self.gpu_ids):
            self.missing = True
            return
        self.samples += 1
        self.peak_total = max(self.peak_total, sum(mem[i] for i in self.gpu_ids))
        for i in self.gpu_ids:
            self.peak_each[i] = max(self.peak_each[i], mem[i])

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=lambda: [self.sample() for _ in iter(lambda: self._stop.wait(self.interval), True)], daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join()
        self.sample()

    def result(self) -> dict:
        if self.missing or not self.samples or self.peak_total <= 0:  # nothing in use on those GPUs means the model is not there: not a measurement
            return {"vram_gib": None, "vram_gib_per_gpu": None, "vram_sampling": None}
        return {"vram_gib": round(self.peak_total / 1024, 2), "vram_gib_per_gpu": [round(self.peak_each[i] / 1024, 2) for i in self.gpu_ids],
                "vram_sampling": {"interval_s": self.interval, "samples": self.samples, "gpu_ids": self.gpu_ids}}


def stats(xs: list[float]) -> dict:
    """Spread of repeated measurements. p95 is the nearest-rank value; cv (stdev / mean) says how noisy the sample is."""
    s = sorted(xs)
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if len(s) > 1 else 0.0
    return {"n": len(s), "median": statistics.median(s), "mean": mean, "p95": s[max(0, math.ceil(0.95 * len(s)) - 1)], "min": s[0], "max": s[-1],
            "stdev": sd, "cv": sd / mean if mean else 0.0}


def run(args) -> dict:
    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.gpu_count))
    if len(gpu_ids) != args.gpu_count:
        sys.exit(f"--gpu-ids lists {len(gpu_ids)} GPUs but --gpu-count is {args.gpu_count}")
    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None
    local = urlparse(args.url).hostname in ("localhost", "127.0.0.1", "::1")  # nvidia-smi can only see this machine's GPUs
    with (VramSampler(gpu_ids) if local else nullcontext()) as vram, httpx.Client() as client:
        for _ in range(args.warmup):  # warm-up requests are discarded
            one_request(client, args.url, args.served_model, args.input_tokens, args.output_tokens, tok)
        results, errors, walls, lock = [], [], [], threading.Lock()
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")

        def worker(rnd):
            try:
                for _ in range(args.requests):
                    r = one_request(client, args.url, args.served_model, args.input_tokens, args.output_tokens, tok)
                    with lock:
                        results.append({**r, "round": rnd})
            except Exception as e:  # keep going: one failed request must not lose the others
                errors.append(str(e))

        for rnd in range(args.rounds):
            threads = [threading.Thread(target=worker, args=(rnd,)) for _ in range(args.users)]
            t0 = time.perf_counter()
            [t.start() for t in threads]
            [t.join() for t in threads]
            walls.append(time.perf_counter() - t0)
    if not results:
        raise SystemExit("no successful requests: " + "; ".join(errors[:3]))
    method = max({r["method"] for r in results}, key=EXACTNESS.index)  # the weakest method used by any request
    tpot = statistics.median(r["tpot_ms"] for r in results)
    ratio = statistics.median(r["chunks"] / r["tokens"] for r in results)
    st = {"ttft_ms": stats([r["ttft_ms"] for r in results]), "tpot_ms": stats([r["tpot_ms"] for r in results]), "tok_s_per_user": stats([1000 / r["tpot_ms"] for r in results])}
    round_tok_s = [sum(r["tokens"] for r in results if r["round"] == i) / w for i, w in enumerate(walls) if any(r["round"] == i for r in results)]  # output tokens of all users / wall time, per round
    agg = stats(round_tok_s)
    known = all(r["prompt_tokens"] is not None for r in results)
    context = max((r["prompt_tokens"] if known else args.input_tokens) + r["tokens"] for r in results)
    warnings = []
    if errors:
        warnings.append(f"{len(errors)} worker(s) stopped after an error ({errors[0][:80]}): the other requests are kept, but the sample is smaller than asked")
    if len(results) < 10:
        warnings.append(f"only {len(results)} measured request(s): the spreads are rough, use more --requests or --rounds")
    if st["tpot_ms"]["cv"] > 0.25:
        warnings.append(f"TPOT varies a lot between requests (CV {st['tpot_ms']['cv']:.0%}): the median is less trustworthy. Check for other load on the GPU or send more requests.")
    if len(round_tok_s) > 1 and agg["cv"] > 0.1:
        warnings.append(f"aggregate throughput differs between rounds (CV {agg['cv']:.0%}): the server did not settle or something else was running")
    notes = [f"{len(results)} requests in {len(walls)} round(s) ({len(errors)} failed), {args.warmup} warm-up, medians",
             {"usage": "tokens counted from the server's usage report (exact)", "tokenizer": f"tokens counted by re-tokenizing the output with {args.tokenizer}",
              "chunks": "tokens ESTIMATED as one per streamed chunk: tokens/s and TPOT may be off"}[method],
             "context_tokens is the largest prompt + completion seen" + ("" if known else " (prompt length assumed from --input-tokens)")]
    if ratio < 0.9 and method != "chunks":
        notes.append(f"the server packs about {1 / ratio:.1f} tokens per chunk, so TPOT assumes the first chunk held one token")
    vr = vram.result() if vram else {"vram_gib": None, "vram_gib_per_gpu": None, "vram_sampling": None}
    if not local:
        notes.append("GPU memory not measured: the server is not on this machine, and nvidia-smi only sees local GPUs")
    elif vr["vram_gib"] is None:
        notes.append("GPU memory not measured (nvidia-smi missing, did not list the GPUs asked for, or showed none in use)")
    else:
        notes.append("vram_gib is the sampled peak of the named GPUs' used memory, including other processes and engine pre-allocation")
    return {
        "model": args.model, "quant": args.quant, "gpu": args.gpu, "engine": args.engine,
        "context_tokens": context, "users": args.users,
        "tok_s_per_user": 1000 / tpot, "ttft_ms": statistics.median(r["ttft_ms"] for r in results), "tpot_ms": tpot,
        **vr, "gpu_count": args.gpu_count, "kv_dtype": args.kv_dtype,
        "token_count_method": method, "tokens_exact": method == "usage", "chunk_token_ratio": round(ratio, 3),
        "source_url": args.source_url,
        "aggregate_tok_s": agg["median"],  # output tokens of all users / wall time of one measured window, median over rounds
        "stats": st, "aggregate_tok_s_stats": agg, "rounds": len(walls), "failed_requests": len(errors), "warnings": warnings,
        "requests_detail": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "method"} for r in results],
        "run": {"schema_version": SCHEMA_VERSION, "runner": "omen_bench", "started_at": started, "python": platform.python_version(), "warmup": args.warmup,
                "requests_per_user": args.requests, "rounds": args.rounds, "input_tokens_requested": args.input_tokens, "output_tokens_requested": args.output_tokens,
                "served_model": args.served_model},
        "notes": "; ".join(notes),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="server base URL, e.g. http://localhost:8000")
    p.add_argument("--served-model", required=True, help="model name the server expects in requests")
    p.add_argument("--model", required=True, help="OMEN catalog model name")
    p.add_argument("--quant", required=True, help="OMEN catalog quantization name")
    p.add_argument("--gpu", required=True, help="OMEN catalog GPU name")
    p.add_argument("--engine", required=True, help='engine and version, e.g. "vllm 0.6.3"')
    p.add_argument("--gpu-count", type=int, default=1, help="GPUs the server uses")
    p.add_argument("--gpu-ids", type=lambda s: [int(x) for x in s.split(",")], default=None, help="nvidia-smi indices the server uses, e.g. 0,1 (default: the first --gpu-count)")
    p.add_argument("--kv-dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8"], help="KV cache type the server runs with")
    p.add_argument("--tokenizer", default=None, help="Hugging Face tokenizer for the served model, used only when the server does not report usage")
    p.add_argument("--users", type=int, default=1, help="concurrent requests")
    p.add_argument("--requests", type=int, default=5, help="requests per user in each round")
    p.add_argument("--rounds", type=int, default=1, help="repeat the whole measured window this many times (concurrent users x requests each time)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--source-url", default=None)
    p.add_argument("--output", default="omen-result.json")
    a = p.parse_args(argv)
    res = run(a)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    st = res["stats"]
    print(f"wrote {a.output}: {res['tok_s_per_user']:.1f} tok/s per user, TTFT {res['ttft_ms']:.0f} ms (p95 {st['ttft_ms']['p95']:.0f}), TPOT {res['tpot_ms']:.1f} ms (p95 {st['tpot_ms']['p95']:.1f}), "
          f"aggregate {res['aggregate_tok_s']:.0f} tok/s, tokens counted by {res['token_count_method']}")
    for w in res["warnings"]:
        print("warning:", w)
    return res


if __name__ == "__main__":
    main()
