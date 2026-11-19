"""OMEN benchmark runner. Measures a running OpenAI-compatible server (vLLM, llama.cpp server, ...) and writes a result JSON.

  python omen_bench.py --url http://localhost:8000 --served-model my-model --model Llama-3.1-8B --quant "Q4_K_M" \
      --gpu "RTX 4090 24GB" --engine "llama.cpp b4567" --users 1 --output-tokens 256

Nothing is sent anywhere unless you pass --submit API_URL --consent. Tokens are counted as streamed chunks, which matches tokens for most servers
(one chunk per token) but can differ if a server batches tokens per chunk; the result records this in notes.
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time

import httpx

PROMPT_WORD = "benchmark "


def one_request(client: httpx.Client, url: str, served_model: str, input_tokens: int, output_tokens: int) -> dict:
    body = {"model": served_model, "stream": True, "max_tokens": output_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT_WORD * input_tokens}]}
    t0 = time.perf_counter()
    first = None
    chunks = 0
    with client.stream("POST", url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            try:
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, ValueError):
                continue
            if delta:
                chunks += 1
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    if first is None or chunks < 2:
        raise RuntimeError("server returned fewer than 2 tokens")
    return {"ttft_ms": (first - t0) * 1000, "tpot_ms": (end - first) / (chunks - 1) * 1000, "tokens": chunks, "seconds": end - t0}


def run(args) -> dict:
    with httpx.Client() as client:
        for _ in range(args.warmup):  # warm-up requests are discarded
            one_request(client, args.url, args.served_model, args.input_tokens, args.output_tokens)
        results, errors, lock = [], [], threading.Lock()

        def worker():
            try:
                for _ in range(args.requests):
                    r = one_request(client, args.url, args.served_model, args.input_tokens, args.output_tokens)
                    with lock:
                        results.append(r)
            except Exception as e:  # keep going: one failed request must not lose the others
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(args.users)]
        t0 = time.perf_counter()
        [t.start() for t in threads]
        [t.join() for t in threads]
        wall = time.perf_counter() - t0
    if not results:
        raise SystemExit("no successful requests: " + "; ".join(errors[:3]))
    tpot = statistics.median(r["tpot_ms"] for r in results)
    return {
        "model": args.model, "quant": args.quant, "gpu": args.gpu, "engine": args.engine,
        "context_tokens": args.input_tokens + args.output_tokens, "users": args.users,
        "tok_s_per_user": 1000 / tpot, "ttft_ms": statistics.median(r["ttft_ms"] for r in results), "tpot_ms": tpot,
        "vram_gib": vram_gib(args.gpu_count), "gpu_count": args.gpu_count, "kv_dtype": args.kv_dtype, "source_url": args.source_url, "submitter": args.submitter, "consent_to_publish": False, "attribute_me": args.attribute,
        "aggregate_tok_s": sum(r["tokens"] for r in results) / wall,
        "notes": f"{len(results)} requests ({len(errors)} failed), {args.warmup} warm-up, medians; tokens counted as streamed chunks",
    }


def vram_gib(gpu_count: int = 1) -> float | None:
    """memory.used summed over the first `gpu_count` GPUs, sampled after the run. It is the footprint at that moment (engines such as vLLM pre-allocate), not a minimum."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return round(sum(float(x) for x in out.split()[:gpu_count]) / 1024, 2)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="server base URL, e.g. http://localhost:8000")
    p.add_argument("--served-model", required=True, help="model name the server expects in requests")
    p.add_argument("--model", required=True, help="OMEN catalog model name")
    p.add_argument("--quant", required=True, help="OMEN catalog quantization name")
    p.add_argument("--gpu", required=True, help="OMEN catalog GPU name")
    p.add_argument("--engine", required=True, help='engine and version, e.g. "vllm 0.6.3"')
    p.add_argument("--gpu-count", type=int, default=1, help="GPUs the server uses (VRAM is summed over this many)")
    p.add_argument("--kv-dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8"], help="KV cache type the server runs with")
    p.add_argument("--users", type=int, default=1, help="concurrent requests")
    p.add_argument("--requests", type=int, default=5, help="requests per user")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--source-url", default=None)
    p.add_argument("--submitter", default=None)
    p.add_argument("--attribute", action="store_true", help="opt in to showing --submitter publicly (default: anonymous)")
    p.add_argument("--output", default="omen-result.json")
    p.add_argument("--submit", metavar="API_URL", help="OMEN API base URL to submit to")
    p.add_argument("--consent", action="store_true", help="required with --submit: you agree to publish this result")
    a = p.parse_args(argv)
    if a.submit and not a.consent:  # fail before spending minutes measuring
        sys.exit("--submit needs --consent: OMEN never publishes a result without it")
    res = run(a)
    if a.submit:
        res["consent_to_publish"] = True
        body = {k: v for k, v in res.items() if k not in ("aggregate_tok_s", "notes")}
        r = httpx.post(a.submit.rstrip("/") + "/api/v1/benchmarks/submit", json=body, timeout=30)
        print("submit:", r.status_code, r.text)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {a.output}: {res['tok_s_per_user']:.1f} tok/s per user, TTFT {res['ttft_ms']:.0f} ms, TPOT {res['tpot_ms']:.1f} ms")
    return res


if __name__ == "__main__":
    main()
