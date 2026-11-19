"""Runs the benchmark CLI against a stub OpenAI-style streaming server that emits one token every 10 ms after a 50 ms first-token delay."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import omen_bench


class Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05)
        for i in range(20):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "t"}}]}).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def test_runner_measures_the_stub(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp_path / "r.json"
    try:
        res = omen_bench.main(["--url", f"http://127.0.0.1:{srv.server_port}", "--served-model", "x", "--model", "Llama-3.1-8B", "--quant", "Q4_K_M",
                               "--gpu", "RTX 4090 24GB", "--engine", "stub 1", "--requests", "3", "--warmup", "1", "--users", "2", "--output", str(out)])
    finally:
        srv.shutdown()
    assert 40 < res["ttft_ms"] < 200 and 8 < res["tpot_ms"] < 30  # 50 ms to first token, ~10 ms per token
    assert json.loads(out.read_text())["users"] == 2


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _stub(pieces, usage=None):
    """A server that streams `pieces` (one chunk each, 10 ms apart) and optionally ends with a usage chunk."""
    class H(Stub):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            time.sleep(0.05)
            for p in pieces:
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": p}}]}).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.01)
            if usage:
                self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
    return H


ARGS = ["--served-model", "x", "--model", "Llama-3.1-8B", "--quant", "Q4_K_M", "--gpu", "RTX 4090 24GB", "--engine", "stub 1", "--requests", "2", "--warmup", "0"]


def _bench(handler, tmp_path, *extra):
    srv = _serve(handler)
    try:
        return omen_bench.main(["--url", f"http://127.0.0.1:{srv.server_port}", *ARGS, "--output", str(tmp_path / "r.json"), *extra])
    finally:
        srv.shutdown()


def test_usage_counts_are_exact_even_when_chunks_hold_several_tokens(tmp_path):
    # 10 chunks carrying 2 tokens each: counting chunks would say 10 tokens and a 10 ms TPOT; the server says 20
    res = _bench(_stub(["ab"] * 10, {"prompt_tokens": 300, "completion_tokens": 20}), tmp_path, "--input-tokens", "128", "--output-tokens", "20")
    assert res["token_count_method"] == "usage" and res["tokens_exact"] is True
    assert res["chunk_token_ratio"] == 0.5 and any("tokens per chunk" in n for n in res["notes"].split("; "))
    assert res["context_tokens"] == 320  # prompt tokens from usage (300), not the 128 words sent
    assert res["aggregate_tok_s"] > 0


def test_tpot_divides_decode_time_by_tokens_not_chunks():
    srv = _serve(_stub(["ab"] * 10, {"prompt_tokens": 300, "completion_tokens": 20}))
    try:
        with omen_bench.httpx.Client() as client:
            r = omen_bench.one_request(client, f"http://127.0.0.1:{srv.server_port}", "x", 128, 20)
    finally:
        srv.shutdown()
    assert (r["tokens"], r["chunks"], r["method"]) == (20, 10, "usage")
    decode_ms = r["seconds"] * 1000 - r["ttft_ms"]  # time from the first chunk to the end, whatever the machine's timer resolution
    assert abs(r["tpot_ms"] - decode_ms / 19) < 1e-6  # 19 intervals between 20 tokens, not 9 between 10 chunks


def test_tokenizer_counts_when_the_server_reports_no_usage(tmp_path, monkeypatch):
    class Tok:  # stands in for a Hugging Face tokenizer: one token per character
        def encode(self, text, add_special_tokens=False):
            return list(text)
    monkeypatch.setattr(omen_bench, "load_tokenizer", lambda name: Tok())
    res = _bench(_stub(["ab"] * 10), tmp_path, "--tokenizer", "any/tok", "--input-tokens", "4", "--output-tokens", "20")
    assert res["token_count_method"] == "tokenizer" and res["tokens_exact"] is False
    assert res["context_tokens"] == len("benchmark " * 4) + 20 and "re-tokenizing" in res["notes"]


def test_chunk_counting_is_labelled_an_estimate(tmp_path):
    res = _bench(_stub(["t"] * 20), tmp_path)
    assert res["token_count_method"] == "chunks" and res["tokens_exact"] is False and "ESTIMATED" in res["notes"]
    assert any("assumed" in n for n in res["notes"].split("; "))  # no prompt count: the input length is only assumed


def test_vram_is_a_sampled_peak_with_per_gpu_values(monkeypatch):
    reads = iter([{0: 1000.0, 1: 2000.0}, {0: 3000.0, 1: 1500.0}, {0: 2000.0, 1: 2000.0}])
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: next(reads))
    s = omen_bench.VramSampler([0, 1])
    for _ in range(3):
        s.sample()
    r = s.result()
    assert r["vram_gib_per_gpu"] == [2.93, 1.95]  # each GPU's own peak: 3000 and 2000 MiB
    assert r["vram_gib"] == 4.39 and r["vram_sampling"]["samples"] == 3  # peak of the SUM (3000 + 1500 = 4500 MiB), not after-the-fact memory
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: {0: 1000.0})  # GPU 1 missing
    s2 = omen_bench.VramSampler([0, 1]); s2.sample()
    assert s2.result()["vram_gib"] is None
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: None)  # no nvidia-smi
    s3 = omen_bench.VramSampler([0]); s3.sample()
    assert s3.result()["vram_gib_per_gpu"] is None


def test_the_background_sampler_polls_during_the_run(monkeypatch):
    calls = []
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: calls.append(1) or {0: 5000.0})
    with omen_bench.VramSampler([0], interval=0.02) as s:
        time.sleep(0.15)
    assert len(calls) >= 5 and s.result()["vram_gib"] == 4.88  # it kept sampling while the work ran


def test_stats_describe_repeated_measurements():
    s = omen_bench.stats([10, 12, 11, 30, 11, 10, 12, 11, 10, 13])
    assert (s["n"], s["min"], s["max"], s["median"]) == (10, 10, 30, 11.0) and s["p95"] == 30 and s["cv"] > 0.4  # one slow outlier shows up in p95 and cv
    one = omen_bench.stats([5.0])
    assert one["stdev"] == 0 and one["p95"] == 5.0 and one["n"] == 1


def test_result_carries_spreads_rounds_per_request_detail_and_warnings(tmp_path):
    res = _bench(_stub(["t"] * 20), tmp_path, "--users", "2", "--requests", "3", "--rounds", "2")
    assert res["rounds"] == 2 and res["stats"]["tpot_ms"]["n"] == 12 and len(res["requests_detail"]) == 12  # 2 users x 3 requests x 2 rounds
    assert res["aggregate_tok_s_stats"]["n"] == 2 and res["aggregate_tok_s"] == res["aggregate_tok_s_stats"]["median"]
    assert res["tpot_ms"] == res["stats"]["tpot_ms"]["median"] and abs(res["tok_s_per_user"] - 1000 / res["tpot_ms"]) < 1e-9
    assert {"ttft_ms", "tpot_ms", "tokens", "round"} <= set(res["requests_detail"][0])
    assert res["run"]["schema_version"] == 2 and res["run"]["rounds"] == 2 and res["run"]["requests_per_user"] == 3 and res["run"]["started_at"].endswith("+00:00")
    assert json.loads((tmp_path / "r.json").read_text())["stats"]["ttft_ms"]["p95"] >= res["stats"]["ttft_ms"]["median"]
    few = _bench(_stub(["t"] * 20), tmp_path, "--requests", "2")  # 2 requests: say the spreads are rough
    assert any("only 2 measured request" in w for w in few["warnings"])


def test_a_noisy_run_is_flagged(tmp_path):
    class Noisy(Stub):
        n = 0

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            type(self).n += 1
            delay = 0.002 if type(self).n % 2 else 0.04  # alternate fast and slow requests
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(12):
                self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "t"}}]}).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(delay)
            self.wfile.write(b"data: [DONE]\n\n")
    res = _bench(Noisy, tmp_path, "--requests", "6")
    assert res["stats"]["tpot_ms"]["cv"] > 0.25 and any("varies a lot" in w for w in res["warnings"])


def test_the_result_file_is_importable_as_a_reference_run(tmp_path):
    from app import benchmarks
    res = _bench(_stub(["ab"] * 10, {"prompt_tokens": 100, "completion_tokens": 20}), tmp_path, "--source-url", "https://example.com/my-run")
    assert benchmarks.validate(res) == []  # the runner's own file passes the importer's checks as it is
    assert benchmarks.validate({**res, "source_url": None}) == ["source_url is required: a reference run must say where it came from"]


def test_memory_is_not_reported_for_a_remote_server_or_an_idle_gpu(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: seen.append(1) or {0: 5000.0})
    monkeypatch.setattr(omen_bench, "one_request", lambda *a, **k: {"ttft_ms": 50.0, "tpot_ms": 10.0, "tokens": 20, "chunks": 20, "prompt_tokens": 100, "method": "usage", "seconds": 0.3})
    res = omen_bench.main(["--url", "http://gpu-box.internal:8000", *ARGS, "--output", str(tmp_path / "r.json")])  # a server on another machine
    assert res["vram_gib"] is None and seen == [] and "not on this machine" in res["notes"]  # nvidia-smi here says nothing about that server
    local = omen_bench.main(["--url", "http://localhost:8000", *ARGS, "--output", str(tmp_path / "r2.json")])
    assert local["vram_gib"] == 4.88 and seen  # the same data is used when the server is local
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: {0: 0.0})  # a GPU with nothing loaded is not a memory measurement
    idle = omen_bench.VramSampler([0]); idle.sample()
    assert idle.result()["vram_gib"] is None
