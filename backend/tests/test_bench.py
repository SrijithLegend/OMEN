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
    assert res["consent_to_publish"] is False and json.loads(out.read_text())["users"] == 2


def test_submit_requires_consent():
    import pytest
    with pytest.raises(SystemExit):
        omen_bench.main(["--url", "http://127.0.0.1:1", "--served-model", "x", "--model", "m", "--quant", "q", "--gpu", "g", "--engine", "e", "--submit", "http://x"])


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
    assert res["tpot_ms"] < 7  # ~90 ms of decode over 19 tokens, not over 9 chunks
    assert res["context_tokens"] == 320  # prompt tokens from usage (300), not the 128 words sent
    assert res["aggregate_tok_s"] > 0


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


def test_submit_body_carries_only_schema_fields(tmp_path, monkeypatch):
    sent = {}
    monkeypatch.setattr(omen_bench.httpx, "post", lambda url, json, timeout: type("R", (), {"status_code": 202, "text": "{}"})() if sent.update(json) is None else None)
    monkeypatch.setattr(omen_bench, "read_gpu_mem", lambda: {0: 9000.0})
    srv = _serve(_stub(["ab"] * 10, {"prompt_tokens": 50, "completion_tokens": 20}))
    try:
        omen_bench.main(["--url", f"http://127.0.0.1:{srv.server_port}", *ARGS, "--output", str(tmp_path / "r.json"), "--submit", "http://api", "--consent"])
    finally:
        srv.shutdown()
    assert sent["token_count_method"] == "usage" and sent["vram_gib_per_gpu"] == [8.79] and sent["consent_to_publish"] is True
    assert "notes" not in sent and "aggregate_tok_s" not in sent and "tokens_exact" not in sent
