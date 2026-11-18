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
