from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)
BASE = {"model": "Llama-3.1-8B", "quantization": "Q4_K_M", "gpu": "RTX 4090 24GB", "context_length": 8192, "concurrency": 4}


def test_memory_matches_the_web_calculator():  # same numbers the page shows for this setup
    m = c.post("/api/v1/estimate/memory", json=BASE).json()["memory"]
    assert round(m["weights_gib"], 1) == 4.6 and m["max_users"] == 17 and m["fits"] and m["status"] == "likely_fits"
    assert m["largest_component"] == "weights"


def test_performance_ceiling_and_unknown_prefill():
    p = c.post("/api/v1/estimate/performance", json=BASE).json()
    assert round(p["performance"]["decode_ceiling_tok_s"]) == 205
    assert p["performance"]["ttft_ms"] > 0  # 4090 compute is known
    q = c.post("/api/v1/estimate/performance", json={**BASE, "gpu": "RTX 3060 12GB"}).json()
    assert q["performance"]["ttft_ms"] is None and any("prefill" in w for w in q["warnings"])


def test_does_not_fit_and_measured_evidence():
    big = c.post("/api/v1/estimate/memory", json={**BASE, "model": "Llama-3.1-70B", "quantization": "fp16 / bf16"}).json()
    assert not big["memory"]["fits"] and big["memory"]["status"] == "does_not_fit_under_stated_assumptions"
    m = c.post("/api/v1/estimate/memory", json={"model": "Llama-2-7B", "quantization": "Q4_0", "gpu": "RTX 4090 24GB"}).json()
    assert m["evidence"]["benchmark_count"] == 1 and m["memory"]["status"] == "measured_to_fit"


def test_lookup_and_errors():
    assert c.get("/api/v1/models/Llama-3.1-8B").json()["layers"] == 32
    assert c.get("/api/v1/gpus/H100 80GB SXM").json()["tflops_fp16"] == 989
    assert c.get("/api/v1/models/nope").status_code == 404
    assert c.post("/api/v1/estimate/memory", json={**BASE, "gpu": "nope"}).status_code == 404
    assert c.post("/api/v1/estimate/memory", json={**BASE, "concurrency": 0}).status_code == 422
