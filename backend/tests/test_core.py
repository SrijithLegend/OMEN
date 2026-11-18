from fastapi.testclient import TestClient

from app.main import app

c = TestClient(app)
c.__enter__()  # runs the app startup (creates and seeds the in-memory database)
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


def test_workload_and_cost():
    w = c.post("/api/v1/workloads/simulate", json={**BASE, "gpu": "H100 80GB SXM", "requests_per_month": 1_000_000, "output_tokens": 500}).json()
    assert w["status"] == "estimated" and w["workload"]["usd_per_million_output_tokens"] > 0
    assert w["workload"]["monthly_usd_always_on"] >= w["workload"]["monthly_usd_pay_per_use"]
    k = c.post("/api/v1/cost/estimate", json={**BASE, "hardware_usd": 1800}).json()
    assert k["self_hosted"]["usd_hr_while_running"] > 0
    big = c.post("/api/v1/cost/estimate", json={**BASE, "model": "Llama-3.1-70B", "quantization": "fp16 / bf16"}).json()
    assert big["status"] == "does_not_fit"


def test_workload_math_and_break_even():
    from app.workload import break_even_months, self_hosted_usd_hr, simulate
    s = simulate(100, 4, 1, 500, 1_000_000, 0.5)  # same case as test.js
    assert s["aggregate_tok_s"] == 200 and round(s["usd_per_million_output_tokens"], 3) == 1.389 and round(s["gpu_hours_per_month"]) == 694
    # $3650 over 36 months = $101.4/month, no power/hosting: $0.139/hr; renting at $0.40 saves $0.261/hr x 730h
    own = self_hosted_usd_hr(3650, 36, 0, 0)
    assert round(own, 3) == 0.139 and round(break_even_months(3650, 0.40, own, 730), 1) == 19.1
    assert break_even_months(3650, 0.10, own, 730) is None


def test_compare_warns_on_different_workloads():
    a = {**BASE, "requests_per_month": 1000}
    same = c.post("/api/v1/configurations/compare", json={"configurations": [a, {**a, "gpu": "A100 80GB"}]}).json()
    assert same["warnings"] == [] and len(same["results"]) == 2
    diff = c.post("/api/v1/configurations/compare", json={"configurations": [a, {**a, "output_tokens": 999}]}).json()
    assert diff["warnings"]


def test_recommendations():
    r = c.post("/api/v1/recommendations", json={"model": "Llama-3.1-8B", "concurrency": 4, "max_usd_hr": 1.0}).json()
    assert r["feasible"] > 0 and r["picks"]["cheapest"]["usd_hr"] <= 1.0
    assert r["picks"]["fastest"]["decode_ceiling_tok_s"] >= r["picks"]["cheapest"]["decode_ceiling_tok_s"]
    none = c.post("/api/v1/recommendations", json={"model": "Llama-3.1-405B", "max_usd_hr": 0.01}).json()
    assert none["feasible"] == 0 and none["picks"] == {}


GOOD = {"model": "Llama-3.1-8B", "quant": "Q4_K_M", "gpu": "RTX 4090 24GB", "engine": "llama.cpp b9999", "context_tokens": 4096, "users": 1,
        "tok_s_per_user": 120, "source_url": "https://example.com/run", "consent_to_publish": True}


def test_benchmark_flow():
    seeded = c.get("/api/v1/benchmarks").json()["results"]
    assert len(seeded) == 5 and all(r["status"] == "sourced" for r in seeded)
    assert c.post("/api/v1/benchmarks/submit", json={**GOOD, "consent_to_publish": False}).status_code == 400
    r = c.post("/api/v1/benchmarks/submit", json=GOOD)  # background validation has run by the time the response returns
    assert r.status_code == 202
    got = c.get("/api/v1/benchmarks", params={"status": "community"}).json()["results"]
    assert [x["id"] for x in got] == [r.json()["id"]] and "submitter" not in got[0]
    dup = c.post("/api/v1/benchmarks/submit", json={**GOOD, "tok_s_per_user": 121}).json()["id"]
    row = c.get("/api/v1/benchmarks", params={"status": "rejected"}).json()["results"]
    assert [x["id"] for x in row] == [dup] and row[0]["duplicate_of"] == r.json()["id"]


def test_implausible_and_unknown_runs_are_rejected_and_verify_needs_key():
    fast = c.post("/api/v1/benchmarks/submit", json={**GOOD, "context_tokens": 1, "tok_s_per_user": 5000}).json()["id"]  # 8B Q4 on a 4090 cannot do 5000 tok/s
    bogus = c.post("/api/v1/benchmarks/submit", json={**GOOD, "gpu": "Imaginary 9000"}).json()["id"]
    rej = {x["id"]: x for x in c.get("/api/v1/benchmarks", params={"status": "rejected"}).json()["results"]}
    assert "ceiling" in rej[fast]["problems"][0] and "not in the catalog" in rej[bogus]["problems"][0]
    ok = c.get("/api/v1/benchmarks", params={"status": "community"}).json()["results"][0]["id"]
    assert c.post(f"/api/v1/benchmarks/{ok}/verify").status_code == 403
    assert c.post(f"/api/v1/benchmarks/{ok}/verify", headers={"x-admin-key": "test-admin"}).json()["status"] == "verified"
    assert c.post(f"/api/v1/benchmarks/{ok}/verify", headers={"x-admin-key": "test-admin"}).status_code == 409
