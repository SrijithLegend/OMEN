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


def _key(owner):
    return {"x-api-key": c.post("/api/v1/keys", json={"owner": owner}).json()["api_key"]}


def test_keys_workspaces_and_privacy():
    a, b = _key("alice"), _key("bob")
    assert c.post("/api/v1/keys", json={"owner": "alice"}).status_code == 409
    assert c.get("/api/v1/workspaces/p/analyses").status_code == 401
    assert c.get("/api/v1/workspaces/p/analyses", headers={"x-api-key": "nope"}).status_code == 401
    s = c.post("/api/v1/workspaces/support/analyses", headers=a, json={"kind": "estimate", "note": "n", "input": BASE})
    assert s.status_code == 201 and s.json()["output"]["memory"]["fits"]
    assert len(c.get("/api/v1/workspaces/support/analyses", headers=a).json()["analyses"]) == 1
    assert c.get("/api/v1/workspaces/support/analyses", headers=b).json()["analyses"] == []  # bob cannot see alice's
    assert c.delete(f"/api/v1/workspaces/support/analyses/{s.json()['id']}", headers=b).status_code == 404
    assert c.post("/api/v1/workspaces/support/analyses", headers=a, json={"kind": "nonsense", "input": BASE}).status_code == 422
    assert c.delete(f"/api/v1/workspaces/support/analyses/{s.json()['id']}", headers=a).status_code == 200


def test_change_tracking_reports_only_material_changes():
    from app.main import material_changes
    assert material_changes({"a": 100, "b": "x", "c": {"d": 1}}, {"a": 100.5, "b": "x", "c": {"d": 1}}) == []  # 0.5% is noise
    assert material_changes({"a": 100}, {"a": 120}) == [{"path": "a", "before": 100, "after": 120}]
    assert material_changes({"s": "fits"}, {"s": "no"})[0]["path"] == "s"
    h = _key("carol")
    c.post("/api/v1/workspaces/w/analyses", headers=h, json={"kind": "estimate", "input": BASE})
    assert c.get("/api/v1/workspaces/w/changes", headers=h).json() == {"checked": 1, "changed": []}  # nothing moved since saving
    from sqlalchemy import update

    from app import db
    with db.engine.begin() as con:  # simulate an old snapshot from before a catalog change
        con.execute(update(db.saved_analyses).values(output_snapshot={"memory": {"max_users": 5}}).where(db.saved_analyses.c.project == "w"))
    ch = c.get("/api/v1/workspaces/w/changes", headers=h).json()["changed"]
    assert ch and any(x["path"] == "memory.max_users" and x["before"] == 5 for x in ch[0]["changes"])


def test_rate_limit():
    from app import main
    h = _key("dave")
    main.RATE_LIMIT_PER_MINUTE = 3
    try:
        codes = [c.get("/api/v1/workspaces/p/analyses", headers=h).status_code for _ in range(5)]
    finally:
        main.RATE_LIMIT_PER_MINUTE = 60
    assert codes == [200, 200, 200, 429, 429]


def test_metrics_endpoint():
    c.get("/api/v1/models/Llama-3.1-8B")
    text = c.get("/metrics").text
    assert 'omen_http_requests_total{method="GET",route="/api/v1/models/{model_id}",status="200"}' in text


def test_submitter_is_shown_only_when_opted_in():
    a = c.post("/api/v1/benchmarks/submit", json={**GOOD, "context_tokens": 777, "submitter": "anon-person"}).json()["id"]
    b = c.post("/api/v1/benchmarks/submit", json={**GOOD, "context_tokens": 778, "submitter": "proud-person", "attribute_me": True}).json()["id"]
    got = {x["id"]: x for x in c.get("/api/v1/benchmarks", params={"status": "community"}).json()["results"]}
    assert "submitter" not in got[a] and got[b]["submitter"] == "proud-person" and "attribute_me" not in got[b]
