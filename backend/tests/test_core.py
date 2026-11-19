import os
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
    # a published speed run exists for this setup, but none recorded memory: it must not read as "measured to fit"
    assert m["evidence"]["type"] == "performance_measured" and m["evidence"]["memory_measured"] == 0 and m["memory"]["status"] == "likely_fits"
    assert any("speed benchmark" in w for w in m["warnings"])


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


def test_price_catalog_records_sources_and_feeds_cost():
    P = {"provider": "AWS", "instance_type": "test.4x", "region": "us-east-1", "gpu": "RTX 4090 24GB", "gpu_count": 4, "usd_hr": 8.0,
         "source_url": "https://example.com/pricing", "observed_at": "2026-11-01"}
    body = {**BASE, "provider": "AWS"}
    assert c.post("/api/v1/prices", json=P).status_code == 403  # reviewer key needed
    assert c.post("/api/v1/cost/estimate", json=body).status_code == 404  # no price recorded: refuses rather than inventing one
    h = {"x-admin-key": "test-admin"}
    assert c.post("/api/v1/prices", headers=h, json={**P, "provider": "Nope"}).status_code == 422
    assert c.post("/api/v1/prices", headers=h, json=P).status_code == 201
    c.post("/api/v1/prices", headers=h, json={**P, "usd_hr": 12.0, "observed_at": "2026-11-15"})  # price change: new row, old one kept
    assert [r["usd_hr"] for r in c.get("/api/v1/prices", params={"provider": "AWS"}).json()["prices"]] == [12.0]
    assert len(c.get("/api/v1/prices", params={"history": True}).json()["prices"]) == 2
    r = c.post("/api/v1/cost/estimate", json={**body, "gpu_count": 1}).json()
    assert r["price_used"]["usd_hr_per_gpu"] == 3.0 and "2026-11-15" in r["warnings"][0]


def test_confidence_threshold_is_75_percent():
    from app.recommend import HIGH_CONFIDENCE, confidence, label
    assert HIGH_CONFIDENCE == 0.75
    from app.catalog import RUNS
    assert confidence(RUNS, "Llama-2-7B", "Q4_0", "H100 80GB SXM", 1)[0] == 0.9 and label(0.9) == "high"  # measured at that concurrency
    assert confidence(RUNS, "Llama-2-7B", "Q4_0", "H100 80GB SXM", 4)[0] == 0.7 and label(0.7) == "moderate"  # measured, other concurrency: below threshold
    assert confidence(RUNS, "Llama-3.1-70B", "Q4_K_M", "H100 80GB SXM", 1)[0] == 0.5  # other models measured on this GPU
    assert confidence(RUNS, "Llama-3.1-70B", "Q4_K_M", "A10 24GB", 1)[0] == 0.3 and label(0.3) == "low"
    body = {"model": "Llama-2-7B", "concurrency": 1, "min_tok_s": 1}
    every = c.post("/api/v1/recommendations", json=body).json()
    sure = c.post("/api/v1/recommendations", json={**body, "min_confidence": 0.75}).json()
    assert sure["feasible"] < every["feasible"] and sure["removed"]["low_confidence"] > 0
    assert all(p["confidence"] >= 0.75 and p["confidence_label"] == "high" for p in sure["picks"].values())
    assert any(p["confidence_label"] != "high" for p in every["picks"].values()) or every["picks"]["best_evidence"]["confidence"] == 0.9


def _run(**kw):
    return {"model": "Llama-2-7B", "quant": "Q4_0", "gpu": "RTX 4090 24GB", "gpu_count": 1, "engine": "llama.cpp b1", "context": 128, "users": 1, "tok_s": 150, "vram_gib": None, "kv_dtype": "fp16", **kw}


def test_only_a_peak_vram_measurement_establishes_measured_fit():
    from app.evidence import evidence
    ask = dict(model="Llama-2-7B", quant="Q4_0", gpu="RTX 4090 24GB", vram_gib=24, context=4096, users=2)
    assert evidence([_run()], **ask)["type"] == "performance_measured"  # speed only
    mem = _run(context=4096, users=4, vram_gib=9.0)
    assert evidence([mem], **ask)["type"] == "memory_measured"
    assert evidence([_run(context=2048, users=4, vram_gib=9.0)], **ask)["memory_measured"] == 0  # measured at a shorter context
    assert evidence([_run(context=4096, users=1, vram_gib=9.0)], **ask)["memory_measured"] == 0  # measured with fewer users
    assert evidence([{**mem, "gpu_count": 2}], **ask)["memory_measured"] == 0  # different GPU count
    assert evidence([{**mem, "kv_dtype": "fp8"}], **ask)["memory_measured"] == 0  # fp8 cache measured, fp16 requested: no proof
    assert evidence([{**mem, "kv_dtype": "fp16"}], **{**ask, "kv_dtype": "fp8"})["memory_measured"] == 1  # fp16 fits, so fp8 certainly does
    assert evidence([{**mem, "engine": "vllm 0.6"}], **ask, engine="llama.cpp")["memory_measured"] == 0
    assert evidence([_run(gpu="H100 80GB SXM")], **ask)["type"] == "model_estimate"
    assert evidence([_run(gpu="RTX 4090 24GB", model="Llama-3.1-8B")], **ask)["type"] == "calibrated_estimate"


def test_verified_vram_run_promotes_to_measured_to_fit_and_bad_vram_is_rejected():
    run = {"model": "Llama-3.1-8B", "quant": "Q4_K_M", "gpu": "RTX 4090 24GB", "engine": "vllm 0.6.3", "context_tokens": 8192, "users": 4, "tok_s_per_user": 90,
           "vram_gib": 9.1, "source_url": "https://example.com/vram", "consent_to_publish": True}
    rid = c.post("/api/v1/benchmarks/submit", json=run).json()["id"]
    assert c.post("/api/v1/estimate/memory", json=BASE).json()["memory"]["status"] == "likely_fits"  # community runs are not evidence yet
    assert c.post(f"/api/v1/benchmarks/{rid}/verify", headers={"x-admin-key": "test-admin"}).status_code == 200
    m = c.post("/api/v1/estimate/memory", json=BASE).json()
    assert m["memory"]["status"] == "measured_to_fit" and m["evidence"]["type"] == "memory_measured" and m["evidence"]["peak_vram_gib"] == 9.1
    assert c.post("/api/v1/estimate/memory", json={**BASE, "context_length": 16384}).json()["memory"]["status"] != "measured_to_fit"
    huge = c.post("/api/v1/benchmarks/submit", json={**run, "context_tokens": 4096, "vram_gib": 40}).json()["id"]
    tiny = c.post("/api/v1/benchmarks/submit", json={**run, "context_tokens": 2048, "vram_gib": 1.0}).json()["id"]
    rej = {x["id"]: x["problems"][0] for x in c.get("/api/v1/benchmarks", params={"status": "rejected"}).json()["results"]}
    assert "more than" in rej[huge] and "below" in rej[tiny]


def test_kv_cache_dtype_and_sliding_window():
    fp16 = c.post("/api/v1/estimate/memory", json=BASE).json()["memory"]
    for dt, half in (("bf16", False), ("fp8", True), ("int8", True)):
        m = c.post("/api/v1/estimate/memory", json={**BASE, "kv_dtype": dt}).json()["memory"]
        assert (m["kv_gib_per_user"] == fp16["kv_gib_per_user"] / 2) == half and m["kv_dtype"] == dt
    assert c.post("/api/v1/estimate/memory", json={**BASE, "kv_dtype": "fp4"}).status_code == 422
    g = {**BASE, "model": "Gemma-3-27B", "gpu": "H100 80GB SXM", "context_length": 32768}
    full = c.post("/api/v1/estimate/memory", json=g).json()
    win = c.post("/api/v1/estimate/memory", json={**g, "sliding_cache": True}).json()["memory"]
    assert full["memory"]["kv_gib_per_user_if_sliding_freed"] < full["memory"]["kv_gib_per_user"] and not full["memory"]["sliding_window_applied"]
    assert win["sliding_window_applied"] and abs(win["kv_gib_per_user"] - full["memory"]["kv_gib_per_user_if_sliding_freed"]) < 1e-9
    assert any("sliding-window" in w for w in full["warnings"])
    plain = c.post("/api/v1/estimate/memory", json={**BASE, "sliding_cache": True}).json()  # Llama-3.1-8B has no sliding layers
    assert not plain["memory"]["sliding_window_applied"] and any("ignored" in w for w in plain["warnings"])


def test_calibration_prefers_this_gpu_and_never_hides_how_thin_it_is():
    from app.performance import calibrate
    ask = dict(gpu="RTX 4090 24GB", context=8192, users=4)
    none = calibrate([], **ask)
    assert none["utilization"] is None and none["level"] == "none"
    other = calibrate([_run(gpu="H100 80GB SXM", tok_s=200)], **ask)
    assert other["level"] == "extrapolated" and other["tier"] == "other GPUs" and other["low"] is None and other["caveats"]
    mine = [_run(tok_s=100), _run(tok_s=140), _run(tok_s=120, engine="vllm 0.6")]
    one = calibrate(mine[:1], **ask)
    assert one["level"] == "thin" and one["tier"] == "this GPU" and one["conditions_matched"] == 0
    many = calibrate(mine * 2, **ask, engine="llama.cpp b9")  # engine given: only llama.cpp runs count, 4 of them
    assert many["tier"] == "this GPU and engine" and many["runs"] == 4 and many["level"] == "gpu_specific" and many["low"] < many["utilization"] < many["high"]
    assert calibrate(mine[2:], **ask, engine="llama.cpp")["tier"] == "this GPU, other engines"  # falls back, and says so
    assert calibrate([_run(context=8192, users=4)], **ask)["conditions_matched"] == 1


def test_performance_endpoint_reports_calibration_support_and_range():
    p = c.post("/api/v1/estimate/performance", json=BASE).json()["performance"]
    cal = p["calibration"]
    assert cal["runs"] >= 1 and cal["level"] in ("thin", "gpu_specific") and cal["tier"].startswith("this GPU")  # the 4090 run is in the seed data
    far = c.post("/api/v1/estimate/performance", json={**BASE, "gpu": "A100 40GB"}).json()
    assert far["performance"]["calibration"]["level"] == "extrapolated" and far["performance"]["decode_tok_s_range"] and any("borrows" in w for w in far["warnings"])
    manual = c.post("/api/v1/estimate/performance", json={**BASE, "utilization": 0.5}).json()["performance"]
    assert manual["utilization_source"] == "user" and manual["decode_tok_s_range"] is None


def test_quality_loss_is_labelled_and_unknown_cannot_pass_a_limit():
    q = c.get("/api/v1/quantizations").json()
    assert q["Q4_0"]["quality_loss_kind"] == "measured" and round(q["Q4_0"]["quality_loss_pct"], 1) == 4.2
    assert q["GPTQ / AWQ int4"]["quality_loss_kind"] == "estimated" and q["int2"]["quality_loss_kind"] == "unknown" and q["int2"]["quality_loss_pct"] is None
    assert round(q["Q2_K"]["bytes_per_param"] * 8, 2) == 3.35  # whole-file bits per weight in the llama.cpp README, not the 2.56 of the type alone
    r = c.post("/api/v1/recommendations", json={"model": "Llama-3.1-8B", "max_quality_loss_pct": 50}).json()
    assert r["removed"]["unknown_quality_loss"] > 0
    assert all(p["quality_loss_pct"] is not None for p in r["picks"].values())


def test_gpu_specs_carry_provenance_and_unverified_ones_warn():
    h100, t4, rtx = (c.get(f"/api/v1/gpus/{n}").json() for n in ("H100 80GB SXM", "T4 16GB", "RTX 4090 24GB"))
    assert h100["spec_status"] == "verified" and h100["spec_source"]["date"] and h100["spec_source"]["source"].startswith("https://")
    assert t4["bandwidth_gbs"] == 300  # corrected to the datasheet (was 320 from memory)
    assert rtx["spec_status"] == "unverified" and rtx["spec_source"] is None
    assert any("not verified" in w for w in c.post("/api/v1/estimate/memory", json=BASE).json()["warnings"])
    assert not any("verified" in w for w in c.post("/api/v1/estimate/memory", json={**BASE, "gpu": "H100 80GB SXM"}).json()["warnings"])


def test_public_submission_is_rate_limited_bounded_and_size_capped():
    from app import main
    bad = {**GOOD, "context_tokens": 4097, "consent_to_publish": True}
    for patch in ({"model": "x" * 101}, {"tok_s_per_user": 1e9}, {"source_url": "javascript:alert(1)"}, {"users": 10**9}, {"submitter": "y" * 65}):
        assert c.post("/api/v1/benchmarks/submit", json={**bad, **patch}).status_code == 422, patch
    assert c.post("/api/v1/benchmarks/submit", content="x" * 70_000, headers={"content-type": "application/json"}).status_code == 413
    main.SUBMIT_LIMIT, main._submits = 2, {}
    try:
        codes = [c.post("/api/v1/benchmarks/submit", json={**bad, "context_tokens": 5000 + i}).status_code for i in range(4)]
    finally:
        main.SUBMIT_LIMIT, main._submits = 1000, {}
    assert codes == [202, 202, 429, 429]


def test_production_refuses_to_start_with_open_cors():
    import subprocess, sys
    base = {**os.environ, "OMEN_ENV": "production", "OMEN_DB": "sqlite://"}
    base.pop("OMEN_CORS", None)
    run = lambda env: subprocess.run([sys.executable, "-c", "import app.main"], env=env, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)))
    assert "OMEN_CORS" in run(base).stderr
    assert run({**base, "OMEN_CORS": "https://omen.example"}).returncode == 0
