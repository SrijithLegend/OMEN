"""End to end: a benchmark travels submit -> pending -> community -> verified -> public explorer -> trusted calibration, and only a verified run
(never a pending, community, rejected or later-rejected one) moves the speed estimate. Everything goes through the HTTP API except the one step
that cannot be seen from outside: a run caught while still pending (the test client finishes background work before it responds)."""
import os

from fastapi.testclient import TestClient

from app import benchmarks
from app.catalog import GPUS, MODELS, QUANTS
from app.main import BenchmarkSubmission, app
from app.performance import decode_ceiling

c = TestClient(app)
c.__enter__()  # app startup: creates and seeds the database if no other test module has yet

ADMIN = {"x-admin-key": os.environ["OMEN_ADMIN_KEY"]}
REVIEW = {"reviewer": "E2E Reviewer", "notes": "Reproduced the figure from the linked log.", "evidence_reviewed": "linked log, reproduction on the same GPU"}
GPU, MODEL, QUANT = "RTX 3090 24GB", "Llama-3.1-8B", "Q4_K_M"  # one GPU no other test touches, with one seeded run (Llama-2-7B, 1 user, 128 tokens)
ASK = {"model": MODEL, "quantization": QUANT, "gpu": GPU, "concurrency": 7, "context_length": 5000}  # a workload no seeded run matches
RUN = {"model": MODEL, "quant": QUANT, "gpu": GPU, "engine": "vllm 0.6.3", "context_tokens": 5000, "users": 7, "tok_s_per_user": 50,
       "source_url": "https://example.com/e2e", "consent_to_publish": True}
CEIL = decode_ceiling(MODELS[MODEL]["params_b"], QUANTS[QUANT]["bytes_per_param"], GPUS[GPU]["bandwidth_gbs"])  # what a tok_s of N means as a ratio


def calibration() -> dict:
    return c.post("/api/v1/estimate/performance", json=ASK).json()["performance"]["calibration"]


def explorer(**params) -> list[int]:
    return [r["id"] for r in c.get("/api/v1/benchmarks", params=params).json()["results"]]


def submit(**patch) -> int:
    return c.post("/api/v1/benchmarks/submit", json={**RUN, **patch}).json()["id"]


def status(rid: int) -> str:
    return c.get(f"/api/v1/benchmarks/{rid}").json()["status"]


def trusted_ids_for_gpu() -> list[float]:
    return [r["tok_s"] for r in benchmarks.trusted_runs() if r["gpu"] == GPU and r["users"] == 7]


def test_the_whole_lifecycle_and_what_each_stage_may_influence():
    # baseline: nothing measured at 7 users, so the estimate leans on the seeded single-user run and says so
    base = calibration()
    assert base["workload_match"] == "none" and base["level"] == "extrapolated" and trusted_ids_for_gpu() == []

    # 1. a run caught while pending: stored, but invisible to the public and to calibration
    pending = benchmarks.submit(BenchmarkSubmission(**RUN).model_dump())
    assert status(pending) == "pending" and pending not in explorer() and trusted_ids_for_gpu() == [] and calibration() == base

    # 2. validation promotes it to community; the submitter can see it, the explorer lists it, calibration still ignores it
    benchmarks.process(pending)
    assert status(pending) == "community" and pending in explorer() and pending in explorer(status="community")
    assert trusted_ids_for_gpu() == [] and calibration() == base  # unverified data never calibrates
    assert c.get(f"/api/v1/benchmarks/{pending}").json()["reviews"] == []

    # 3. a reviewer verifies it: now it is evidence
    assert c.post(f"/api/v1/benchmarks/{pending}/verify", json=REVIEW).status_code == 403  # no key, no verification
    assert c.post(f"/api/v1/benchmarks/{pending}/verify", json=REVIEW, headers=ADMIN).json()["status"] == "verified"
    assert status(pending) == "verified" and pending in explorer(status="verified") and pending not in explorer(status="community")
    row = next(r for r in c.get("/api/v1/benchmarks", params={"status": "verified"}).json()["results"] if r["id"] == pending)
    assert row["verified_by"].startswith("E2E Reviewer") and row["source_url"] == RUN["source_url"]
    assert trusted_ids_for_gpu() == [50]
    cal = calibration()
    assert cal["workload_match"] == "context_and_concurrency" and cal["tier"] == "this GPU" and cal["runs"] == 1 and cal["level"] == "thin"
    assert abs(cal["utilization"] - 50 / CEIL) < 1e-9  # the estimate is now exactly this run's measured ratio, not the seeded run's

    # 4. everything that is NOT verified is shut out, however extreme: community, rejected by validation, rejected by a reviewer, still pending
    wild = submit(context_tokens=5001, tok_s_per_user=140)  # community, never verified
    impossible = submit(gpu="Imaginary 9000", context_tokens=5002)  # rejected by validation
    pending2 = benchmarks.submit(BenchmarkSubmission(**{**RUN, "context_tokens": 5003, "tok_s_per_user": 150}).model_dump())  # never processed
    reviewed_out = submit(context_tokens=5004, tok_s_per_user=160)
    c.post(f"/api/v1/benchmarks/{reviewed_out}/verify", json=REVIEW, headers=ADMIN)
    assert 160 in trusted_ids_for_gpu()  # while verified it counts, as it should
    rej = {**REVIEW, "reason": "Source shows a different batch size.", "notes": "The linked log was run at 1 user, not 7."}
    assert c.post(f"/api/v1/benchmarks/{reviewed_out}/reject", json=rej, headers=ADMIN).json()["status"] == "rejected"
    assert (status(wild), status(impossible), status(pending2), status(reviewed_out)) == ("community", "rejected", "pending", "rejected")
    assert trusted_ids_for_gpu() == [50]  # only the one verified run survives
    after = calibration()
    assert after["runs"] == 1 and abs(after["utilization"] - 50 / CEIL) < 1e-9 and after["low"] is None  # not one of the four moved it
    assert {wild, impossible, pending2, reviewed_out}.isdisjoint(explorer(status="verified"))
    assert impossible not in explorer() and pending2 not in explorer() and reviewed_out not in explorer()  # rejected and pending runs are not public
    assert any("not in the catalog" in p for p in c.get(f"/api/v1/benchmarks/{impossible}").json()["problems"])  # but their submitter can see why

    # 5. a reviewer can withdraw the verified run: calibration reverts at once
    assert c.post(f"/api/v1/benchmarks/{pending}/reject", json=rej, headers=ADMIN).json()["status"] == "rejected"
    assert trusted_ids_for_gpu() == [] and calibration() == base
    assert [r["action"] for r in c.get(f"/api/v1/benchmarks/{pending}").json()["reviews"]] == ["verify", "reject"]  # the history stays


def test_seeded_published_runs_stay_trusted_throughout():
    ids = explorer(status="sourced")
    assert len(ids) >= 5 and all(i in [r["id"] for r in c.get("/api/v1/benchmarks").json()["results"]] for i in ids)
    assert any(r["gpu"] == GPU and r["model"] == "Llama-2-7B" for r in benchmarks.trusted_runs())  # the published 3090 run feeds calibration
