"""End to end, through the HTTP API: a maintainer imports a sourced reference run, it shows up in the public explorer, and it moves the speed estimate.
A run that fails validation or repeats a stored one is refused, is never stored, and never influences anything. The public has no way to submit,
verify or reject a run."""
import os

from fastapi.testclient import TestClient

from app import benchmarks
from app.catalog import GPUS, MODELS, QUANTS
from app.main import app
from app.performance import decode_ceiling

c = TestClient(app)
c.__enter__()  # app startup: creates and seeds the database if no other test module has yet

ADMIN = {"x-admin-key": os.environ["OMEN_ADMIN_KEY"]}
GPU, MODEL, QUANT = "RTX 3090 24GB", "Llama-3.1-8B", "Q4_K_M"  # one GPU no other test touches, with one seeded run (Llama-2-7B, 1 user, 128 tokens)
ASK = {"model": MODEL, "quantization": QUANT, "gpu": GPU, "concurrency": 7, "context_length": 5000}  # a workload no seeded run matches
RUN = {"model": MODEL, "quant": QUANT, "gpu": GPU, "engine": "vllm 0.6.3", "context_tokens": 5000, "users": 7, "tok_s_per_user": 50, "source_url": "https://example.com/e2e"}
CEIL = decode_ceiling(MODELS[MODEL]["params_b"], QUANTS[QUANT]["bytes_per_param"], GPUS[GPU]["bandwidth_gbs"])


def calibration() -> dict:
    return c.post("/api/v1/estimate/performance", json=ASK).json()["performance"]["calibration"]


def explorer(**params) -> list[dict]:
    return c.get("/api/v1/benchmarks", params=params).json()["results"]


def stored_at_7_users() -> list[float]:
    return [r["tok_s"] for r in benchmarks.trusted_runs() if r["gpu"] == GPU and r["users"] == 7]


def test_import_to_explorer_to_calibration_and_what_is_refused():
    base = calibration()  # nothing measured at 7 users: the seeded single-user run is borrowed, and the answer says so
    assert base["workload_match"] == "none" and base["level"] == "extrapolated" and stored_at_7_users() == []

    # only a maintainer can import
    assert c.post("/api/v1/benchmarks/import", json=RUN).status_code == 403
    assert c.post("/api/v1/benchmarks/import", json=RUN, headers={"x-admin-key": "wrong"}).status_code == 403
    assert explorer(gpu=GPU, model=MODEL) == [] and calibration() == base  # a refused attempt changes nothing

    # a valid, sourced run: stored, public, and now the estimate's matched sample
    imp = c.post("/api/v1/benchmarks/import", json=RUN, headers=ADMIN)
    assert imp.status_code == 201
    rid = imp.json()["id"]
    (row,) = explorer(gpu=GPU, model=MODEL)
    assert row["id"] == rid and row["source_url"] == RUN["source_url"] and "status" not in row and "submitter" not in row
    assert c.get(f"/api/v1/benchmarks/{rid}").json()["tok_s_per_user"] == 50
    assert stored_at_7_users() == [50]
    cal = calibration()
    assert cal["workload_match"] == "context_and_concurrency" and cal["tier"] == "this GPU" and cal["runs"] == 1 and cal["level"] == "thin"
    assert abs(cal["utilization"] - 50 / CEIL) < 1e-9  # exactly this run's measured ratio, not the seeded run's

    # everything refused leaves no trace: not stored, not public, not in calibration
    refused = [
        ({**RUN, "gpu": "Imaginary 9000", "context_tokens": 5001}, 422, "not in the catalog"),
        ({**RUN, "users": 1, "context_tokens": 5002, "tok_s_per_user": 5000}, 422, "bandwidth ceiling"),
        ({**RUN, "context_tokens": 5003, "vram_gib": 90}, 422, "more than"),
        ({**RUN, "context_tokens": 5004, "vram_gib": 1}, 422, "weights alone"),
        ({**RUN, "tok_s_per_user": 50.4}, 409, "duplicate of run"),  # same setup, within 2%
    ]
    for body, code, why in refused:
        r = c.post("/api/v1/benchmarks/import", json=body, headers=ADMIN)
        assert r.status_code == code and why in " ".join(r.json()["detail"]), (body, r.json())
    assert c.post("/api/v1/benchmarks/import", json={k: v for k, v in RUN.items() if k != "source_url"}, headers=ADMIN).status_code == 422  # a source is mandatory
    assert c.post("/api/v1/benchmarks/import", json={**RUN, "source_url": "javascript:alert(1)"}, headers=ADMIN).status_code == 422
    assert len(explorer(gpu=GPU, model=MODEL)) == 1 and stored_at_7_users() == [50]
    after = calibration()
    assert after["runs"] == 1 and abs(after["utilization"] - 50 / CEIL) < 1e-9

    # a different setup (two GPUs, or an fp8 KV cache) is not a duplicate of the same numbers
    assert c.post("/api/v1/benchmarks/import", json={**RUN, "gpu_count": 2, "vram_gib": 20, "vram_gib_per_gpu": [10, 10]}, headers=ADMIN).status_code == 201
    assert c.post("/api/v1/benchmarks/import", json={**RUN, "kv_dtype": "fp8"}, headers=ADMIN).status_code == 201

    # rows are never edited or deleted, and there is no public path to change the data
    for method, path in [("post", "/api/v1/benchmarks/submit"), ("post", f"/api/v1/benchmarks/{rid}/verify"), ("post", f"/api/v1/benchmarks/{rid}/reject"),
                         ("delete", f"/api/v1/benchmarks/{rid}"), ("put", f"/api/v1/benchmarks/{rid}")]:
        assert c.request(method.upper(), path, headers=ADMIN, content=b"{}").status_code in (404, 405), (method, path)


def test_explorer_filters_and_the_seeded_reference_runs():
    seeded = explorer(model="Llama-2-7B")
    assert len(seeded) >= 5 and all(r["source_url"].startswith("https://") for r in seeded)
    assert {r["gpu"] for r in explorer(model="Llama-2-7B", gpu="H100 80GB SXM")} == {"H100 80GB SXM"} and explorer(model="Nope") == []
    assert any(r["gpu"] == GPU and r["model"] == "Llama-2-7B" for r in benchmarks.trusted_runs())  # the published 3090 run feeds calibration


def test_chunked_and_malformed_bodies_are_bounded_and_headers_are_set():
    chunks = (b"x" * 10_000 for _ in range(8))  # no Content-Length: sent chunked, 80 kB in total
    assert c.post("/api/v1/benchmarks/import", content=chunks, headers={**ADMIN, "content-type": "application/json"}).status_code == 413
    assert c.post("/api/v1/benchmarks/import", content="{}", headers={**ADMIN, "content-length": "abc", "content-type": "application/json"}).status_code in (400, 413)
    h = c.get("/api/v1/benchmarks").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY" and h["referrer-policy"] == "no-referrer"


def test_payloads_are_bounded():
    assert c.post("/api/v1/benchmarks/import", content="x" * 70_000, headers={**ADMIN, "content-type": "application/json"}).status_code == 413
    for patch in ({"model": "x" * 101}, {"tok_s_per_user": 1e9}, {"users": 10**9}, {"token_count_method": "guess"}, {"kv_dtype": "fp4"}):
        assert c.post("/api/v1/benchmarks/import", json={**RUN, "context_tokens": 6001, **patch}, headers=ADMIN).status_code == 422, patch
