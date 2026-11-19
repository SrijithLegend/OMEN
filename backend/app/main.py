"""OMEN API. Run from the backend folder: uvicorn app.main:app --reload   (interactive docs at /docs)."""
import hashlib
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import delete, insert, select, update
from pydantic import BaseModel, Field

from . import benchmarks, db, prices
from .catalog import CATALOG_BYTES, GPUS, MODELS, QUANTS, RUNS, get
from .evidence import evidence
from .memory import KV_BYTES, estimate_memory
from .performance import estimate_performance
from .recommend import HIGH_CONFIDENCE, candidates, recommend
from .workload import HOURS_PER_MONTH, break_even_months, self_hosted_usd_hr, simulate

@asynccontextmanager
async def lifespan(_):
    db.init(RUNS)
    yield


app = FastAPI(title="OMEN Inference Intelligence", version="0.1.0", lifespan=lifespan)
if os.environ.get("OMEN_ENV") == "production" and not os.environ.get("OMEN_CORS"):
    raise RuntimeError("OMEN_ENV=production needs OMEN_CORS set to your site's origin (comma-separated). '*' is only the default for local development.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("OMEN_CORS", "*").split(","), allow_methods=["*"], allow_headers=["*"])  # restrict with OMEN_CORS=https://your.site

MAX_BODY_BYTES = 64_000  # ponytail: trusts Content-Length, so a chunked body without one is only bounded by field limits; put a proxy limit in front for hostile traffic


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
        return JSONResponse({"detail": f"payload larger than {MAX_BODY_BYTES} bytes"}, status_code=413)
    return await call_next(request)


REQUESTS = Counter("omen_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("omen_http_request_seconds", "HTTP request latency", ["route"])


@app.middleware("http")
async def metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    route = getattr(request.scope.get("route"), "path", "unmatched")  # the route template, so labels stay low-cardinality
    REQUESTS.labels(request.method, route, response.status_code).inc()
    LATENCY.labels(route).observe(time.perf_counter() - start)
    return response


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _lookup(table, name, kind):
    try:
        return get(table, name, kind)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0]))


class Config(BaseModel):
    model: str
    quantization: str
    gpu: str
    gpu_count: int = Field(1, ge=1, le=64)
    context_length: int = Field(8192, ge=1)
    concurrency: int = Field(1, ge=1)
    overhead_gib: float = Field(1.5, ge=0)
    kv_dtype: str = Field("fp16", pattern="^(" + "|".join(KV_BYTES) + ")$")  # KV cache element type
    sliding_cache: bool = False  # engine frees KV for sliding-window layers (only matters for models that have them)
    engine: str | None = None  # narrows evidence and calibration to this engine, e.g. "vllm" or "llama.cpp"


class PerfRequest(Config):
    input_tokens: int = Field(2000, ge=1)
    output_tokens: int = Field(512, ge=1)
    utilization: float | None = Field(None, gt=0, le=1)


def _evidence(c: Config) -> dict:
    return evidence(benchmarks.trusted_runs(), model=c.model, quant=c.quantization, gpu=c.gpu, vram_gib=_lookup(GPUS, c.gpu, "GPU")["vram_gib"] * c.gpu_count,
                    n_gpus=c.gpu_count, context=c.context_length, users=c.concurrency, kv_dtype=c.kv_dtype, engine=c.engine)


@app.get("/api/v1/models")
def list_models(use_case: str | None = None):
    return {n: m for n, m in MODELS.items() if not use_case or use_case in m["use_cases"]}


@app.get("/api/v1/models/{model_id}")
def get_model(model_id: str):
    return _lookup(MODELS, model_id, "model")


@app.get("/api/v1/gpus")
def list_gpus():
    return GPUS


@app.get("/api/v1/gpus/{gpu_id}")
def get_gpu(gpu_id: str):
    return _lookup(GPUS, gpu_id, "GPU")


@app.get("/api/v1/quantizations")
def list_quants():
    return QUANTS


def _spec_warning(gpu: str) -> list[str]:
    st = _lookup(GPUS, gpu, "GPU")["spec_status"]
    return [] if st == "verified" else [f"GPU specs for {gpu} are {'only partly' if st == 'partial' else 'not'} verified against a vendor source (see /gpus/{gpu}): VRAM, bandwidth and TFLOPS may be off."]


@app.post("/api/v1/estimate/memory")
def memory(c: Config):
    ev = _evidence(c)
    r = estimate_memory(_lookup(MODELS, c.model, "model"), _lookup(QUANTS, c.quantization, "quantization"), _lookup(GPUS, c.gpu, "GPU"),
                        n_gpus=c.gpu_count, context=c.context_length, users=c.concurrency, overhead_gib=c.overhead_gib,
                        memory_measured=ev["memory_measured"] > 0, kv_dtype=c.kv_dtype, sliding_cache=c.sliding_cache)
    warnings = ["Overhead is an assumption, not a measurement."] + _spec_warning(c.gpu)
    if not ev["memory_measured"]:
        warnings.append("No recorded peak-VRAM measurement for this setup at your context and concurrency: fit is an estimate" + (" (a speed benchmark exists, but it does not show memory)." if ev["performance_measured"] else "."))
    if c.kv_dtype in ("fp8", "int8"):
        warnings.append(f"{c.kv_dtype} KV cache assumes per-tensor scales and OMEN has not measured its quality impact.")
    if r["kv_gib_per_user_if_sliding_freed"] is not None:
        warnings.append(f"This model has sliding-window layers: the figure counts a full cache. An engine that frees them needs {r['kv_gib_per_user_if_sliding_freed']:.2f} GiB per user (set sliding_cache).")
    if c.sliding_cache and not r["sliding_window_applied"]:
        warnings.append("sliding_cache ignored: this model's config does not declare sliding-window layers.")
    return {"status": "estimated", "memory": r, "evidence": ev, "warnings": warnings}


class WorkloadRequest(PerfRequest):
    requests_per_month: int = Field(100_000, ge=1)


def _simulate(c: WorkloadRequest, usd_hr_per_gpu: float | None = None) -> dict:
    mem = memory(c)["memory"]
    if not mem["fits"]:
        return {"status": "does_not_fit", "memory": mem}
    perf = performance(c)["performance"]
    w = simulate(perf["decode_ceiling_tok_s"], c.concurrency, (usd_hr_per_gpu if usd_hr_per_gpu is not None else _lookup(GPUS, c.gpu, "GPU")["usd_hr"]) * c.gpu_count,
                 c.output_tokens, c.requests_per_month, perf["utilization_used"] if perf["utilization_used"] is not None else 1.0)  # no calibration: ceiling, a best case
    return {"status": "estimated", "memory": mem, "performance": perf, "workload": w}


@app.post("/api/v1/workloads/simulate")
def simulate_workload(c: WorkloadRequest):
    s = _simulate(c)
    return {**s, "evidence": _evidence(c), "warnings": ["Ceiling-based: ignores queueing, so treat latency as a best case."] + s.get("performance", {}).get("calibration", {}).get("caveats", [])}


class CostRequest(WorkloadRequest):
    hardware_usd: float | None = Field(None, gt=0)
    lifetime_months: float = Field(36, gt=0)
    power_watts: float = Field(350, ge=0)
    usd_per_kwh: float = Field(0.15, ge=0)
    hosting_usd_month: float = Field(0, ge=0)
    utilization: float | None = Field(None, gt=0, le=1)
    provider: str | None = None  # price from the recorded catalog (see /prices) instead of the assumed $/hr
    pricing_model: str = "on_demand"


@app.post("/api/v1/cost/estimate")
def cost(c: CostRequest):
    rec = None
    if c.provider:
        rec = prices.cheapest_per_gpu(c.provider, c.gpu, c.pricing_model)
        if not rec:
            raise HTTPException(404, f"no recorded {c.pricing_model} price for {c.gpu} at {c.provider}: add one with POST /api/v1/prices, or omit provider to use the assumed price")
    s = _simulate(c, rec["usd_hr_per_gpu"] if rec else None)
    if s["status"] != "estimated":
        return s
    warn = f"Price recorded from {rec['source_url']} on {rec['observed_at']}: it may have changed." if rec else "Cloud price is an assumed $/hr, not a quote."
    out = {**s, "evidence": _evidence(c), "warnings": [warn]}
    if rec:
        out["price_used"] = rec
    if c.hardware_usd:
        w = s["workload"]
        own = self_hosted_usd_hr(c.hardware_usd * c.gpu_count, c.lifetime_months, c.power_watts * c.gpu_count, c.usd_per_kwh, c.hosting_usd_month)
        cloud = (rec["usd_hr_per_gpu"] if rec else _lookup(GPUS, c.gpu, "GPU")["usd_hr"]) * c.gpu_count
        out["self_hosted"] = {"usd_hr_while_running": own, "monthly_usd": own * HOURS_PER_MONTH * w["copies_needed"],
                              "break_even_months": break_even_months(c.hardware_usd * c.gpu_count, cloud, own, w["gpu_hours_per_month"])}
    return out


class CompareRequest(BaseModel):
    configurations: list[WorkloadRequest] = Field(min_length=2, max_length=8)


@app.post("/api/v1/configurations/compare")
def compare(r: CompareRequest):
    """Every configuration is evaluated with its own stated workload; the response lists the workloads so differences are visible."""
    rows = [{"config": c.model_dump(), **_simulate(c), "evidence": _evidence(c)} for c in r.configurations]
    keys = {(c.input_tokens, c.output_tokens, c.concurrency, c.context_length, c.requests_per_month) for c in r.configurations}
    return {"results": rows, "warnings": [] if len(keys) == 1 else ["These configurations use different workloads: do not rank them directly."]}


class RecommendRequest(BaseModel):
    model: str
    gpu_count: int = Field(1, ge=1, le=64)
    context_length: int = Field(8192, ge=1)
    concurrency: int = Field(1, ge=1)
    overhead_gib: float = Field(1.5, ge=0)
    kv_dtype: str = Field("fp16", pattern="^(" + "|".join(KV_BYTES) + ")$")
    max_usd_hr: float | None = Field(None, gt=0)
    min_tok_s: float = Field(10, ge=0)
    max_quality_loss_pct: float = Field(5, ge=0)
    min_confidence: float = Field(0, ge=0, le=1)  # 0.75 keeps only high-confidence picks


@app.post("/api/v1/recommendations")
def recommendations(r: RecommendRequest):
    _lookup(MODELS, r.model, "model")
    cands = candidates(benchmarks.trusted_runs(), r.model, n_gpus=r.gpu_count, context=r.context_length, users=r.concurrency, overhead_gib=r.overhead_gib, kv_dtype=r.kv_dtype)
    return {"combinations_tried": len(cands), **recommend(cands, max_usd_hr=r.max_usd_hr, min_tok_s=r.min_tok_s, max_quality_loss_pct=r.max_quality_loss_pct, min_confidence=r.min_confidence),
            "warnings": ["Estimates only: speeds are ceilings. Benchmark before buying.", f"Confidence is high at {HIGH_CONFIDENCE:.0%} or above; below that treat the pick as a lead to benchmark."]}


@app.post("/api/v1/estimate/performance")
def performance(c: PerfRequest):
    gpu = _lookup(GPUS, c.gpu, "GPU")
    mem = memory(c)["memory"]
    r = estimate_performance(_lookup(MODELS, c.model, "model"), _lookup(QUANTS, c.quantization, "quantization"), gpu,
                             n_gpus=c.gpu_count, input_tokens=c.input_tokens, output_tokens=c.output_tokens, utilization=c.utilization,
                             runs=benchmarks.trusted_runs(), gpu_name=c.gpu, engine=c.engine, context=c.context_length, users=c.concurrency,
                             kv_total_gib=mem["kv_gib_total"], weights_gib=mem["weights_gib"])
    warnings = list(r["calibration"]["caveats"]) + _spec_warning(c.gpu)
    if r["decode_tok_s"] is None:
        warnings.append("No calibration available: only the bandwidth ceiling is reported, not an expected speed.")
    if r["kv_read_fraction"] > 0.1:
        warnings.append(f"The KV cache is {r['kv_read_fraction']:.0%} of the bytes read per step; the weights-only ceiling ignores it, so real speed will be lower.")
    if r["ttft_ms"] is None:
        warnings.append("No compute figure for this GPU: prefill/TTFT is not estimated.")
    return {"status": "estimated", "performance": r, "evidence": _evidence(c), "warnings": warnings}


class ReferenceRun(BaseModel):
    """A sourced reference measurement. Every field is bounded and the source is required."""
    model: str = Field(max_length=100)
    quant: str = Field(max_length=100)
    gpu: str = Field(max_length=100)
    engine: str = Field(min_length=1, max_length=100)
    context_tokens: int = Field(ge=1, le=10_000_000)
    users: int = Field(1, ge=1, le=10_000)
    tok_s_per_user: float = Field(gt=0, le=100_000)
    ttft_ms: float | None = Field(None, gt=0, le=3_600_000)
    tpot_ms: float | None = Field(None, gt=0, le=3_600_000)
    vram_gib: float | None = Field(None, gt=0, le=100_000)  # peak VRAM across all GPUs; without it the run says nothing about memory
    vram_gib_per_gpu: list[Annotated[float, Field(gt=0, le=100_000)]] | None = Field(None, max_length=64)  # each GPU's own peak; must match gpu_count and sum to vram_gib
    gpu_count: int = Field(1, ge=1, le=64)
    kv_dtype: str = Field("fp16", pattern="^(" + "|".join(KV_BYTES) + ")$")
    token_count_method: str | None = Field(None, pattern="^(usage|tokenizer|chunks)$")  # how tokens/s was counted; chunks is an estimate
    source_url: str = Field(pattern=r"^https?://\S{1,500}$")


def _admin(key: str | None) -> None:
    """Maintainer actions (importing a reference run, entering a price) need OMEN_ADMIN_KEY. Unset means nobody can."""
    admin = os.environ.get("OMEN_ADMIN_KEY")
    if not (key and admin and secrets.compare_digest(key, admin)):
        raise HTTPException(403, "maintainer key required")


@app.post("/api/v1/benchmarks/import", status_code=201)
def import_benchmark(b: ReferenceRun, x_admin_key: str | None = Header(None)):
    """Maintainer action: store a sourced reference run (for instance a result file from omen_bench.py plus the link that backs it). The run is validated
    first and refused with the reasons if it is implausible or repeats a stored run. There is no public submission."""
    _admin(x_admin_key)
    try:
        return {"id": benchmarks.import_run(b.model_dump())}
    except benchmarks.Rejected as e:
        raise HTTPException(409 if e.problems[0].startswith("duplicate") else 422, e.problems)


@app.get("/api/v1/benchmarks")
def search_benchmarks(model: str | None = None, quant: str | None = None, gpu: str | None = None, engine: str | None = None):
    return {"results": benchmarks.search(model=model, quant=quant, gpu=gpu, engine=engine)}


@app.get("/api/v1/benchmarks/{run_id}")
def get_benchmark(run_id: int):
    r = benchmarks.get(run_id)
    if r is None:
        raise HTTPException(404, "no such run")
    return r


class PriceRecord(BaseModel):
    provider: str
    instance_type: str = Field(min_length=1)
    region: str = Field(min_length=1)
    gpu: str
    gpu_count: int = Field(ge=1)
    pricing_model: str = "on_demand"
    usd_hr: float = Field(gt=0)
    source_url: str = Field(min_length=8)
    observed_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


@app.post("/api/v1/prices", status_code=201)
def add_price(p: PriceRecord, x_admin_key: str | None = Header(None)):
    """Reviewer action (same key as /verify). Every price needs a source URL and the date it was read; rows are never overwritten."""
    _admin(x_admin_key)
    if p.provider not in prices.PROVIDERS or p.pricing_model not in prices.PRICING_MODELS:
        raise HTTPException(422, f"provider must be one of {prices.PROVIDERS}, pricing_model one of {prices.PRICING_MODELS}")
    _lookup(GPUS, p.gpu, "GPU")
    return {"id": prices.add(p.model_dump())}


@app.get("/api/v1/prices")
def list_prices(provider: str | None = None, gpu: str | None = None, pricing_model: str | None = None, history: bool = False):
    """Newest price per instance by default; history=true returns every recorded observation."""
    return {"providers": prices.PROVIDERS, "prices": prices.search(provider=provider, gpu=gpu, pricing_model=pricing_model, latest_only=not history)}


# ---- API keys, saved workspaces, change tracking -------------------------------------------------------------------
RATE_LIMIT_PER_MINUTE = 60
_hash = lambda s: hashlib.sha256(s.encode()).hexdigest()
CATALOG_HASH = hashlib.sha256(CATALOG_BYTES).hexdigest()[:12]


class KeyRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")


@app.post("/api/v1/keys", status_code=201)
def create_key(k: KeyRequest):
    """Returns the key once; only its hash is stored. Self-hosted default: anyone can create one. Put this behind your own gate in production."""
    key = "omen_" + secrets.token_urlsafe(24)
    with db.engine.begin() as c:
        if c.execute(select(db.api_keys.c.id).where(db.api_keys.c.owner == k.owner)).first():
            raise HTTPException(409, "owner already has a key")
        c.execute(insert(db.api_keys).values(owner=k.owner, key_hash=_hash(key), calls_this_minute=0, minute=0))
    return {"owner": k.owner, "api_key": key, "note": "Store it now: it cannot be shown again."}


def current_owner(x_api_key: str | None = Header(None)) -> str:
    """Authenticates by API key and enforces a per-minute rate limit."""
    if not x_api_key:
        raise HTTPException(401, "X-API-Key header required")
    minute = int(time.time() // 60)
    with db.engine.begin() as c:
        row = c.execute(select(db.api_keys).where(db.api_keys.c.key_hash == _hash(x_api_key))).mappings().first()
        if not row:
            raise HTTPException(401, "unknown API key")
        used = row["calls_this_minute"] + 1 if row["minute"] == minute else 1
        if used > RATE_LIMIT_PER_MINUTE:
            raise HTTPException(429, f"rate limit: {RATE_LIMIT_PER_MINUTE} calls per minute")
        c.execute(update(db.api_keys).where(db.api_keys.c.id == row["id"]).values(calls_this_minute=used, minute=minute))
    return row["owner"]


KINDS = {"estimate": (Config, lambda x: memory(x)), "workload": (WorkloadRequest, lambda x: simulate_workload(x)),
         "recommendation": (RecommendRequest, lambda x: recommendations(x))}


class SaveRequest(BaseModel):
    kind: str
    input: dict
    note: str = ""


def _run(kind: str, payload: dict) -> dict:
    if kind not in KINDS:
        raise HTTPException(422, f"kind must be one of {sorted(KINDS)}")
    model, fn = KINDS[kind]
    return fn(model(**payload))


@app.post("/api/v1/workspaces/{project}/analyses", status_code=201)
def save_analysis(project: str, s: SaveRequest, owner: str = Depends(current_owner)):
    """Runs the analysis now and stores the exact input, the output and the catalog version it used."""
    out = _run(s.kind, s.input)
    with db.engine.begin() as c:
        i = c.execute(insert(db.saved_analyses).values(owner=owner, project=project, note=s.note, kind=s.kind, input_snapshot=s.input,
                                                      output_snapshot=out, catalog_hash=CATALOG_HASH)).inserted_primary_key[0]
    return {"id": i, "catalog_hash": CATALOG_HASH, "output": out}


@app.get("/api/v1/workspaces/{project}/analyses")
def list_analyses(project: str, owner: str = Depends(current_owner)):
    with db.engine.connect() as c:
        rows = c.execute(select(db.saved_analyses).where((db.saved_analyses.c.owner == owner) & (db.saved_analyses.c.project == project))).mappings()
        return {"analyses": [dict(r) for r in rows]}  # only the caller's own rows: workspaces are private


@app.delete("/api/v1/workspaces/{project}/analyses/{analysis_id}")
def delete_analysis(project: str, analysis_id: int, owner: str = Depends(current_owner)):
    with db.engine.begin() as c:
        n = c.execute(delete(db.saved_analyses).where((db.saved_analyses.c.id == analysis_id) & (db.saved_analyses.c.owner == owner)
                                                      & (db.saved_analyses.c.project == project))).rowcount
    if not n:
        raise HTTPException(404, "not found")
    return {"deleted": analysis_id}


def _leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def material_changes(before: dict, after: dict, tolerance: float = 0.01) -> list[dict]:
    """Differences worth telling a user about: numbers that moved more than 1%, or any other value that changed."""
    a, b = dict(_leaves(before)), dict(_leaves(after))
    out = []
    for p in sorted(set(a) | set(b)):
        x, y = a.get(p), b.get(p)
        num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool)
        if (num and abs(x - y) > tolerance * max(abs(x), abs(y), 1e-12)) or (not num and x != y):
            out.append({"path": p, "before": x, "after": y})
    return out


@app.get("/api/v1/workspaces/{project}/changes")
def changes(project: str, owner: str = Depends(current_owner)):
    """Re-runs every saved analysis against the current catalog and code, and reports what moved. Pull-based; no email is sent."""
    with db.engine.connect() as c:
        rows = list(c.execute(select(db.saved_analyses).where((db.saved_analyses.c.owner == owner) & (db.saved_analyses.c.project == project))).mappings())
    report = []
    for r in rows:
        diff = material_changes(r["output_snapshot"], _run(r["kind"], r["input_snapshot"]))
        if diff:
            report.append({"id": r["id"], "kind": r["kind"], "note": r["note"], "saved_with_catalog": r["catalog_hash"], "now": CATALOG_HASH, "changes": diff[:20]})
    return {"checked": len(rows), "changed": report}
