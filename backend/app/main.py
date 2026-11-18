"""OMEN API. Run from the backend folder: uvicorn app.main:app --reload   (interactive docs at /docs)."""
import os
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import benchmarks, db
from .catalog import GPUS, MODELS, QUANTS, RUNS, get
from .memory import estimate_memory
from .performance import estimate_performance
from .recommend import candidates, recommend
from .workload import HOURS_PER_MONTH, break_even_months, self_hosted_usd_hr, simulate

@asynccontextmanager
async def lifespan(_):
    db.init(RUNS)
    yield


app = FastAPI(title="OMEN Inference Intelligence", version="0.1.0", lifespan=lifespan)


def _measured(model: str, quant: str, gpu: str) -> int:
    return sum(1 for r in RUNS if (r["model"], r["quant"], r["gpu"]) == (model, quant, gpu))


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


class PerfRequest(Config):
    input_tokens: int = Field(2000, ge=1)
    output_tokens: int = Field(512, ge=1)
    utilization: float | None = Field(None, gt=0, le=1)


def _evidence(c: Config) -> dict:
    n = _measured(c.model, c.quantization, c.gpu)
    return {"type": "measured_plus_estimate" if n else "model_estimate", "benchmark_count": n}


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


@app.post("/api/v1/estimate/memory")
def memory(c: Config):
    ev = _evidence(c)
    r = estimate_memory(_lookup(MODELS, c.model, "model"), _lookup(QUANTS, c.quantization, "quantization"), _lookup(GPUS, c.gpu, "GPU"),
                        n_gpus=c.gpu_count, context=c.context_length, users=c.concurrency, overhead_gib=c.overhead_gib,
                        measured=ev["benchmark_count"] > 0)
    return {"status": "estimated", "memory": r, "evidence": ev, "warnings": ["KV cache assumed fp16.", "Overhead is an assumption, not a measurement."]}


class WorkloadRequest(PerfRequest):
    requests_per_month: int = Field(100_000, ge=1)


def _simulate(c: WorkloadRequest) -> dict:
    mem = memory(c)["memory"]
    if not mem["fits"]:
        return {"status": "does_not_fit", "memory": mem}
    perf = performance(c)["performance"]
    w = simulate(perf["decode_ceiling_tok_s"], c.concurrency, _lookup(GPUS, c.gpu, "GPU")["usd_hr"] * c.gpu_count,
                 c.output_tokens, c.requests_per_month, perf["utilization_used"])
    return {"status": "estimated", "memory": mem, "performance": perf, "workload": w}


@app.post("/api/v1/workloads/simulate")
def simulate_workload(c: WorkloadRequest):
    return {**_simulate(c), "evidence": _evidence(c), "warnings": ["Ceiling-based: ignores queueing, so treat latency as a best case."]}


class CostRequest(WorkloadRequest):
    hardware_usd: float | None = Field(None, gt=0)
    lifetime_months: float = Field(36, gt=0)
    power_watts: float = Field(350, ge=0)
    usd_per_kwh: float = Field(0.15, ge=0)
    hosting_usd_month: float = Field(0, ge=0)
    utilization: float | None = Field(None, gt=0, le=1)


@app.post("/api/v1/cost/estimate")
def cost(c: CostRequest):
    s = _simulate(c)
    if s["status"] != "estimated":
        return s
    out = {**s, "evidence": _evidence(c), "warnings": ["Cloud price is an assumed $/hr, not a quote."]}
    if c.hardware_usd:
        w = s["workload"]
        own = self_hosted_usd_hr(c.hardware_usd * c.gpu_count, c.lifetime_months, c.power_watts * c.gpu_count, c.usd_per_kwh, c.hosting_usd_month)
        cloud = _lookup(GPUS, c.gpu, "GPU")["usd_hr"] * c.gpu_count
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
    max_usd_hr: float | None = Field(None, gt=0)
    min_tok_s: float = Field(10, ge=0)
    max_quality_loss_pct: float = Field(5, ge=0)


@app.post("/api/v1/recommendations")
def recommendations(r: RecommendRequest):
    _lookup(MODELS, r.model, "model")
    cands = candidates(r.model, n_gpus=r.gpu_count, context=r.context_length, users=r.concurrency, overhead_gib=r.overhead_gib)
    return {"combinations_tried": len(cands), **recommend(cands, max_usd_hr=r.max_usd_hr, min_tok_s=r.min_tok_s, max_quality_loss_pct=r.max_quality_loss_pct),
            "warnings": ["Estimates only: speeds are ceilings. Benchmark before buying."]}


@app.post("/api/v1/estimate/performance")
def performance(c: PerfRequest):
    gpu = _lookup(GPUS, c.gpu, "GPU")
    r = estimate_performance(_lookup(MODELS, c.model, "model"), _lookup(QUANTS, c.quantization, "quantization"), gpu,
                             n_gpus=c.gpu_count, input_tokens=c.input_tokens, output_tokens=c.output_tokens, utilization=c.utilization)
    warnings = ["Calibration comes from one model on one engine: low confidence elsewhere."]
    if r["ttft_ms"] is None:
        warnings.append("No compute figure for this GPU: prefill/TTFT is not estimated.")
    return {"status": "estimated", "performance": r, "evidence": _evidence(c), "warnings": warnings}


class BenchmarkSubmission(BaseModel):
    model: str
    quant: str
    gpu: str
    engine: str = Field(min_length=1)
    context_tokens: int = Field(ge=1)
    users: int = Field(1, ge=1)
    tok_s_per_user: float = Field(gt=0)
    ttft_ms: float | None = Field(None, gt=0)
    tpot_ms: float | None = Field(None, gt=0)
    vram_gib: float | None = Field(None, gt=0)
    source_url: str | None = None
    submitter: str | None = None
    consent_to_publish: bool


@app.post("/api/v1/benchmarks/submit", status_code=202)
def submit_benchmark(b: BenchmarkSubmission, background: BackgroundTasks):
    """Needs explicit consent. Validation runs in the background; poll GET /benchmarks?status=... or the returned id."""
    if not b.consent_to_publish:
        raise HTTPException(400, "consent_to_publish must be true: OMEN never publishes a result without it")
    run_id = benchmarks.submit(b.model_dump())
    background.add_task(benchmarks.process, run_id)
    return {"id": run_id, "status": "pending"}


@app.get("/api/v1/benchmarks")
def search_benchmarks(model: str | None = None, quant: str | None = None, gpu: str | None = None, engine: str | None = None, status: str | None = None):
    return {"results": benchmarks.search(model=model, quant=quant, gpu=gpu, engine=engine, status=status)}


@app.post("/api/v1/benchmarks/{run_id}/verify")
def verify_benchmark(run_id: int, x_admin_key: str | None = Header(None)):
    """Reviewer action. Needs OMEN_ADMIN_KEY set on the server; promotes a Community run to Verified."""
    if not os.environ.get("OMEN_ADMIN_KEY") or x_admin_key != os.environ["OMEN_ADMIN_KEY"]:
        raise HTTPException(403, "reviewer key required")
    if not benchmarks.verify(run_id):
        raise HTTPException(409, "only community runs can be verified")
    return {"id": run_id, "status": "verified"}
