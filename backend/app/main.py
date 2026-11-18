"""OMEN API. Run from the backend folder: uvicorn app.main:app --reload   (interactive docs at /docs)."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .catalog import GPUS, MODELS, QUANTS, RUNS, get
from .memory import estimate_memory
from .performance import estimate_performance

app = FastAPI(title="OMEN Inference Intelligence", version="0.1.0")


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


@app.post("/api/v1/estimate/performance")
def performance(c: PerfRequest):
    gpu = _lookup(GPUS, c.gpu, "GPU")
    r = estimate_performance(_lookup(MODELS, c.model, "model"), _lookup(QUANTS, c.quantization, "quantization"), gpu,
                             n_gpus=c.gpu_count, input_tokens=c.input_tokens, output_tokens=c.output_tokens, utilization=c.utilization)
    warnings = ["Calibration comes from one model on one engine: low confidence elsewhere."]
    if r["ttft_ms"] is None:
        warnings.append("No compute figure for this GPU: prefill/TTFT is not estimated.")
    return {"status": "estimated", "performance": r, "evidence": _evidence(c), "warnings": warnings}
