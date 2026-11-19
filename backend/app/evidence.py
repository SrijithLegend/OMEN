"""Evidence for a requested setup, in separate categories: a speed benchmark is not a memory measurement.
memory_measured      peak VRAM recorded for this model, quantization, GPU and GPU count at >= your context and concurrency, with a KV cache no narrower than yours
performance_measured decode speed recorded for this model, quantization, GPU and GPU count (any context or concurrency; compare them yourself)
calibrated           speed recorded on this GPU for any other setup, so the estimate's utilization rests on this hardware
estimated            none of the above: first-principles maths only
"""
from .memory import KV_BYTES
from .performance import family


def evidence(runs: list[dict], *, model: str, quant: str, gpu: str, vram_gib: float, n_gpus: int = 1, context: int, users: int,
             kv_dtype: str = "fp16", engine: str | None = None) -> dict:
    same = [r for r in runs if (r["model"], r["quant"], r["gpu"], r.get("gpu_count", 1)) == (model, quant, gpu, n_gpus) and (not engine or family(r["engine"]) == family(engine))]
    mem = [r for r in same if r.get("vram_gib") and r["vram_gib"] <= vram_gib and r["context"] >= context and r["users"] >= users
           and KV_BYTES[r.get("kv_dtype", "fp16")] >= KV_BYTES[kv_dtype]]
    cal = [r for r in runs if r["gpu"] == gpu]
    kind = "memory_measured" if mem else "performance_measured" if same else "calibrated_estimate" if cal else "model_estimate"
    return {"type": kind, "memory_measured": len(mem), "performance_measured": len(same), "calibrated_on_this_gpu": len(cal),
            "peak_vram_gib": max((r["vram_gib"] for r in mem), default=None), "engines": sorted({family(r["engine"]) for r in same})}
