"""Performance Intelligence: decode ceiling, prefill/TTFT, TPOT, calibration from measured runs."""
from statistics import median

from .catalog import GPUS, MODELS, QUANTS, RUNS
from .memory import TP_EFF

MFU = 0.4  # ponytail: typical large-batch fraction of peak FLOPs for prefill; uncalibrated (no measured prefill data yet)


def decode_ceiling(active_params_b: float, bytes_per_param: float, bandwidth_gbs: float, n_gpus: int = 1) -> float:
    """tokens/s upper bound for one user: memory bandwidth / weight bytes read per token."""
    bw = bandwidth_gbs * n_gpus * (TP_EFF if n_gpus > 1 else 1)
    return bw * 1e9 / (active_params_b * 1e9 * bytes_per_param)


def ttft_seconds(active_params_b: float, input_tokens: int, tflops: float | None, n_gpus: int = 1) -> float | None:
    """Prefill is compute-bound: ~2 FLOPs per active parameter per prompt token. None when the GPU's compute is unknown."""
    if not tflops:
        return None
    return 2 * active_params_b * 1e9 * input_tokens / (tflops * 1e12 * (TP_EFF * n_gpus if n_gpus > 1 else 1) * MFU)


def calibration() -> dict:
    """Real/ceiling ratio over the published runs. One model, one engine: low confidence outside that."""
    ratios = []
    for r in RUNS:
        m, q, g = MODELS[r["model"]], QUANTS[r["quant"]], GPUS[r["gpu"]]
        ratios.append(r["tok_s"] / decode_ceiling(m["active_params_b"] or m["params_b"], q["bytes_per_param"], g["bandwidth_gbs"]))
    return {"median_ratio": median(ratios), "min": min(ratios), "max": max(ratios), "runs": len(ratios)}


def estimate_performance(model: dict, quant: dict, gpu: dict, *, n_gpus: int = 1, input_tokens: int, output_tokens: int,
                         utilization: float | None = None) -> dict:
    active = model["active_params_b"] or model["params_b"]
    ceiling = decode_ceiling(active, quant["bytes_per_param"], gpu["bandwidth_gbs"], n_gpus)
    cal = calibration()
    util = cal["median_ratio"] if utilization is None else utilization
    tok = ceiling * util
    ttft = ttft_seconds(active, input_tokens, gpu["tflops_fp16"], n_gpus)
    return {
        "decode_ceiling_tok_s": ceiling, "utilization_used": util, "decode_tok_s": tok,
        "tpot_ms": 1000 / tok, "ttft_ms": None if ttft is None else ttft * 1000,
        "end_to_end_s": None if ttft is None else ttft + output_tokens / tok,
        "calibration": cal,
        "bottleneck": "memory_bandwidth" if ttft is None or ttft < output_tokens / tok else "compute_prefill",
    }
