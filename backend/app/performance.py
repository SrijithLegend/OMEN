"""Performance Intelligence: decode ceiling, prefill/TTFT, TPOT, calibration from measured runs."""
from statistics import median

from .catalog import GPUS, MODELS, QUANTS
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


MIN_RUNS = 3  # ponytail: fewer supporting runs than this on one GPU is labelled "thin"; raise it as the dataset grows


def family(engine: str | None) -> str:
    return (engine or "").split()[0].lower()  # "llama.cpp b3900" -> "llama.cpp": builds differ, engines are what matter here


def calibrate(runs: list[dict], *, gpu: str, engine: str | None = None, context: int, users: int) -> dict:
    """Real/ceiling ratio from measured runs, from the most specific evidence available: this GPU (and engine) first, other GPUs only as a flagged
    extrapolation. Never one global factor presented as reliable: the result carries how many runs back it, how far they spread, and what they covered."""
    rows = []
    for r in runs:
        m, q, g = MODELS[r["model"]], QUANTS[r["quant"]], GPUS[r["gpu"]]
        rows.append((r, r["tok_s"] / decode_ceiling(m["active_params_b"] or m["params_b"], q["bytes_per_param"], g["bandwidth_gbs"], r.get("gpu_count", 1))))
    same_gpu, same_engine = (lambda r: r["gpu"] == gpu), (lambda r: family(r["engine"]) == family(engine))
    tiers = ([("this GPU and engine", lambda r: same_gpu(r) and same_engine(r)), ("this GPU, other engines", same_gpu),
              ("other GPUs, same engine", same_engine), ("other GPUs and engines", lambda r: True)] if engine else
             [("this GPU", same_gpu), ("other GPUs", lambda r: True)])
    for tier, ok in tiers:
        got = [(r, x) for r, x in rows if ok(r)]
        if got:
            break
    else:
        return {"tier": None, "level": "none", "runs": 0, "utilization": None, "low": None, "high": None, "caveats": ["No measured runs: speed is a bandwidth ceiling only."]}
    xs, rs = [x for _, x in got], [r for r, _ in got]
    level = "extrapolated" if not tier.startswith("this GPU") else "gpu_specific" if len(rs) >= MIN_RUNS and tier != "this GPU, other engines" else "thin"
    matched = sum(1 for r in rs if r["users"] == users and 0.25 <= r["context"] / context <= 4)
    cover = {"models": sorted({r["model"] for r in rs}), "quants": sorted({r["quant"] for r in rs}), "engines": sorted({family(r["engine"]) for r in rs}),
             "contexts": sorted({r["context"] for r in rs}), "users": sorted({r["users"] for r in rs})}
    caveats = []
    if level == "extrapolated":
        caveats.append(f"No measured run on {gpu}{' with this engine' if engine else ''}: this borrows the ratio from {len(rs)} run(s) on other hardware and can be far off.")
    elif level == "thin":
        caveats.append(f"Only {len(rs)} measured run(s) on this GPU (need {MIN_RUNS}+ for a gpu_specific label): treat the speed as a lead, not a prediction.")
    if tier == "this GPU, other engines":
        caveats.append("Measured with a different engine than requested; engines differ in kernel efficiency.")
    if not matched:
        caveats.append(f"None of the supporting runs match your conditions (context {context}, {users} user(s)); they used context {cover['contexts']}, users {cover['users']}.")
    if len(cover["models"]) == 1 or len(cover["quants"]) == 1:
        caveats.append(f"Supporting runs cover {len(cover['models'])} model(s) and {len(cover['quants'])} quantization(s): the ratio may not transfer to others.")
    return {"tier": tier, "level": level, "runs": len(rs), "utilization": median(xs), "low": min(xs) if len(xs) > 1 else None, "high": max(xs) if len(xs) > 1 else None,
            "conditions_matched": matched, "coverage": cover, "caveats": caveats}


def estimate_performance(model: dict, quant: dict, gpu: dict, *, n_gpus: int = 1, input_tokens: int, output_tokens: int,
                         utilization: float | None = None, runs: list[dict], gpu_name: str, engine: str | None = None,
                         context: int = 8192, users: int = 1, kv_total_gib: float = 0, weights_gib: float = 0) -> dict:
    active = model["active_params_b"] or model["params_b"]
    ceiling = decode_ceiling(active, quant["bytes_per_param"], gpu["bandwidth_gbs"], n_gpus)
    cal = calibrate(runs, gpu=gpu_name, engine=engine, context=context, users=users)
    util = cal["utilization"] if utilization is None else utilization
    tok = None if util is None else ceiling * util
    rng = [ceiling * cal["low"], ceiling * cal["high"]] if utilization is None and cal["low"] is not None else None
    ttft = ttft_seconds(active, input_tokens, gpu["tflops_fp16"], n_gpus)
    t = tok or ceiling
    return {
        "decode_ceiling_tok_s": ceiling, "utilization_used": util, "utilization_source": "user" if utilization is not None else cal["tier"],
        "decode_tok_s": tok, "decode_tok_s_range": rng, "tpot_ms": None if tok is None else 1000 / tok, "ttft_ms": None if ttft is None else ttft * 1000,
        "end_to_end_s": None if ttft is None or tok is None else ttft + output_tokens / tok,
        # the weights-only ceiling ignores the KV cache read every step; this is the share of bytes per step it leaves out
        "kv_read_fraction": kv_total_gib / (weights_gib + kv_total_gib) if weights_gib + kv_total_gib else 0,
        "calibration": cal,
        "bottleneck": "memory_bandwidth" if ttft is None or ttft < output_tokens / t else "compute_prefill",
    }
