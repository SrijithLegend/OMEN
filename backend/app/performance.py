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


CONTEXT_BAND = 4  # ponytail: a run counts as the same workload when its context is within 4x of yours and its user count equals yours


def calibrate(runs: list[dict], *, gpu: str, engine: str | None = None, context: int, users: int) -> dict:
    """Real/ceiling ratio from measured runs. The SAMPLE is chosen by hardware and by workload, not just described afterwards:
      1. this GPU (and engine), same concurrency and similar context     -> gpu_specific / thin
      2. this GPU, same concurrency, different context                   -> extrapolated
      3. other GPUs, same concurrency and similar context, then 4. same concurrency only -> extrapolated
      5. nothing matches your concurrency anywhere: the nearest hardware's runs as they are -> extrapolated, workload_match "none"
    Runs at other concurrency or context never dilute a better-matched sample. The result carries the sample size, the spread and what it covered."""
    rows = []
    for r in runs:
        m, q, g = MODELS[r["model"]], QUANTS[r["quant"]], GPUS[r["gpu"]]
        rows.append((r, r["tok_s"] / decode_ceiling(m["active_params_b"] or m["params_b"], q["bytes_per_param"], g["bandwidth_gbs"], r.get("gpu_count", 1))))
    same_gpu, same_engine = (lambda r: r["gpu"] == gpu), (lambda r: family(r["engine"]) == family(engine))
    tiers = ([("this GPU and engine", lambda r: same_gpu(r) and same_engine(r), True), ("this GPU, other engines", same_gpu, True),
              ("other GPUs, same engine", same_engine, False), ("other GPUs and engines", lambda r: True, False)] if engine else
             [("this GPU", same_gpu, True), ("other GPUs", lambda r: True, False)])
    same_users = lambda r: r["users"] == users
    similar_ctx = lambda r: 1 / CONTEXT_BAND <= r["context"] / context <= CONTEXT_BAND
    strata = [("context_and_concurrency", lambda r: same_users(r) and similar_ctx(r)), ("concurrency_only", same_users)]
    for local, (match, ok_w) in [(True, strata[0]), (True, strata[1]), (False, strata[0]), (False, strata[1])]:
        for tier, ok_h, is_local in tiers:
            if is_local != local:
                continue
            got = [(r, x) for r, x in rows if ok_h(r) and ok_w(r)]
            if got:
                break
        else:
            continue
        break
    else:  # no run anywhere at your concurrency: say so, and use the nearest hardware's runs unmatched
        match = "none"
        for tier, ok_h, is_local in tiers:
            got = [(r, x) for r, x in rows if ok_h(r)]
            if got:
                break
        else:
            return {"tier": None, "level": "none", "workload_match": None, "runs": 0, "runs_in_tier": 0, "conditions_matched": 0, "utilization": None, "low": None, "high": None,
                    "caveats": ["No measured runs: speed is a bandwidth ceiling only."]}
    in_tier = sum(1 for r, _ in rows if next(ok for t, ok, _ in tiers if t == tier)(r))
    xs, rs = [x for _, x in got], [r for r, _ in got]
    local_tier = tier.startswith("this GPU")
    level = ("gpu_specific" if len(rs) >= MIN_RUNS and tier != "this GPU, other engines" else "thin") if local_tier and match == "context_and_concurrency" else "extrapolated"
    cover = {"models": sorted({r["model"] for r in rs}), "quants": sorted({r["quant"] for r in rs}), "engines": sorted({family(r["engine"]) for r in rs}),
             "contexts": sorted({r["context"] for r in rs}), "users": sorted({r["users"] for r in rs})}
    caveats = []
    if match == "none":
        caveats.append(f"No measured run at {users} concurrent user(s) anywhere: this uses {len(rs)} run(s) at users {cover['users']}, context {cover['contexts']}, so the speed is an extrapolation across workloads.")
    elif match == "concurrency_only":
        caveats.append(f"No measured run at a context near {context}: this uses {len(rs)} run(s) at {users} user(s) but context {cover['contexts']}. Longer context reads more KV cache per token, so real speed is likely lower.")
    if not local_tier:
        caveats.append(f"No usable measured run on {gpu}{' with this engine' if engine and 'same engine' in tier else ''}: this borrows the ratio from {len(rs)} run(s) on other hardware and can be far off.")
    elif level == "thin":
        caveats.append(f"Only {len(rs)} measured run(s) match your workload on this GPU (need {MIN_RUNS}+ for a gpu_specific label): treat the speed as a lead, not a prediction.")
    if tier == "this GPU, other engines":
        caveats.append("Measured with a different engine than requested; engines differ in kernel efficiency.")
    if len(cover["models"]) == 1 or len(cover["quants"]) == 1:
        caveats.append(f"Supporting runs cover {len(cover['models'])} model(s) and {len(cover['quants'])} quantization(s): the ratio may not transfer to others.")
    return {"tier": tier, "level": level, "workload_match": match, "runs": len(rs), "runs_in_tier": in_tier, "conditions_matched": len(rs) if match == "context_and_concurrency" else 0,
            "utilization": median(xs), "low": min(xs) if len(xs) > 1 else None, "high": max(xs) if len(xs) > 1 else None, "coverage": cover, "caveats": caveats}


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
