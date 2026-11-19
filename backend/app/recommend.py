"""Recommendation Engine: hard constraints first, then one pick per priority. No single blended score."""
from .catalog import GPUS, MODELS, QUANTS
from .evidence import evidence
from .memory import estimate_memory
from .performance import decode_ceiling


HIGH_CONFIDENCE = 0.75  # product decision: a recommendation is labelled "high" only at or above this


def confidence(runs: list[dict], model: str, quant: str, gpu: str, users: int) -> tuple[float, str]:
    """Score from the evidence hierarchy in claude.md section K. Not a probability: a fixed ladder, so it can be audited."""
    exact = [r for r in runs if (r["model"], r["quant"], r["gpu"]) == (model, quant, gpu)]
    if any(r["users"] == users for r in exact):
        return 0.9, "this exact model, quantization and GPU was measured at your concurrency"
    if exact:
        return 0.7, "this exact setup was measured, but at a different concurrency"
    if any(r["gpu"] == gpu for r in runs):
        return 0.5, "estimate calibrated on measured runs of other models on this GPU"
    return 0.3, "first-principles estimate: nothing measured on this GPU"


def label(score: float) -> str:
    return "high" if score >= HIGH_CONFIDENCE else "moderate" if score >= 0.5 else "low"


def candidates(runs: list[dict], model_name: str, *, n_gpus: int, context: int, users: int, overhead_gib: float, kv_dtype: str = "fp16") -> list[dict]:
    m = MODELS[model_name]
    active = m["active_params_b"] or m["params_b"]
    out = []
    for gname, g in GPUS.items():
        for qname, q in QUANTS.items():
            mem = estimate_memory(m, q, g, n_gpus=n_gpus, context=context, users=users, overhead_gib=overhead_gib, kv_dtype=kv_dtype)
            score, basis = confidence(runs, model_name, qname, gname, users)
            mem_measured = evidence(runs, model=model_name, quant=qname, gpu=gname, vram_gib=g["vram_gib"] * n_gpus, n_gpus=n_gpus, context=context, users=users, kv_dtype=kv_dtype)["memory_measured"] > 0
            if score >= 0.7 and not mem_measured:
                basis += "; speed only, the memory fit is still an estimate"
            out.append({
                "gpu": gname, "quantization": qname, "fits": mem["fits"], "usd_hr": g["usd_hr"] * n_gpus,
                "decode_ceiling_tok_s": decode_ceiling(active, q["bytes_per_param"], g["bandwidth_gbs"], n_gpus),
                "quality_loss_pct": q["quality_loss_pct"], "quality_loss_kind": q["quality_loss_kind"], "max_users": mem["max_users"],
                "memory_measured": mem_measured, "performance_measured": any((r["model"], r["quant"], r["gpu"]) == (model_name, qname, gname) for r in runs),
                "confidence": score, "confidence_label": label(score), "confidence_basis": basis,
            })
    return out


def recommend(cands: list[dict], *, max_usd_hr: float | None, min_tok_s: float, max_quality_loss_pct: float, min_confidence: float = 0) -> dict:
    removed = {"does_not_fit": 0, "over_budget": 0, "too_slow": 0, "too_much_quality_loss": 0, "unknown_quality_loss": 0, "low_confidence": 0}
    ok = []
    for c in cands:
        if not c["fits"]:
            removed["does_not_fit"] += 1
        elif max_usd_hr is not None and c["usd_hr"] > max_usd_hr:
            removed["over_budget"] += 1
        elif c["decode_ceiling_tok_s"] < min_tok_s:
            removed["too_slow"] += 1
        elif c["quality_loss_pct"] is None:  # the limit cannot be checked against an unknown
            removed["unknown_quality_loss"] += 1
        elif c["quality_loss_pct"] > max_quality_loss_pct:
            removed["too_much_quality_loss"] += 1
        elif c["confidence"] < min_confidence:
            removed["low_confidence"] += 1
        else:
            ok.append(c)
    if not ok:
        return {"feasible": 0, "removed": removed, "picks": {}}
    per_tok = lambda c: c["usd_hr"] / c["decode_ceiling_tok_s"]
    picks = {
        "cheapest": min(ok, key=lambda c: c["usd_hr"]),
        "fastest": max(ok, key=lambda c: c["decode_ceiling_tok_s"]),
        "best_quality": min(ok, key=lambda c: (c["quality_loss_pct"] if c["quality_loss_pct"] is not None else 1e9, c["usd_hr"])),
        "best_value": min(ok, key=per_tok),
        "best_evidence": min(ok, key=lambda c: (-c["confidence"], c["usd_hr"])),
    }
    for name, c in picks.items():
        c["why"] = f"{name.replace('_', ' ')} among {len(ok)} setups that pass your limits"
        c["validate_before_production"] = "Benchmark this exact configuration: the speed is a ceiling estimate." if c["confidence"] < HIGH_CONFIDENCE else "Measured at your concurrency, but on one engine and prompt length: re-measure on yours."
    return {"feasible": len(ok), "removed": removed, "picks": picks}
