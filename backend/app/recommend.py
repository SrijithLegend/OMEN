"""Recommendation Engine: hard constraints first, then one pick per priority. No single blended score."""
from .catalog import GPUS, MODELS, QUANTS, RUNS
from .memory import estimate_memory
from .performance import decode_ceiling


def candidates(model_name: str, *, n_gpus: int, context: int, users: int, overhead_gib: float) -> list[dict]:
    m = MODELS[model_name]
    active = m["active_params_b"] or m["params_b"]
    out = []
    for gname, g in GPUS.items():
        for qname, q in QUANTS.items():
            mem = estimate_memory(m, q, g, n_gpus=n_gpus, context=context, users=users, overhead_gib=overhead_gib)
            out.append({
                "gpu": gname, "quantization": qname, "fits": mem["fits"], "usd_hr": g["usd_hr"] * n_gpus,
                "decode_ceiling_tok_s": decode_ceiling(active, q["bytes_per_param"], g["bandwidth_gbs"], n_gpus),
                "quality_loss_pct": q["quality_loss_pct"], "max_users": mem["max_users"],
                "measured": any((r["model"], r["quant"], r["gpu"]) == (model_name, qname, gname) for r in RUNS),
            })
    return out


def recommend(cands: list[dict], *, max_usd_hr: float | None, min_tok_s: float, max_quality_loss_pct: float) -> dict:
    removed = {"does_not_fit": 0, "over_budget": 0, "too_slow": 0, "too_much_quality_loss": 0}
    ok = []
    for c in cands:
        if not c["fits"]:
            removed["does_not_fit"] += 1
        elif max_usd_hr is not None and c["usd_hr"] > max_usd_hr:
            removed["over_budget"] += 1
        elif c["decode_ceiling_tok_s"] < min_tok_s:
            removed["too_slow"] += 1
        elif c["quality_loss_pct"] > max_quality_loss_pct:
            removed["too_much_quality_loss"] += 1
        else:
            ok.append(c)
    if not ok:
        return {"feasible": 0, "removed": removed, "picks": {}}
    per_tok = lambda c: c["usd_hr"] / c["decode_ceiling_tok_s"]
    picks = {
        "cheapest": min(ok, key=lambda c: c["usd_hr"]),
        "fastest": max(ok, key=lambda c: c["decode_ceiling_tok_s"]),
        "best_quality": min(ok, key=lambda c: (c["quality_loss_pct"], c["usd_hr"])),
        "best_value": min(ok, key=per_tok),
        "best_evidence": min(ok, key=lambda c: (not c["measured"], c["usd_hr"])),
    }
    for name, c in picks.items():
        c["why"] = f"{name.replace('_', ' ')} among {len(ok)} setups that pass your limits"
        c["validate_before_production"] = "Benchmark this exact configuration: the speed is a ceiling estimate." if not c["measured"] else "Speed was measured on a different workload: re-measure yours."
    return {"feasible": len(ok), "removed": removed, "picks": picks}
