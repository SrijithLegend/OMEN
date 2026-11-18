"""Memory Intelligence: weights, KV cache, overhead, headroom, fit status. Same formulas as fit.js."""
GIB = 2**30
TP_EFF = 0.85  # ponytail: flat tensor-parallel bandwidth efficiency; calibrate against measured multi-GPU runs


def weights_gib(params_b: float, bytes_per_param: float) -> float:
    return params_b * 1e9 * bytes_per_param / GIB


def kv_gib_per_user(layers: int, kv_heads: int, head_dim: int, context: int, kv_bytes: int = 2) -> float:
    return 2 * layers * kv_heads * head_dim * kv_bytes * context / GIB


def fit_status(fits: bool, spare_fraction: float, measured: bool) -> str:
    if not fits:
        return "does_not_fit_under_stated_assumptions"
    if measured:
        return "measured_to_fit"
    return "likely_fits" if spare_fraction >= 0.1 else "fits_theoretically"


def estimate_memory(model: dict, quant: dict, gpu: dict, *, n_gpus: int = 1, context: int, users: int,
                    overhead_gib: float = 1.5, measured: bool = False) -> dict:
    vram = gpu["vram_gib"] * n_gpus
    overhead = overhead_gib * n_gpus
    w = weights_gib(model["params_b"], quant["bytes_per_param"])
    kv = kv_gib_per_user(model["layers"], model["kv_heads"], model["head_dim"], context)
    free = vram - w - overhead
    max_users = max(0, int(free // kv)) if kv > 0 else 0
    need = w + overhead + kv * users
    parts = {"weights": w, "kv_cache": kv * users, "overhead": overhead}
    return {
        "weights_gib": w, "kv_gib_per_user": kv, "kv_gib_total": kv * users, "overhead_gib": overhead,
        "total_gib": need, "vram_gib": vram, "headroom_gib": free, "max_users": max_users,
        "fits": max_users >= users,
        "status": fit_status(max_users >= users, (vram - need) / vram, measured),
        "largest_component": max(parts, key=parts.get),
    }
