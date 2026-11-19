"""Memory Intelligence: weights, KV cache, overhead, headroom, fit status. Same formulas as fit.js / intel.js."""
GIB = 2**30
TP_EFF = 0.85  # ponytail: flat tensor-parallel bandwidth efficiency; calibrate against measured multi-GPU runs
KV_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}  # bytes per cached element; per-tensor scales assumed (per-head scales add a little)


def weights_gib(params_b: float, bytes_per_param: float) -> float:
    return params_b * 1e9 * bytes_per_param / GIB


def effective_layers(layers: int, sliding: list | None, context: int) -> float:
    """Layers-equivalent of full-context cache. Sliding-window layers only keep `window` tokens, so they count for min(context, window)/context of a layer."""
    if not sliding:
        return layers
    local, window = sliding
    return layers - local + local * min(context, window) / context


def kv_gib_per_user(layers: float, kv_heads: int, head_dim: int, context: int, kv_bytes: int = 2) -> float:
    return 2 * layers * kv_heads * head_dim * kv_bytes * context / GIB


def fit_status(fits: bool, spare_fraction: float, memory_measured: bool) -> str:
    if not fits:
        return "does_not_fit_under_stated_assumptions"
    if memory_measured:  # only a recorded peak-VRAM measurement at this context and concurrency counts; a speed benchmark does not
        return "measured_to_fit"
    return "likely_fits" if spare_fraction >= 0.1 else "fits_theoretically"


def estimate_memory(model: dict, quant: dict, gpu: dict, *, n_gpus: int = 1, context: int, users: int,
                    overhead_gib: float = 1.5, memory_measured: bool = False, kv_dtype: str = "fp16", sliding_cache: bool = False) -> dict:
    """sliding_cache=True assumes the engine frees KV for sliding-window layers (llama.cpp does for SWA models; many engines do not)."""
    if kv_dtype not in KV_BYTES:
        raise ValueError(f"kv_dtype must be one of {sorted(KV_BYTES)}")
    vram = gpu["vram_gib"] * n_gpus
    overhead = overhead_gib * n_gpus
    w = weights_gib(model["params_b"], quant["bytes_per_param"])
    sliding = model.get("sliding")
    kv_args = (model["kv_heads"], model["head_dim"], context, KV_BYTES[kv_dtype])
    kv = kv_gib_per_user(effective_layers(model["layers"], sliding if sliding_cache else None, context), *kv_args)
    free = vram - w - overhead
    max_users = max(0, int(free // kv)) if kv > 0 else 0
    need = w + overhead + kv * users
    parts = {"weights": w, "kv_cache": kv * users, "overhead": overhead}
    return {
        "weights_gib": w, "kv_gib_per_user": kv, "kv_gib_total": kv * users, "overhead_gib": overhead,
        "kv_dtype": kv_dtype, "kv_bytes_per_element": KV_BYTES[kv_dtype],
        "sliding_window_applied": bool(sliding and sliding_cache),
        # when the model has sliding-window layers but the engine may not free them, show the other number instead of hiding it
        "kv_gib_per_user_if_sliding_freed": kv_gib_per_user(effective_layers(model["layers"], sliding, context), *kv_args) if sliding and not sliding_cache else None,
        "total_gib": need, "vram_gib": vram, "headroom_gib": free, "max_users": max_users,
        "fits": max_users >= users,
        "status": fit_status(max_users >= users, (vram - need) / vram, memory_measured),
        "largest_component": max(parts, key=parts.get),
    }
