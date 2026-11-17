def fit(params_b, bytes_per_param, layers, kv_heads, head_dim, kv_bytes,
        context, gpu_gib, gpu_bw_gbs, gpu_usd_hr, overhead_gib=1.5, active_params_b=None):
    active = active_params_b or params_b  # MoE: speed reads only active params
    weights = params_b * 1e9 * bytes_per_param / 2**30
    kv_per_user = 2 * layers * kv_heads * head_dim * kv_bytes * context / 2**30
    free = gpu_gib - weights - overhead_gib
    users = max(0, int(free // kv_per_user))
    tok_s = gpu_bw_gbs * 1e9 / (active * 1e9 * bytes_per_param)
    return dict(fits=free >= kv_per_user, users=users, tok_s_max=round(tok_s),
                usd_per_user_hr=round(gpu_usd_hr / users, 3) if users else None)


if __name__ == "__main__":
    # Llama-3-8B-style, fp16 weights, 8k ctx, 24 GiB GPU (assumed specs)
    r = fit(8, 2, 32, 8, 128, 2, 8192, 24, 1008, 0.40)
    assert r == dict(fits=True, users=7, tok_s_max=63, usd_per_user_hr=0.057), r
    # Mixtral-8x7B: 46.7B total, 12.9B active. Memory from total, speed from active.
    m = fit(46.7, 0.57, 32, 8, 128, 2, 8192, 80, 2039, 1.80, active_params_b=12.9)
    assert m['tok_s_max'] == round(2039e9 / (12.9e9 * 0.57)), m
    assert m['users'] == int((80 - 46.7e9 * 0.57 / 2**30 - 1.5) // 1.0), m
    print(r, m)
