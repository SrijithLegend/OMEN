def fit(params_b, bytes_per_param, layers, kv_heads, head_dim, kv_bytes,
        context, gpu_gib, gpu_bw_gbs, gpu_usd_hr, overhead_gib=1.5):
    weights = params_b * 1e9 * bytes_per_param / 2**30
    kv_per_user = 2 * layers * kv_heads * head_dim * kv_bytes * context / 2**30
    free = gpu_gib - weights - overhead_gib
    users = max(0, int(free // kv_per_user))
    tok_s = gpu_bw_gbs * 1e9 / (params_b * 1e9 * bytes_per_param)
    return dict(fits=free >= kv_per_user, users=users, tok_s_max=round(tok_s),
                usd_per_user_hr=round(gpu_usd_hr / users, 3) if users else None)


if __name__ == "__main__":
    # Llama-3-8B-style, fp16 weights, 8k ctx, 24 GiB GPU (assumed specs)
    r = fit(8, 2, 32, 8, 128, 2, 8192, 24, 1008, 0.40)
    assert r == dict(fits=True, users=7, tok_s_max=63, usd_per_user_hr=0.057), r
    print(r)
