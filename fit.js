function fit(params_b, bytes_per_param, layers, kv_heads, head_dim, kv_bytes,
             context, gpu_gib, gpu_bw_gbs, gpu_usd_hr, overhead_gib = 1.5, active_params_b = params_b) {
  const weights = params_b * 1e9 * bytes_per_param / 2 ** 30;
  const kv = 2 * layers * kv_heads * head_dim * kv_bytes * context / 2 ** 30;
  const free = gpu_gib - weights - overhead_gib;
  const users = Math.max(0, Math.floor(free / kv));
  return { weights, kv, free, users, fits: free >= kv,
    tok: gpu_bw_gbs * 1e9 / (active_params_b * 1e9 * bytes_per_param),
    cost: users ? gpu_usd_hr / users : null };
}
if (typeof module !== 'undefined') module.exports = fit; // for test.js
