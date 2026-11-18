// Multi-GPU and workload math, kept out of the page so test.js can lock it.
// ponytail: tensor-parallel efficiency is a flat 0.85 on summed bandwidth; real scaling depends on interconnect. Calibrate with measured runs.
const TP_EFF = 0.85;
// n identical GPUs as one pool: VRAM and price add up, bandwidth adds up minus a parallelism loss.
const cluster = ([gib, bw, usd], n) => [gib * n, n > 1 ? bw * n * TP_EFF : bw, usd * n];
// Turn a decode ceiling into capacity and cost for a monthly volume. util = share of the ceiling you really get (0..1).
// Ceiling-based: ignores prefill and queueing, so treat it as a best case.
function workload(tokS, users, usdHr, outTok, reqMonth, util) {
  const agg = tokS * users * util, hours = reqMonth * outTok / agg / 3600, copies = Math.max(1, Math.ceil(hours / 730));
  return { agg, perUser: tokS * util, latency: outTok / (tokS * util), perM: usdHr / (agg * 3600) * 1e6, hours, copies,
    payPerUse: hours * usdHr, alwaysOn: copies * 730 * usdHr };
}
// Recommendation: hard constraints first (with a count of what each removed), then a pick per priority. No single blended score.
// cands: [{gpu, quant, fits, usd (total $/hr), tok, loss (%)}]. cons: {maxUsd, minTok, maxLoss}; blank/0 = no limit.
function recommend(cands, cons) {
  const dropped = { "does not fit": 0, "over budget": 0, "too slow": 0, "too much quality loss": 0 }, ok = [];
  for (const x of cands) {
    if (!x.fits) dropped["does not fit"]++;
    else if (cons.maxUsd && x.usd > cons.maxUsd) dropped["over budget"]++;
    else if (cons.minTok && x.tok < cons.minTok) dropped["too slow"]++;
    else if (cons.maxLoss !== "" && cons.maxLoss != null && x.loss > cons.maxLoss) dropped["too much quality loss"]++;
    else ok.push(x);
  }
  const by = (f) => ok.slice().sort(f)[0], perTok = x => x.usd / x.tok;
  const picks = ok.length ? [["Cheapest", by((a, b) => a.usd - b.usd)], ["Fastest", by((a, b) => b.tok - a.tok)],
    ["Best quality", by((a, b) => a.loss - b.loss || a.usd - b.usd)], ["Best value ($ per token)", by((a, b) => perTok(a) - perTok(b))]] : [];
  return { ok, dropped, picks };
}
// Hugging Face import: config.json + safetensors total (in params) -> [params_b, layers, kv_heads, head_dim]. Missing fields throw, never guessed.
function parseHF(cfg, totalParams) {
  const c = cfg.text_config || cfg, warn = [], need = (k, v) => { if (!(v > 0)) throw new Error("config.json has no usable " + k); return v; };
  const layers = need("num_hidden_layers", c.num_hidden_layers), heads = need("num_attention_heads", c.num_attention_heads);
  const kv = c.num_key_value_heads || heads, hd = c.head_dim || need("hidden_size", c.hidden_size) / heads;
  need("parameter count (safetensors)", totalParams);
  if (c.kv_lora_rank) throw new Error("MLA attention (compressed KV cache) is not supported for import");
  if (c.num_local_experts || c.n_routed_experts || c.num_experts) warn.push("Mixture-of-experts: active parameters are unknown, so speed uses the total and is underestimated.");
  if (c.sliding_window && c.use_sliding_window !== false) warn.push("Sliding-window attention: KV cache is overestimated at long context.");
  return { entry: [+(totalParams / 1e9).toFixed(2), layers, kv, hd], warn };
}
if (typeof module !== 'undefined') module.exports = { cluster, workload, recommend, parseHF, TP_EFF }; // for test.js
