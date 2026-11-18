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
// Benchmark validation: returns a list of problems (empty = acceptable). Names must match the calculator's lists so runs can be joined to estimates.
function validateRun(r, known) {
  const bad = [], num = (k, int) => { if (!(typeof r[k] == "number" && r[k] > 0) || (int && !Number.isInteger(r[k]))) bad.push(`${k} must be a positive ${int ? "whole " : ""}number`); };
  if (!r || typeof r != "object") return ["not a JSON object"];
  for (const k of ["model", "quant", "gpu", "engine"]) if (!r[k] || typeof r[k] != "string") bad.push(k + " is required");
  num("context", true); num("users", true); num("tok_s");
  for (const [k, list] of Object.entries(known)) if (r[k] && !list.includes(r[k])) bad.push(`${k} "${r[k]}" is not in the calculator's list`);
  if (r.tok_s > 20000) bad.push("tok_s above 20000 per user is implausible");
  if (!r.source) bad.push("source is missing: without a link this stays a Community record that cannot be checked");
  return bad;
}
// Prefill (prompt processing) is compute-bound: ~2 FLOPs per active parameter per prompt token, at a fraction (mfu) of peak.
// ponytail: mfu 0.4 is a typical large-batch figure, not calibrated (no measured prefill data yet). Replace with a fitted value once runs record TTFT.
const MFU = 0.4;
const ttft = (activeB, inTok, tflops, n = 1, mfu = MFU) => tflops ? 2 * activeB * 1e9 * inTok / (tflops * 1e12 * (n > 1 ? n * TP_EFF : 1) * mfu) : null;
const median = a => { const s = [...a].sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
if (typeof module !== 'undefined') module.exports = { cluster, workload, recommend, parseHF, validateRun, ttft, median, TP_EFF }; // for test.js
