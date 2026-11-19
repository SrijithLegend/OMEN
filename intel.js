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
  const dropped = { "does not fit": 0, "over budget": 0, "too slow": 0, "too much quality loss": 0, "unknown quality loss": 0, "below the confidence limit": 0 }, ok = [];
  for (const x of cands) {
    if (!x.fits) dropped["does not fit"]++;
    else if (cons.maxUsd && x.usd > cons.maxUsd) dropped["over budget"]++;
    else if (cons.minTok && x.tok < cons.minTok) dropped["too slow"]++;
    else if (cons.maxLoss !== "" && cons.maxLoss != null && x.loss == null) dropped["unknown quality loss"]++; // a limit cannot be checked against an unknown
    else if (cons.maxLoss !== "" && cons.maxLoss != null && x.loss > cons.maxLoss) dropped["too much quality loss"]++;
    else if (cons.minConf && x.conf < cons.minConf) dropped["below the confidence limit"]++;
    else ok.push(x);
  }
  const by = (f) => ok.slice().sort(f)[0], perTok = x => x.usd / x.tok;
  const picks = ok.length ? [["Cheapest", by((a, b) => a.usd - b.usd)], ["Fastest", by((a, b) => b.tok - a.tok)],
    ["Best quality", by((a, b) => (a.loss ?? 1e9) - (b.loss ?? 1e9) || a.usd - b.usd)], ["Best value ($ per token)", by((a, b) => perTok(a) - perTok(b))]] : [];
  return { ok, dropped, picks };
}
// Confidence ladder (claude.md section K). High = 75% or above, which only a measured setup reaches. A fixed ladder, not a probability.
const HIGH_CONFIDENCE = 0.75;
function confidence(runs, model, quant, gpu, users) {
  const exact = runs.filter(r => r.model == model && r.quant == quant && r.gpu == gpu);
  if (exact.some(r => r.users == users)) return [0.9, "this exact model, quantization and GPU was measured at your concurrency"];
  if (exact.length) return [0.7, "this exact setup was measured, but at a different concurrency"];
  if (runs.some(r => r.gpu == gpu)) return [0.5, "estimate calibrated on measured runs of other models on this GPU"];
  return [0.3, "first-principles estimate: nothing measured on this GPU"];
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
// KV cache element sizes in bytes. fp8/int8 assume per-tensor scales; their quality impact is not measured by OMEN.
const KV_TYPES = { "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1 };
// Sliding-window layers keep only `window` tokens, so they count as a fractional layer: fit()'s KV formula stays unchanged. sliding = [local layers, window] or null.
const effLayers = (layers, sliding, ctx) => sliding ? layers - sliding[0] + sliding[0] * Math.min(ctx, sliding[1]) / ctx : layers;
// Evidence categories (see backend/app/evidence.py). A speed benchmark is not a memory measurement: only a recorded peak-VRAM run counts for "measured to fit".
// o = {model, quant, gpu, vram (total GiB), n (GPUs), ctx, users, kv ("fp16"...)}. Returns how many runs establish memory.
const memoryMeasured = (runs, o) => runs.filter(r => r.model == o.model && r.quant == o.quant && r.gpu == o.gpu && (r.gpu_count || 1) == o.n && r.vram_gib > 0 && r.vram_gib <= o.vram
  && r.context >= o.ctx && r.users >= o.users && KV_TYPES[r.kv_dtype || "fp16"] >= KV_TYPES[o.kv]).length;
// Calibration (mirrors backend/app/performance.py): real/ceiling ratio from the most specific evidence: this GPU (and engine) first, other GPUs only as a flagged extrapolation.
const family = e => (e || "").split(" ")[0].toLowerCase(), MIN_RUNS = 3; // ponytail: fewer than 3 runs on one GPU is "thin"; raise as the dataset grows
function calibrate(runs, ceilOf, o) { // ceilOf(run) = that run's bandwidth ceiling in tok/s; o = {gpu, engine?, ctx, users}
  const rows = runs.map(r => [r, r.tok_s / ceilOf(r)]), gpu = r => r.gpu == o.gpu, eng = r => family(r.engine) == family(o.engine);
  const tiers = o.engine ? [["this GPU and engine", r => gpu(r) && eng(r)], ["this GPU, other engines", gpu], ["other GPUs, same engine", eng], ["other GPUs and engines", () => true]]
    : [["this GPU", gpu], ["other GPUs", () => true]];
  for (const [tier, ok] of tiers) {
    const got = rows.filter(([r]) => ok(r)); if (!got.length) continue;
    const rs = got.map(x => x[0]), xs = got.map(x => x[1]), uniq = k => [...new Set(rs.map(r => r[k]))].sort();
    const level = !tier.startsWith("this GPU") ? "extrapolated" : rs.length >= MIN_RUNS && tier != "this GPU, other engines" ? "gpu_specific" : "thin";
    const matched = rs.filter(r => r.users == o.users && r.context / o.ctx >= 0.25 && r.context / o.ctx <= 4).length, caveats = [];
    if (level == "extrapolated") caveats.push(`No measured run on ${o.gpu}${o.engine ? " with this engine" : ""}: this borrows the ratio from ${rs.length} run(s) on other hardware and can be far off.`);
    else if (level == "thin") caveats.push(`Only ${rs.length} measured run(s) on this GPU (need ${MIN_RUNS}+ for a gpu_specific label): treat the speed as a lead, not a prediction.`);
    if (tier == "this GPU, other engines") caveats.push("Measured with a different engine than requested; engines differ in kernel efficiency.");
    if (!matched) caveats.push(`None of the supporting runs match your conditions (context ${o.ctx}, ${o.users} user(s)); they used context ${uniq("context")}, users ${uniq("users")}.`);
    const models = uniq("model"), quants = uniq("quant");
    if (models.length == 1 || quants.length == 1) caveats.push(`Supporting runs cover ${models.length} model(s) and ${quants.length} quantization(s): the ratio may not transfer to others.`);
    return { tier, level, runs: rs.length, utilization: median(xs), low: xs.length > 1 ? Math.min(...xs) : null, high: xs.length > 1 ? Math.max(...xs) : null, conditionsMatched: matched, caveats };
  }
  return { tier: null, level: "none", runs: 0, utilization: null, low: null, high: null, caveats: ["No measured runs: speed is a bandwidth ceiling only."] };
}
if (typeof module !== 'undefined') module.exports = { confidence, HIGH_CONFIDENCE, cluster, workload, recommend, parseHF, validateRun, ttft, median, TP_EFF, KV_TYPES, effLayers, memoryMeasured, calibrate }; // for test.js
