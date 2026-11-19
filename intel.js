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
// Calibration (mirrors backend/app/performance.py). The SAMPLE is chosen by hardware and workload, not just described afterwards:
// this GPU (and engine) at your concurrency and a similar context first; then this GPU at your concurrency; then other GPUs the same way;
// and only if nothing matches your concurrency anywhere, the nearest hardware's runs unmatched. Only a fully matched same-GPU sample escapes "extrapolated".
const family = e => (e || "").split(" ")[0].toLowerCase(), MIN_RUNS = 3, CONTEXT_BAND = 4; // ponytail: <3 runs on one GPU is "thin"; same workload = same users and context within 4x
function calibrate(runs, ceilOf, o) { // ceilOf(run) = that run's bandwidth ceiling in tok/s; o = {gpu, engine?, ctx, users}
  const rows = runs.map(r => [r, r.tok_s / ceilOf(r)]), gpu = r => r.gpu == o.gpu, eng = r => family(r.engine) == family(o.engine);
  const tiers = o.engine ? [["this GPU and engine", r => gpu(r) && eng(r), true], ["this GPU, other engines", gpu, true], ["other GPUs, same engine", eng, false], ["other GPUs and engines", () => true, false]]
    : [["this GPU", gpu, true], ["other GPUs", () => true, false]];
  const sameUsers = r => r.users == o.users, similarCtx = r => r.context / o.ctx >= 1 / CONTEXT_BAND && r.context / o.ctx <= CONTEXT_BAND;
  const strata = [["context_and_concurrency", r => sameUsers(r) && similarCtx(r)], ["concurrency_only", sameUsers]];
  let pick = null;
  for (const [local, [match, okW]] of [[true, strata[0]], [true, strata[1]], [false, strata[0]], [false, strata[1]]]) {
    for (const [tier, okH, isLocal] of tiers) { if (isLocal != local) continue; const got = rows.filter(([r]) => okH(r) && okW(r)); if (got.length) { pick = { tier, match, got }; break; } }
    if (pick) break;
  }
  if (!pick) for (const [tier, okH] of tiers) { const got = rows.filter(([r]) => okH(r)); if (got.length) { pick = { tier, match: "none", got }; break; } }
  if (!pick) return { tier: null, level: "none", workloadMatch: null, runs: 0, runsInTier: 0, conditionsMatched: 0, utilization: null, low: null, high: null, caveats: ["No measured runs: speed is a bandwidth ceiling only."] };
  const { tier, match, got } = pick, rs = got.map(x => x[0]), xs = got.map(x => x[1]), uniq = k => [...new Set(rs.map(r => r[k]))].sort((a, b) => a < b ? -1 : 1);
  const inTier = rows.filter(([r]) => tiers.find(t => t[0] == tier)[1](r)).length, local = tier.startsWith("this GPU");
  const level = local && match == "context_and_concurrency" ? (rs.length >= MIN_RUNS && tier != "this GPU, other engines" ? "gpu_specific" : "thin") : "extrapolated", caveats = [];
  if (match == "none") caveats.push(`No measured run at ${o.users} concurrent user(s) anywhere: this uses ${rs.length} run(s) at users ${uniq("users")}, context ${uniq("context")}, so the speed is an extrapolation across workloads.`);
  else if (match == "concurrency_only") caveats.push(`No measured run at a context near ${o.ctx}: this uses ${rs.length} run(s) at ${o.users} user(s) but context ${uniq("context")}. Longer context reads more KV cache per token, so real speed is likely lower.`);
  if (!local) caveats.push(`No usable measured run on ${o.gpu}${o.engine && tier.includes("same engine") ? " with this engine" : ""}: this borrows the ratio from ${rs.length} run(s) on other hardware and can be far off.`);
  else if (level == "thin") caveats.push(`Only ${rs.length} measured run(s) match your workload on this GPU (need ${MIN_RUNS}+ for a gpu_specific label): treat the speed as a lead, not a prediction.`);
  if (tier == "this GPU, other engines") caveats.push("Measured with a different engine than requested; engines differ in kernel efficiency.");
  const models = uniq("model"), quants = uniq("quant");
  if (models.length == 1 || quants.length == 1) caveats.push(`Supporting runs cover ${models.length} model(s) and ${quants.length} quantization(s): the ratio may not transfer to others.`);
  return { tier, level, workloadMatch: match, runs: rs.length, runsInTier: inTier, conditionsMatched: match == "context_and_concurrency" ? rs.length : 0,
    utilization: median(xs), low: xs.length > 1 ? Math.min(...xs) : null, high: xs.length > 1 ? Math.max(...xs) : null, caveats };
}
// Benchmark comparison: which fields differ between runs, and can the difference in speed be put down to one of them?
// Engine is compared by family (vllm, llama.cpp); a different build of the same engine is a separate, softer difference.
const CMP_FIELDS = [["model", "model"], ["gpu", "GPU"], ["gpu_count", "GPU count"], ["quant", "quantization"], ["engine", "engine"], ["engine_build", "engine build"],
  ["kv_dtype", "KV cache type"], ["context", "context length"], ["users", "concurrency"]];
function compareRuns(runs) {
  const val = (r, k) => k == "engine" ? family(r.engine) : k == "engine_build" ? (r.engine || "").split(" ").slice(1).join(" ") : k == "gpu_count" ? (r.gpu_count || 1) : k == "kv_dtype" ? (r.kv_dtype || "fp16") : r[k];
  const found = CMP_FIELDS.filter(([k]) => new Set(runs.map(r => val(r, k))).size > 1).map(([key, label]) => ({ key, label }));
  const familyDiffers = found.some(d => d.key == "engine"), differs = familyDiffers ? found.filter(d => d.key != "engine_build") : found; // another engine: its build is not a separate difference
  const build = differs.some(d => d.key == "engine_build");
  const main = differs.filter(d => d.key != "engine_build"), warnings = [];
  let kind, compared = null;
  if (!main.length && !build) kind = "identical";
  else if (!main.length) { kind = "clean"; compared = "engine build"; }
  else if (main.length == 1) { kind = build ? "caveat" : "clean"; compared = main[0].label; if (build) warnings.push("These are different builds of the same engine, which can move speed on its own: treat the gap as approximate."); }
  else { kind = "not_comparable"; warnings.push(`${main.map(d => d.label).join(", ")} all differ at once, so the gap in speed cannot be put down to any one of them. Pick runs that differ in a single thing.`); }
  if (kind == "identical") warnings.push("The configuration is identical: these are repeat measurements of the same setup, so the gap shows run-to-run noise.");
  if (main.some(d => d.key == "users")) warnings.push("Per-user speed falls as users rise: compare the aggregate row as well.");
  if (main.some(d => d.key == "context")) warnings.push("Longer context reads more KV cache per token, so a lower speed at a longer context is expected.");
  if (main.some(d => d.key == "model") && main.length == 1) warnings.push("Different models: the gap reflects model size and architecture, not hardware or format.");
  const methods = new Set(runs.map(r => r.token_count_method).filter(Boolean));
  if (methods.has("chunks")) warnings.push(methods.size > 1 ? "Token counts were made in different ways, and one is an estimate from streamed chunks: tokens/s are not strictly comparable." : "Token counts are estimates from streamed chunks, so tokens/s may be off.");
  if (runs.some(r => r.local)) warnings.push("Includes your own result: it is not independently verified and is not part of OMEN's reference data.");
  return { kind, compared, differs, warnings };
}
if (typeof module !== 'undefined') module.exports = { confidence, HIGH_CONFIDENCE, cluster, workload, recommend, parseHF, validateRun, ttft, median, TP_EFF, KV_TYPES, effLayers, memoryMeasured, calibrate, compareRuns }; // for test.js
