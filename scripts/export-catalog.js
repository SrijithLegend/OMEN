// Exports the calculator's data (models, quantizations, GPUs, measured runs) to backend/app/catalog.json so the API and the page share one source.
// Run: node scripts/export-catalog.js
const fs = require("fs"), dir = __dirname + "/../";
const src = ["usecases", "models", "quants", "memtypes", "gpus", "gpu-compute", "gpu-sources", "measured"].map(f => fs.readFileSync(dir + f + ".js", "utf8")).join("\n");
const d = new Function(src + ";return {MODELS_BY_CREATOR, MODEL_SOURCES, MODEL_SLIDING, MODEL_USE_CASES, QUANTS_BY_FAMILY, GPUS_BY_VENDOR, GPU_TFLOPS, GPU_SOURCES, gpuSpecStatus, RUNS}")();
let QL = {}; // quality loss lives in quants.js too
new Function("o", fs.readFileSync(dir + "quants.js", "utf8") + ";o.QUALITY_LOSS=QUALITY_LOSS")(QL);
const flat = by => Object.assign({}, ...Object.entries(by).map(([k, o]) => Object.fromEntries(Object.entries(o).map(([n, v]) => [n, [v, k]]))));
const out = { models: {}, quants: {}, gpus: {}, runs: d.RUNS };
for (const [n, [[p, L, h, hd, act], creator]] of Object.entries(flat(d.MODELS_BY_CREATOR)))
  out.models[n] = { creator, params_b: p, layers: L, kv_heads: h, head_dim: hd, active_params_b: act ?? null, use_cases: d.MODEL_USE_CASES[n] || ["General"], source: d.MODEL_SOURCES[n] || null, sliding: d.MODEL_SLIDING[n] || null };
for (const [n, [b, fam]] of Object.entries(flat(d.QUANTS_BY_FAMILY))) out.quants[n] = { family: fam, bytes_per_param: b, quality_loss_pct: QL.QUALITY_LOSS[n][0], quality_loss_kind: QL.QUALITY_LOSS[n][1] };
for (const [n, [[gib, bw, usd], vendor]] of Object.entries(flat(d.GPUS_BY_VENDOR))) out.gpus[n] = { vendor, vram_gib: gib, bandwidth_gbs: bw, usd_hr: usd, tflops_fp16: d.GPU_TFLOPS[n] ?? null,
  spec_status: d.gpuSpecStatus(n, n in d.GPU_TFLOPS), spec_source: d.GPU_SOURCES[n] ?? null };
fs.writeFileSync(dir + "backend/app/catalog.json", JSON.stringify(out, null, 1));
console.log(Object.keys(out.models).length + " models, " + Object.keys(out.quants).length + " quants, " + Object.keys(out.gpus).length + " GPUs, " + out.runs.length + " runs");
