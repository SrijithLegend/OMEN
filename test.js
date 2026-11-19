// Locks the verified numbers. Run: node test.js
const assert = require("assert"), fit = require("./fit.js");
// Llama-3-8B-style, fp16, 8k ctx, 24 GiB GPU, 1008 GB/s, $0.40/hr
const r = fit(8, 2, 32, 8, 128, 2, 8192, 24, 1008, 0.40);
assert.strictEqual(r.weights.toFixed(1), "14.9");
assert.strictEqual(r.kv.toFixed(1), "1.0");
assert.strictEqual(r.users, 7);
assert.strictEqual(Math.round(r.tok), 63);
assert.strictEqual(r.cost.toFixed(3), "0.057");
// int4 (0.57 B/param): 4.2 GiB weights, 18 users, ~221 tok/s
const q = fit(8, 0.57, 32, 8, 128, 2, 8192, 24, 1008, 0.40);
assert.strictEqual(q.weights.toFixed(1), "4.2");
assert.strictEqual(q.users, 18);
assert.strictEqual(Math.round(q.tok), 221);
// MoE: memory from total params, speed from active (Mixtral-8x7B int4 on 80 GiB)
const m = fit(46.7, 0.57, 32, 8, 128, 2, 8192, 80, 2039, 1.8, 1.5, 12.9);
assert.strictEqual(m.users, 53);
assert.strictEqual(Math.round(m.tok), 277);
// No fit: 70B fp16 on a 24 GiB card has negative headroom, 0 users, no cost
const n = fit(70, 2, 80, 8, 128, 2, 8192, 24, 1008, 0.40);
assert.strictEqual(n.fits, false);
assert.strictEqual(n.users, 0);
assert.strictEqual(n.cost, null);
// Partial fit: 8B fp16 at 32k context on 24 GiB leaves room for exactly 1 user (KV 4 GiB each)
const p = fit(8, 2, 32, 8, 128, 2, 32768, 24, 1008, 0.40);
assert.strictEqual(p.kv.toFixed(1), "4.0");
assert.strictEqual(p.users, 1);
assert.strictEqual(p.fits, true);
// Workload: 100 tok/s ceiling, 4 users at 50% => 200 tok/s total; 1M requests x 500 tokens at $1/hr
const { cluster, workload } = require("./intel.js"), w = workload(100, 4, 1, 500, 1e6, 0.5);
assert.strictEqual(w.agg, 200);
assert.strictEqual(w.perM.toFixed(3), "1.389");
assert.strictEqual(Math.round(w.hours), 694);
assert.strictEqual(w.copies, 1);
assert.strictEqual(w.latency, 10);
// Cluster: 2 GPUs double VRAM and price, bandwidth x2 x 0.85
const c2 = cluster([24, 1000, 0.4], 2);
assert.deepStrictEqual([c2[0], c2[1], c2[2]], [48, 1700, 0.8]);
assert.deepStrictEqual(cluster([24, 1000, 0.4], 1), [24, 1000, 0.4]);
// Recommend: hard limits first, then one pick per priority
const { recommend } = require("./intel.js"), cs = [
  { gpu: "A", quant: "q4", fits: true, usd: 0.4, tok: 200, loss: 2 }, { gpu: "B", quant: "fp16", fits: true, usd: 3, tok: 300, loss: 0 },
  { gpu: "C", quant: "q4", fits: false, usd: 0.1, tok: 90, loss: 2 }, { gpu: "D", quant: "q2", fits: true, usd: 0.2, tok: 250, loss: 15 }];
const rc = recommend(cs, { maxUsd: 1, minTok: 0, maxLoss: 5 });
assert.deepStrictEqual(rc.ok.map(x => x.gpu), ["A"]);
assert.deepStrictEqual(rc.dropped, { "does not fit": 1, "over budget": 1, "too slow": 0, "too much quality loss": 1, "unknown quality loss": 0, "below the confidence limit": 0 });
const rn = recommend(cs, { maxUsd: 0, minTok: 0, maxLoss: "" });
assert.strictEqual(rn.picks[0][1].gpu, "D"); assert.strictEqual(rn.picks[1][1].gpu, "B"); assert.strictEqual(rn.picks[2][1].gpu, "B");
assert.deepStrictEqual(recommend([cs[2]], {}).picks, []);
// HF import: Qwen2.5-7B config => 28 layers, 4 KV heads, head dim 3584/28 = 128
const { parseHF } = require("./intel.js"), qc = { num_hidden_layers: 28, num_attention_heads: 28, num_key_value_heads: 4, hidden_size: 3584, sliding_window: 131072, use_sliding_window: false };
assert.deepStrictEqual(parseHF(qc, 7615616512), { entry: [7.62, 28, 4, 128], warn: [] });
assert.throws(() => parseHF({ num_attention_heads: 8, hidden_size: 64 }, 1e9), /num_hidden_layers/);
assert.throws(() => parseHF(qc, 0), /parameter count/);
assert.throws(() => parseHF({ ...qc, kv_lora_rank: 512 }, 1e9), /MLA/);
assert.strictEqual(parseHF({ ...qc, num_local_experts: 8 }, 1e9).warn.length, 1);
// Benchmark validation
const { validateRun } = require("./intel.js"), kn = { model: ["A"], gpu: ["G"], quant: ["Q"] }, good = { model: "A", quant: "Q", gpu: "G", engine: "e 1", context: 128, users: 1, tok_s: 50, source: "http://x" };
assert.deepStrictEqual(validateRun(good, kn), []);
assert.strictEqual(validateRun({ ...good, gpu: "Z" }, kn).length, 1);
assert.strictEqual(validateRun({ ...good, tok_s: 0, context: 1.5 }, kn).length, 2);
assert.strictEqual(validateRun({ ...good, source: "" }, kn).length, 1);
// Prefill: 8B active params, 2000 prompt tokens, 989 TFLOPS at 40% => 2*8e9*2000/(989e12*0.4) = 0.0809 s
const { ttft, median } = require("./intel.js");
assert.strictEqual(ttft(8, 2000, 989).toFixed(4), "0.0809");
assert.strictEqual(ttft(8, 2000, undefined), null);
assert.strictEqual(median([3, 1, 2]), 2); assert.strictEqual(median([1, 2, 3, 4]), 2.5);
const { confidence, HIGH_CONFIDENCE } = require("./intel.js"), CR = [{ model: "A", quant: "Q4", gpu: "G1", users: 1 }];
assert.strictEqual(HIGH_CONFIDENCE, 0.75);
assert.strictEqual(confidence(CR, "A", "Q4", "G1", 1)[0], 0.9);
assert.strictEqual(confidence(CR, "A", "Q4", "G1", 4)[0], 0.7);
assert.strictEqual(confidence(CR, "B", "Q4", "G1", 1)[0], 0.5);
assert.strictEqual(confidence(CR, "A", "Q4", "G2", 1)[0], 0.3);
const rf = recommend([{ gpu: "a", quant: "q", fits: true, usd: 1, tok: 50, loss: 0, conf: 0.9 }, { gpu: "b", quant: "q", fits: true, usd: 1, tok: 50, loss: 0, conf: 0.5 }], { minConf: 0.75 });
assert.strictEqual(rf.ok.length, 1);
assert.strictEqual(rf.dropped["below the confidence limit"], 1);
// KV cache types and sliding-window layers: fp8 halves the cache; Gemma-3-27B-style [52 local of 62 layers, window 1024] at 32k keeps ~19% of a full cache
const { KV_TYPES, effLayers, memoryMeasured, calibrate } = require("./intel.js");
assert.deepStrictEqual([KV_TYPES.fp16, KV_TYPES.bf16, KV_TYPES.fp8, KV_TYPES.int8], [2, 2, 1, 1]);
assert.strictEqual(fit(8, 2, 32, 8, 128, KV_TYPES.fp8, 8192, 24, 1008, 0.4).kv, fit(8, 2, 32, 8, 128, 2, 8192, 24, 1008, 0.4).kv / 2);
assert.strictEqual(effLayers(62, null, 32768), 62);
assert.strictEqual(effLayers(62, [52, 1024], 32768), 10 + 52 * 1024 / 32768);
assert.strictEqual(effLayers(62, [52, 1024], 512), 62); // context shorter than the window: nothing to save
// Evidence: only a recorded peak-VRAM run counts for memory; speed alone does not
const base = { model: "A", quant: "Q", gpu: "G", gpu_count: 1, context: 128, users: 1, tok_s: 100, engine: "llama.cpp b1" }, ask = { model: "A", quant: "Q", gpu: "G", vram: 24, n: 1, ctx: 4096, users: 2, kv: "fp16" };
assert.strictEqual(memoryMeasured([base], ask), 0);
const mem = { ...base, context: 4096, users: 4, vram_gib: 9 };
assert.strictEqual(memoryMeasured([mem], ask), 1);
assert.strictEqual(memoryMeasured([{ ...mem, context: 2048 }], ask), 0);
assert.strictEqual(memoryMeasured([{ ...mem, users: 1 }], ask), 0);
assert.strictEqual(memoryMeasured([{ ...mem, gpu_count: 2 }], ask), 0);
assert.strictEqual(memoryMeasured([{ ...mem, kv_dtype: "fp8" }], ask), 0);
assert.strictEqual(memoryMeasured([mem], { ...ask, kv: "fp8" }), 1);
assert.strictEqual(memoryMeasured([{ ...mem, vram_gib: 30 }], ask), 0); // peak larger than the card: not evidence of fit
// Calibration: most specific evidence first, thin and extrapolated are labelled, never one global factor
const ceil = r => 100, cr = (o = {}) => ({ ...base, ...o }), want = { gpu: "G", ctx: 8192, users: 4 };
assert.strictEqual(calibrate([], ceil, want).level, "none");
const ex = calibrate([cr({ gpu: "H", tok_s: 50 })], ceil, want);
assert.deepStrictEqual([ex.level, ex.tier, ex.low], ["extrapolated", "other GPUs", null]);
assert.strictEqual(calibrate([cr()], ceil, want).level, "thin");
const many = calibrate([cr({ tok_s: 40 }), cr({ tok_s: 60 }), cr({ tok_s: 50 }), cr({ tok_s: 99, engine: "vllm 1" })], ceil, { ...want, engine: "llama.cpp" });
assert.deepStrictEqual([many.tier, many.level, many.runs, many.low, many.high, many.utilization], ["this GPU and engine", "gpu_specific", 3, 0.4, 0.6, 0.5]);
assert.strictEqual(calibrate([cr({ engine: "vllm 1" })], ceil, { ...want, engine: "llama.cpp" }).tier, "this GPU, other engines");
assert.strictEqual(calibrate([cr({ context: 8192, users: 4 })], ceil, want).conditionsMatched, 1);
// Quality loss: unknown cannot satisfy a limit; measured vs estimated is explicit
const uq = recommend([{ gpu: "U", quant: "q", fits: true, usd: 1, tok: 50, loss: null }, { gpu: "K", quant: "q", fits: true, usd: 2, tok: 50, loss: 1 }], { maxLoss: 5 });
assert.deepStrictEqual(uq.ok.map(x => x.gpu), ["K"]); assert.strictEqual(uq.dropped["unknown quality loss"], 1);
assert.strictEqual(recommend([{ gpu: "U", quant: "q", fits: true, usd: 1, tok: 50, loss: null }], { maxLoss: "" }).ok.length, 1); // no limit set: kept
assert.strictEqual(recommend([{ gpu: "U", quant: "q", fits: true, usd: 1, tok: 50, loss: null }, { gpu: "K", quant: "q", fits: true, usd: 9, tok: 50, loss: 1 }], { maxLoss: "" }).picks[2][1].gpu, "K");
const qsrc = require("fs").readFileSync(__dirname + "/quants.js", "utf8"), qt = new Function(qsrc + ";return {QUALITY_LOSS, qualityText}")();
assert.deepStrictEqual(qt.qualityText("Q4_0"), ["+4.23%", "measured: perplexity on Llama-2-7B (wikitext); other models differ"]);
assert.strictEqual(qt.qualityText("GPTQ / AWQ int4")[0], "~2%"); assert.strictEqual(qt.qualityText("int2")[0], "unknown"); assert.strictEqual(qt.qualityText("fp8")[0], "<1%");
assert.strictEqual(qt.qualityText("fp16 / bf16")[0], "none");
assert.ok(Object.values(qt.QUALITY_LOSS).every(([v, k]) => ["measured", "estimated", "unknown"].includes(k) && (k == "unknown") == (v == null)));
// measured figures reproduce from the cited perplexities (F16 5.9066)
for (const [n, ppl] of [["Q4_0", 6.1565], ["Q5_0", 5.9862], ["Q8_0", 5.9070]]) assert.ok(Math.abs(qt.QUALITY_LOSS[n][0] - 100 * (ppl - 5.9066) / 5.9066) < 0.01, n);
console.log("all tests pass");
