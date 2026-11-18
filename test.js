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
console.log("all tests pass");
