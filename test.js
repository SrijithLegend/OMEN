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
console.log("all tests pass");
