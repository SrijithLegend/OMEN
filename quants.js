// Bytes per weight, including scale/zero-point overhead (bits per weight / 8).
// GGUF bpw from llama.cpp (Llama-2-7B file sizes, README at tag b4000: Q2_K 3.35 bpw as a whole-file mix, not the 2.56 of the type alone); others are typical values. ESTIMATED: real files vary by model.
const QUANTS_BY_FAMILY = {
  "Full precision": { "fp32": 4, "fp16 / bf16": 2 },
  "Float 8/4 (hardware)": { "fp8": 1, "MXFP4 / NVFP4": 0.53 },
  "Integer (GPU)": { "int8 (W8A8 / LLM.int8)": 1, "GPTQ / AWQ int4": 0.53, "bitsandbytes NF4": 0.53, "int3": 0.4, "int2": 0.28 },
  "GGUF (llama.cpp)": { "Q8_0": 1.0625, "Q6_K": 0.82, "Q5_K_M": 0.71, "Q5_0": 0.6875, "Q4_K_M": 0.611, "Q4_0": 0.5625, "Q3_K_M": 0.489, "Q2_K": 0.419 }
};
// Quality loss = perplexity increase vs fp16, in %. [percent or null, kind]. Three kinds, never blended:
//   measured  a recorded perplexity on ONE model (Llama-2-7B, wikitext), cited below. Other models differ.
//   estimated a typical value with no source OMEN can show. Shown as "~N%", never to a decimal.
//   unknown   varies too much by method, calibration and model to give a number: null.
// Measured source: llama.cpp examples/quantize/README.md at tag b4000 (the table is marked outdated there), 7B perplexity:
// F16 5.9066, Q4_0 6.1565, Q5_0 5.9862, Q8_0 5.9070 -> (q - F16) / F16. The K-quant figures were once labelled measured but are not in that table, so they are estimated.
const QUALITY_SOURCE = "https://github.com/ggml-org/llama.cpp/blob/b4000/examples/quantize/README.md";
const QUALITY_LOSS = {
  "fp32": [0, "measured"], "fp16 / bf16": [0, "measured"], "fp8": [0.1, "estimated"], "MXFP4 / NVFP4": [null, "unknown"],
  "int8 (W8A8 / LLM.int8)": [0.1, "estimated"], "GPTQ / AWQ int4": [2, "estimated"], "bitsandbytes NF4": [2.5, "estimated"], "int3": [null, "unknown"], "int2": [null, "unknown"],
  "Q8_0": [0.007, "measured"], "Q6_K": [0.07, "estimated"], "Q5_K_M": [0.24, "estimated"], "Q5_0": [1.35, "measured"], "Q4_K_M": [0.9, "estimated"], "Q4_0": [4.23, "measured"], "Q3_K_M": [4.1, "estimated"], "Q2_K": [14.7, "estimated"]
};
// [short text, one-line basis] for a quantization's quality loss, without false precision.
function qualityText(name) {
  const [v, kind] = QUALITY_LOSS[name] || [null, "unknown"];
  if (v == null) return ["unknown", "varies too much by method, calibration and model to give a number"];
  if (v == 0) return ["none", "full-precision reference"];
  if (kind == "measured") return ["+" + v + "%", "measured: perplexity on Llama-2-7B (wikitext); other models differ"];
  return [v < 1 ? "<1%" : "~" + Math.round(v) + "%", "estimated: a typical value, not measured on this model"];
}
