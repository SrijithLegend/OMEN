// Bytes per weight, including scale/zero-point overhead (bits per weight / 8).
// GGUF bpw from llama.cpp; others are typical values. ESTIMATED: real files vary by model.
const QUANTS_BY_FAMILY = {
  "Full precision": { "fp32": 4, "fp16 / bf16": 2 },
  "Float 8/4 (hardware)": { "fp8": 1, "MXFP4 / NVFP4": 0.53 },
  "Integer (GPU)": { "int8 (W8A8 / LLM.int8)": 1, "GPTQ / AWQ int4": 0.53, "bitsandbytes NF4": 0.53, "int3": 0.4, "int2": 0.28 },
  "GGUF (llama.cpp)": { "Q8_0": 1.0625, "Q6_K": 0.82, "Q5_K_M": 0.71, "Q5_0": 0.6875, "Q4_K_M": 0.611, "Q4_0": 0.5625, "Q3_K_M": 0.489, "Q2_K": 0.328 }
};
// Quality loss: perplexity increase vs fp16, in %. [percent, measured?]
// Measured = llama.cpp quantize README, Llama-2-7B wikitext (delta ppl / 5.9066). The rest are TYPICAL values
// from papers and community evals; real loss varies by model (small models and low bits lose more).
const QUALITY_LOSS = {
  "fp32": [0, 1], "fp16 / bf16": [0, 1], "fp8": [0.1, 0], "MXFP4 / NVFP4": [2, 0],
  "int8 (W8A8 / LLM.int8)": [0.1, 0], "GPTQ / AWQ int4": [2, 0], "bitsandbytes NF4": [2.5, 0], "int3": [8, 0], "int2": [40, 0],
  "Q8_0": [0.007, 1], "Q6_K": [0.07, 1], "Q5_K_M": [0.24, 1], "Q5_0": [1.2, 1], "Q4_K_M": [0.9, 1], "Q4_0": [3.7, 1], "Q3_K_M": [4.1, 1], "Q2_K": [14.7, 1]
};
