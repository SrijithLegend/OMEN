// Measured runs (see runs.schema.json). Published numbers, not ours: llama.cpp CUDA scoreboard,
// Llama 2 7B Q4_0, llama-bench tg128 (decode, empty context), no flash attention, single GPU.
const S = "https://github.com/ggml-org/llama.cpp/discussions/15013";
const RUNS = [
  { model: "Llama-2-7B", quant: "Q4_0", gpu: "RTX 5090 32GB", context: 128, users: 1, engine: "llama.cpp 8cf6b42", tok_s: 290.02, source: S },
  { model: "Llama-2-7B", quant: "Q4_0", gpu: "H100 80GB SXM", context: 128, users: 1, engine: "llama.cpp 5143fa8", tok_s: 267.81, source: S },
  { model: "Llama-2-7B", quant: "Q4_0", gpu: "A100 80GB", context: 128, users: 1, engine: "llama.cpp 5143fa8", tok_s: 190.88, source: S },
  { model: "Llama-2-7B", quant: "Q4_0", gpu: "RTX 4090 24GB", context: 128, users: 1, engine: "llama.cpp 2241453", tok_s: 186.21, source: S },
  { model: "Llama-2-7B", quant: "Q4_0", gpu: "RTX 3090 24GB", context: 128, users: 1, engine: "llama.cpp c76b420", tok_s: 158.16, source: S }
];
