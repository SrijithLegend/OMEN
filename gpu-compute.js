// Peak dense FP16/BF16 tensor TFLOPS (no sparsity), checked against a vendor page only for the GPUs in gpu-sources.js; the rest were written from memory (UNVERIFIED).
// Only GPUs where the figure is well known are listed. A GPU missing here means "unknown", and prefill is not estimated for it.
const GPU_TFLOPS = {
  "T4 16GB": 65, "L4 24GB": 121, "L40S 48GB": 362, "A100 40GB": 312, "A100 80GB": 312, "H100 80GB SXM": 989,
  "H200 141GB": 989, "B200 192GB": 2250, "MI300X 192GB": 1307, "RTX 4090 24GB": 165, "RTX 3090 24GB": 71
};
