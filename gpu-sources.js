// Provenance for GPU specs. A GPU's figure counts as verified only if its field is in `checked`: read from `source` on `date`.
// Every GPU not listed here has VRAM, bandwidth and TFLOPS that were written from memory: UNVERIFIED. `arch` and `memory` appear only where the source states them.
// TFLOPS = peak dense FP16/BF16 tensor. Vendors headline a figure "with sparsity" that is 2x this; the dense half is what is stored.
// Variants: A100 and H100 figures are the SXM parts (PCIe differs: A100 80GB PCIe 1,935 GB/s; H100 PCIe 2.0 TB/s class).
const GPU_SOURCES = {
  "A100 40GB": { source: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf", date: "2026-11-19", checked: ["vram", "bandwidth", "tflops"], memory: "HBM2e", note: "SXM: 1,555 GB/s; BF16/FP16 tensor 312 dense (624 with sparsity)" },
  "A100 80GB": { source: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf", date: "2026-11-19", checked: ["vram", "bandwidth", "tflops"], memory: "HBM2e", note: "SXM: 2,039 GB/s; PCIe is 1,935 GB/s" },
  "H100 80GB SXM": { source: "https://www.nvidia.com/en-us/data-center/h100/", date: "2026-11-19", checked: ["vram", "bandwidth", "tflops"], arch: "Hopper", memory: "HBM3", note: "SXM: 3.35 TB/s; FP16 tensor 1,979 with sparsity = 989 dense" },
  "H200 141GB": { source: "https://www.nvidia.com/en-us/data-center/h200/", date: "2026-11-19", checked: ["vram", "bandwidth", "tflops"], arch: "Hopper", memory: "HBM3e", note: "4.8 TB/s; FP16 tensor 1,979 with sparsity = 989 dense" },
  "L4 24GB": { source: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/l4/PB-11316-001_v01.pdf", date: "2026-11-19", checked: ["vram", "bandwidth"], arch: "Ada Lovelace", memory: "GDDR6", note: "TFLOPS not in the product brief: unverified" },
  "T4 16GB": { source: "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf", date: "2026-11-19", checked: ["vram", "bandwidth", "tflops"], arch: "Turing", memory: "GDDR6", note: "Datasheet says 300 GB/s (earlier OMEN value 320 was from memory); mixed-precision FP16/FP32 65 TFLOPS" },
  "MI300X 192GB": { source: "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html", date: "2026-11-19", checked: ["vram", "bandwidth", "tflops"], arch: "CDNA 3", memory: "HBM3", note: "5.3 TB/s; FP16 1307.4 TFLOPS peak theoretical (dense)" }
};
// "verified" = every figure OMEN uses for this GPU is checked; "partial" = some; "unverified" = none.
function gpuSpecStatus(name, hasTflops = true) {
  const s = GPU_SOURCES[name], need = hasTflops ? ["vram", "bandwidth", "tflops"] : ["vram", "bandwidth"];
  return !s ? "unverified" : need.every(f => s.checked.includes(f)) ? "verified" : "partial";
}
