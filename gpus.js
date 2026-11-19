// [VRAM GiB, memory bandwidth GB/s, assumed $/hr]. $/hr is an ASSUMED cloud/amortized rate and changes constantly.
// VRAM and bandwidth: only the GPUs listed in gpu-sources.js were checked against a vendor page or datasheet (with date). All others were written from memory: UNVERIFIED.
// Apple: unified memory, only part is usable by the GPU, so raise overhead if you care.
const GPUS_BY_VENDOR = {
  "NVIDIA GeForce": {
    "RTX 3060 12GB": [12, 360, 0.10], "RTX 3080 10GB": [10, 760, 0.15], "RTX 3090 24GB": [24, 936, 0.25],
    "RTX 4060 Ti 16GB": [16, 288, 0.12], "RTX 4070 Ti Super 16GB": [16, 672, 0.20], "RTX 4080 16GB": [16, 717, 0.28],
    "RTX 4090 24GB": [24, 1008, 0.40], "RTX 5080 16GB": [16, 960, 0.35], "RTX 5090 32GB": [32, 1792, 0.70] },
  "NVIDIA Datacenter/Pro": {
    "T4 16GB": [16, 300, 0.35], "L4 24GB": [24, 300, 0.70], "A10G 24GB": [24, 600, 1.00],
    "RTX A6000 48GB": [48, 768, 0.50], "RTX 6000 Ada 48GB": [48, 960, 0.90], "L40S 48GB": [48, 864, 1.80],
    "A100 40GB": [40, 1555, 1.30], "A100 80GB": [80, 2039, 1.80], "H100 80GB SXM": [80, 3350, 3.00],
    "H200 141GB": [141, 4800, 3.80], "B200 192GB": [192, 8000, 6.00] },
  "AMD": {
    "RX 7900 XTX 24GB": [24, 960, 0.30], "MI250X 128GB": [128, 3277, 1.50],
    "MI300X 192GB": [192, 5300, 2.50], "MI325X 256GB": [256, 6000, 3.50] },
  "Intel": { "Arc A770 16GB": [16, 560, 0.12], "Gaudi 2 96GB": [96, 2450, 1.50] },
  "Apple Silicon": {
    "M3 Max 128GB": [128, 400, 0.20], "M4 Max 128GB": [128, 546, 0.25],
    "M2 Ultra 192GB": [192, 800, 0.35], "M3 Ultra 512GB": [512, 819, 0.50] },
  "Cloud accelerators": {
    "Google TPU v5e 16GB": [16, 819, 1.20], "Google TPU v5p 95GB": [95, 2765, 4.20],
    "AWS Inferentia2 32GB": [32, 820, 0.76], "AWS Trainium2 96GB": [96, 2900, 3.00] },
  // CPU-only inference: speed is limited by system RAM bandwidth = channels * per-channel GB/s.
  "CPU / System RAM": Object.fromEntries([ // [name, ram type, channels, GiB, assumed $/hr]
    ["Desktop DDR4-3200 2ch 32GB", "DDR4-3200", 2, 32, 0.05],
    ["Laptop DDR5-5600 2ch 32GB", "DDR5-5600", 2, 32, 0.05],
    ["Desktop DDR5-6400 2ch 64GB", "DDR5-6400", 2, 64, 0.08],
    ["Workstation DDR5-5600 4ch 128GB", "DDR5-5600", 4, 128, 0.25],
    ["AI mini-PC LPDDR5X-8000 4ch 128GB", "LPDDR5X-8000", 4, 128, 0.15],
    ["Server Xeon DDR4-3200 8ch 512GB", "DDR4-3200", 8, 512, 0.60],
    ["Server EPYC DDR5-4800 12ch 768GB", "DDR5-4800", 12, 768, 1.00],
    ["Server EPYC DDR5-6400 12ch 768GB", "DDR5-6400", 12, 768, 1.30],
  ].map(([n, t, ch, gib, usd]) => [n, [gib, +(SYSRAM_GBS_PER_CHANNEL[t] * ch).toFixed(1), usd]]))
};
