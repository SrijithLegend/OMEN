// Memory types. Bandwidth (GB/s) = gbps_per_pin * bus_bits / 8. ESTIMATED: typical shipping speeds from memory.
const VRAM_TYPES = { // gbps per pin [min, max], bus bits per chip/stack
  "GDDR5": { gbps: [5, 8], bits: 32 }, "GDDR6": { gbps: [14, 20], bits: 32 },
  "GDDR6X": { gbps: [19, 24], bits: 32 }, "GDDR7": { gbps: [28, 36], bits: 32 },
  "LPDDR5 / LPDDR5X": { gbps: [6.4, 9.6], bits: 16 },
  "HBM2": { gbps: [2, 2.4], bits: 1024 }, "HBM2e": { gbps: [3.2, 3.6], bits: 1024 },
  "HBM3": { gbps: [5.2, 6.4], bits: 1024 }, "HBM3e": { gbps: [6.4, 9.6], bits: 1024 },
  "HBM4": { gbps: [8, 10], bits: 2048 }
};
const SYSRAM_GBS_PER_CHANNEL = { // 64-bit channel
  "DDR4-2666": 21.3, "DDR4-3200": 25.6, "DDR5-4800": 38.4, "DDR5-5600": 44.8, "DDR5-6400": 51.2,
  "LPDDR5X-8000": 64, "LPDDR5X-8533": 68.3
};
