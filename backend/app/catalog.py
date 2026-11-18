"""Model, quantization and GPU catalog. Data is generated from the calculator's JS files (scripts/export-catalog.js)."""
import json
from pathlib import Path

CATALOG_BYTES = (Path(__file__).parent / "catalog.json").read_bytes()
CATALOG = json.loads(CATALOG_BYTES)
MODELS, QUANTS, GPUS, RUNS = CATALOG["models"], CATALOG["quants"], CATALOG["gpus"], CATALOG["runs"]


def get(table: dict, name: str, kind: str) -> dict:
    if name not in table:
        raise KeyError(f"unknown {kind}: {name}")
    return table[name]
