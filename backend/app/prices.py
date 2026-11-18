"""Cloud price catalog. Prices are entered by a reviewer with a source and date: OMEN ships none, and never invents one."""
from sqlalchemy import insert, select

from . import db

PROVIDERS = ["AWS", "Google Cloud", "Microsoft Azure", "Oracle Cloud", "CoreWeave", "Lambda", "RunPod", "Vast.ai", "Paperspace", "DigitalOcean", "Together AI", "Other"]
PRICING_MODELS = ["on_demand", "reserved", "spot"]
T = db.pricing_records


def add(rec: dict) -> int:
    with db.engine.begin() as c:
        return c.execute(insert(T).values(**rec)).inserted_primary_key[0]


def search(*, provider=None, gpu=None, pricing_model=None, latest_only=True) -> list[dict]:
    q = select(T)
    for k, v in (("provider", provider), ("gpu", gpu), ("pricing_model", pricing_model)):
        if v:
            q = q.where(T.c[k] == v)
    with db.engine.connect() as c:
        rows = [dict(r) | {"usd_hr_per_gpu": r["usd_hr"] / r["gpu_count"]} for r in c.execute(q.order_by(T.c.observed_at.desc(), T.c.id.desc())).mappings()]
    if latest_only:  # newest observation per (provider, instance, region, pricing model); older rows stay in the table as history
        seen, out = set(), []
        for r in rows:
            k = (r["provider"], r["instance_type"], r["region"], r["pricing_model"])
            if k not in seen:
                seen.add(k)
                out.append(r)
        rows = out
    return rows


def cheapest_per_gpu(provider: str, gpu: str, pricing_model: str) -> dict | None:
    rows = search(provider=provider, gpu=gpu, pricing_model=pricing_model)
    return min(rows, key=lambda r: r["usd_hr_per_gpu"]) if rows else None
