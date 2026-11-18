"""Benchmark Intelligence: submission, validation (schema, plausibility, duplicates), search, verification."""
from sqlalchemy import insert, select, update

from . import db
from .catalog import GPUS, MODELS, QUANTS
from .performance import decode_ceiling

T = db.benchmark_runs


def validate(r: dict) -> list[str]:
    """Problems with a run. Mirrors validateRun in intel.js, plus a physics plausibility check."""
    bad = [f'{k} "{r[k]}" is not in the catalog' for k, table in (("model", MODELS), ("quant", QUANTS), ("gpu", GPUS)) if r[k] not in table]
    if not bad and r["users"] == 1:  # one user cannot decode faster than the bandwidth ceiling (10% slack for spec rounding)
        m, q, g = MODELS[r["model"]], QUANTS[r["quant"]], GPUS[r["gpu"]]
        ceil = decode_ceiling(m["active_params_b"] or m["params_b"], q["bytes_per_param"], g["bandwidth_gbs"])
        if r["tok_s_per_user"] > ceil * 1.1:
            bad.append(f"tok_s_per_user {r['tok_s_per_user']:.0f} exceeds the {ceil:.0f} tok/s bandwidth ceiling for this setup: implausible or wrong hardware")
    if not r.get("source_url"):
        bad.append("source_url is missing: kept as a Community record that cannot be checked")  # a warning, not a rejection
    return bad


def process(run_id: int) -> None:
    """Background job: validate a pending run, detect duplicates, set its status. A missing source alone is not fatal."""
    with db.engine.begin() as c:
        r = c.execute(select(T).where(T.c.id == run_id)).mappings().one()
        problems = validate(dict(r))
        fatal = [p for p in problems if not p.startswith("source_url")]  # only the missing-source note is non-fatal
        dup = c.execute(select(T.c.id, T.c.tok_s_per_user).where(
            (T.c.id != run_id) & (T.c.model == r["model"]) & (T.c.quant == r["quant"]) & (T.c.gpu == r["gpu"]) & (T.c.engine == r["engine"])
            & (T.c.context_tokens == r["context_tokens"]) & (T.c.users == r["users"]) & (T.c.status != "rejected"))).all()
        same = next((d.id for d in dup if abs(d.tok_s_per_user - r["tok_s_per_user"]) / d.tok_s_per_user < 0.02), None)
        if same:
            fatal.append("duplicate")
        c.execute(update(T).where(T.c.id == run_id).values(status="rejected" if fatal else "community", problems=problems + ([f"duplicate of run {same}"] if same else []), duplicate_of=same))


def submit(run: dict) -> int:
    with db.engine.begin() as c:
        return c.execute(insert(T).values(**run, status="pending", problems=[])).inserted_primary_key[0]


def search(**f) -> list[dict]:
    q = select(T).where(T.c.status.in_(["sourced", "verified", "community"]) if not f.get("status") else T.c.status == f["status"])
    for k in ("model", "quant", "gpu", "engine"):
        if f.get(k):
            q = q.where(T.c[k] == f[k])
    with db.engine.connect() as c:
        return [{k: v for k, v in r.items() if k not in ("submitter",)} for r in c.execute(q.order_by(T.c.id)).mappings()]


def verify(run_id: int) -> bool:
    with db.engine.begin() as c:
        n = c.execute(update(T).where((T.c.id == run_id) & (T.c.status == "community")).values(status="verified")).rowcount
    return n == 1
