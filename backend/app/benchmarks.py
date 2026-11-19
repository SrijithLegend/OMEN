"""Benchmark Intelligence: submission, validation (schema, plausibility, duplicates), search, verification."""
from sqlalchemy import insert, select, update

from . import db
from .catalog import GPUS, MODELS, QUANTS
from .memory import KV_BYTES, weights_gib
from .performance import decode_ceiling

T = db.benchmark_runs


def validate(r: dict) -> list[str]:
    """Problems with a run. Mirrors validateRun in intel.js, plus a physics plausibility check."""
    bad = [f'{k} "{r[k]}" is not in the catalog' for k, table in (("model", MODELS), ("quant", QUANTS), ("gpu", GPUS)) if r[k] not in table]
    if r.get("kv_dtype", "fp16") not in KV_BYTES:
        bad.append(f"kv_dtype must be one of {sorted(KV_BYTES)}")
    if not bad:
        m, q, g, n = MODELS[r["model"]], QUANTS[r["quant"]], GPUS[r["gpu"]], r.get("gpu_count", 1)
        ceil = decode_ceiling(m["active_params_b"] or m["params_b"], q["bytes_per_param"], g["bandwidth_gbs"], n)
        if r["users"] == 1 and r["tok_s_per_user"] > ceil * 1.1:  # one user cannot decode faster than the bandwidth ceiling (10% slack for spec rounding)
            bad.append(f"tok_s_per_user {r['tok_s_per_user']:.0f} exceeds the {ceil:.0f} tok/s bandwidth ceiling for this setup: implausible or wrong hardware")
        pg = r.get("vram_gib_per_gpu")
        if pg:  # per-GPU peaks must agree with the total and with the hardware, or the memory claim is not credible
            if not r.get("vram_gib"):
                bad.append("vram_gib_per_gpu needs vram_gib")
            elif len(pg) != n:
                bad.append(f"vram_gib_per_gpu lists {len(pg)} GPUs but gpu_count is {n}")
            elif not 0.9 * sum(pg) <= r["vram_gib"] <= 1.01 * sum(pg):  # the peak of the sum cannot exceed the sum of the peaks
                bad.append(f"vram_gib {r['vram_gib']:.1f} does not match the per-GPU peaks, which sum to {sum(pg):.1f}")
            elif max(pg) > g["vram_gib"] * 1.02:
                bad.append(f"a per-GPU peak of {max(pg):.1f} GiB is more than the {g['vram_gib']} GiB this GPU has")
        if r.get("vram_gib"):  # a peak-VRAM figure can promote a setup to "measured to fit", so it must be physically possible
            w = weights_gib(m["params_b"], q["bytes_per_param"])
            if r["vram_gib"] > g["vram_gib"] * n * 1.02:
                bad.append(f"vram_gib {r['vram_gib']:.1f} is more than {n} x {g['vram_gib']} GiB of this GPU")
            elif r["vram_gib"] < w * 0.9:
                bad.append(f"vram_gib {r['vram_gib']:.1f} is below the {w:.1f} GiB the weights alone need: not a peak-memory measurement")
    if r.get("token_count_method") == "chunks":
        bad.append("note: tokens were estimated from streamed chunks, so tokens/s may be off")  # a warning, not a rejection
    if not r.get("source_url"):
        bad.append("source_url is missing: kept as a Community record that cannot be checked")  # a warning, not a rejection
    return bad


def process(run_id: int) -> None:
    """Background job: validate a pending run, detect duplicates, set its status. A missing source alone is not fatal."""
    with db.engine.begin() as c:
        r = c.execute(select(T).where(T.c.id == run_id)).mappings().one()
        problems = validate(dict(r))
        fatal = [p for p in problems if not p.startswith(("source_url", "note:"))]  # missing source and estimated token counts are warnings
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


def _public(r) -> dict:
    """A run as the public sees it: the submitter name only if they opted in to attribution."""
    return {k: v for k, v in r.items() if k != "attribute_me" and (k != "submitter" or r["attribute_me"])}


def get(run_id: int) -> dict | None:
    with db.engine.connect() as c:
        r = c.execute(select(T).where(T.c.id == run_id)).mappings().first()
    return None if r is None else _public(r)


def search(**f) -> list[dict]:
    q = select(T).where(T.c.status.in_(["sourced", "verified", "community"]) if not f.get("status") else T.c.status == f["status"])
    for k in ("model", "quant", "gpu", "engine"):
        if f.get(k):
            q = q.where(T.c[k] == f[k])
    with db.engine.connect() as c:
        return [_public(r) for r in c.execute(q.order_by(T.c.id)).mappings()]


def verify(run_id: int) -> bool:
    with db.engine.begin() as c:
        n = c.execute(update(T).where((T.c.id == run_id) & (T.c.status == "community")).values(status="verified")).rowcount
    return n == 1


def trusted_runs() -> list[dict]:
    """Sourced and verified runs in the shape the estimators take. Community runs are left out until a reviewer verifies them."""
    with db.engine.connect() as c:
        rows = c.execute(select(T).where(T.c.status.in_(["sourced", "verified"])).order_by(T.c.id)).mappings()
        return [{"model": r["model"], "quant": r["quant"], "gpu": r["gpu"], "gpu_count": r["gpu_count"], "engine": r["engine"], "context": r["context_tokens"],
                 "users": r["users"], "tok_s": r["tok_s_per_user"], "vram_gib": r["vram_gib"], "kv_dtype": r["kv_dtype"], "token_count_method": r["token_count_method"], "source": r["source_url"]} for r in rows]
