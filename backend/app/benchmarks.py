"""Benchmark reference data: sourced measurements only. Maintainers import a run (validated, with the source it came from); the public
reads them. There is no public submission, moderation or review lifecycle: every row is a citation to a published or reproducible result."""
from sqlalchemy import insert, select

from . import db
from .catalog import GPUS, MODELS, QUANTS
from .memory import KV_BYTES, weights_gib
from .performance import decode_ceiling

T = db.benchmark_runs


def validate(r: dict) -> list[str]:
    """Reasons a run cannot be stored. Mirrors validateRun in intel.js, plus physics plausibility checks."""
    bad = [f'{k} "{r[k]}" is not in the catalog' for k, table in (("model", MODELS), ("quant", QUANTS), ("gpu", GPUS)) if r[k] not in table]
    if r.get("kv_dtype", "fp16") not in KV_BYTES:
        bad.append(f"kv_dtype must be one of {sorted(KV_BYTES)}")
    if not r.get("source_url"):
        bad.append("source_url is required: a reference run must say where it came from")
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
        if r.get("vram_gib"):  # a peak-VRAM figure can establish "measured to fit", so it must be physically possible
            w = weights_gib(m["params_b"], q["bytes_per_param"])
            if r["vram_gib"] > g["vram_gib"] * n * 1.02:
                bad.append(f"vram_gib {r['vram_gib']:.1f} is more than {n} x {g['vram_gib']} GiB of this GPU")
            elif r["vram_gib"] < w * 0.9:
                bad.append(f"vram_gib {r['vram_gib']:.1f} is below the {w:.1f} GiB the weights alone need: not a peak-memory measurement")
    return bad


class Rejected(ValueError):
    """The run was not stored; .problems says why."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def import_run(run: dict) -> int:
    """Validate and store one reference run. Raises Rejected for a run that fails validation or repeats a stored one (the same setup, GPUs, count,
    KV type, engine, context and users, with tokens/s within 2%). Rows are never edited or deleted afterwards."""
    if problems := validate(run):
        raise Rejected(problems)
    with db.engine.begin() as c:
        for d in c.execute(select(T.c.id, T.c.tok_s_per_user).where(
                (T.c.model == run["model"]) & (T.c.quant == run["quant"]) & (T.c.gpu == run["gpu"]) & (T.c.engine == run["engine"])
                & (T.c.context_tokens == run["context_tokens"]) & (T.c.users == run["users"]) & (T.c.gpu_count == run["gpu_count"]) & (T.c.kv_dtype == run["kv_dtype"]))).all():
            if abs(d.tok_s_per_user - run["tok_s_per_user"]) / d.tok_s_per_user < 0.02:
                raise Rejected([f"duplicate of run {d.id}"])  # 2 GPUs or an fp8 cache is a different measurement, so those are compared too
        return c.execute(insert(T).values(**run)).inserted_primary_key[0]


def get(run_id: int) -> dict | None:
    with db.engine.connect() as c:
        r = c.execute(select(T).where(T.c.id == run_id)).mappings().first()
    return None if r is None else dict(r)


def search(**f) -> list[dict]:
    q = select(T)
    for k in ("model", "quant", "gpu", "engine"):
        if f.get(k):
            q = q.where(T.c[k] == f[k])
    with db.engine.connect() as c:
        return [dict(r) for r in c.execute(q.order_by(T.c.id)).mappings()]


def trusted_runs() -> list[dict]:
    """Every stored run, in the shape the estimators take."""
    return [{"model": r["model"], "quant": r["quant"], "gpu": r["gpu"], "gpu_count": r["gpu_count"], "engine": r["engine"], "context": r["context_tokens"],
             "users": r["users"], "tok_s": r["tok_s_per_user"], "vram_gib": r["vram_gib"], "kv_dtype": r["kv_dtype"], "token_count_method": r["token_count_method"],
             "source": r["source_url"]} for r in search()]
