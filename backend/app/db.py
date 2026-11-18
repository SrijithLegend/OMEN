"""Database: SQLAlchemy Core, SQLite by default. Set OMEN_DB=postgresql+psycopg://user:pass@host/omen for PostgreSQL (needs psycopg installed).
Units are explicit in column names. Benchmark rows are never overwritten: only their status moves forward."""
import os

from sqlalchemy import (JSON, Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine, func, insert, select)
from sqlalchemy.pool import StaticPool

URL = os.environ.get("OMEN_DB", "sqlite:///omen.db")
engine = create_engine(URL, **({"poolclass": StaticPool, "connect_args": {"check_same_thread": False}} if URL.startswith("sqlite") else {}))
meta = MetaData()

benchmark_runs = Table(
    "benchmark_runs", meta,
    Column("id", Integer, primary_key=True),
    Column("model", String, nullable=False), Column("quant", String, nullable=False), Column("gpu", String, nullable=False),
    Column("engine", String, nullable=False), Column("context_tokens", Integer, nullable=False), Column("users", Integer, nullable=False),
    Column("tok_s_per_user", Float, nullable=False), Column("ttft_ms", Float), Column("tpot_ms", Float), Column("vram_gib", Float),
    Column("source_url", String), Column("submitter", String), Column("consent_to_publish", Boolean, nullable=False, default=False),
    Column("attribute_me", Boolean, nullable=False, default=False),  # opt-in: show the submitter name publicly
    Column("status", String, nullable=False),  # pending -> community | rejected; sourced (imported); verified (reviewer)
    Column("problems", JSON, nullable=False, default=list),
    Column("duplicate_of", Integer), Column("created_at", String, server_default=func.now()),
)
pricing_records = Table(  # append-only: a new price is a new row, so a saved report can cite the price it used
    "pricing_records", meta,
    Column("id", Integer, primary_key=True), Column("provider", String, nullable=False), Column("instance_type", String, nullable=False),
    Column("region", String, nullable=False), Column("gpu", String, nullable=False), Column("gpu_count", Integer, nullable=False),
    Column("pricing_model", String, nullable=False), Column("usd_hr", Float, nullable=False),  # whole-instance price per hour
    Column("source_url", String, nullable=False), Column("observed_at", String, nullable=False),  # ISO date the price was read
)
saved_analyses = Table(
    "saved_analyses", meta,
    Column("id", Integer, primary_key=True), Column("owner", String, nullable=False), Column("project", String, nullable=False),
    Column("note", Text, default=""), Column("kind", String, nullable=False),  # estimate | workload | recommendation
    Column("input_snapshot", JSON, nullable=False), Column("output_snapshot", JSON, nullable=False),
    Column("catalog_hash", String, nullable=False), Column("created_at", String, server_default=func.now()),
)
api_keys = Table(
    "api_keys", meta,
    Column("id", Integer, primary_key=True), Column("owner", String, nullable=False, unique=True),
    Column("key_hash", String, nullable=False, unique=True), Column("calls_this_minute", Integer, default=0), Column("minute", Integer, default=0),
)


def init(seed_runs: list[dict]) -> None:
    meta.create_all(engine)
    with engine.begin() as c:
        if c.execute(select(func.count()).select_from(benchmark_runs)).scalar() == 0:
            for r in seed_runs:
                c.execute(insert(benchmark_runs).values(
                    model=r["model"], quant=r["quant"], gpu=r["gpu"], engine=r["engine"], context_tokens=r["context"], users=r["users"],
                    tok_s_per_user=r["tok_s"], source_url=r["source"], submitter=None, consent_to_publish=True, status="sourced", problems=[]))
