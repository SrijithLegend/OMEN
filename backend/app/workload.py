"""Workload and Cost Intelligence: monthly volume -> capacity and cost; cloud vs self-hosted; break-even."""
HOURS_PER_MONTH = 730


def simulate(tok_s: float, users: int, usd_hr: float, out_tokens: int, requests_month: int, util: float) -> dict:
    """tok_s is one user's decode ceiling; util is the share of it you really get. Ceiling-based: ignores queueing."""
    agg = tok_s * users * util
    hours = requests_month * out_tokens / agg / 3600
    copies = max(1, -(-hours // HOURS_PER_MONTH))  # ceil
    return {
        "aggregate_tok_s": agg, "per_user_tok_s": tok_s * util, "reply_seconds": out_tokens / (tok_s * util),
        "usd_per_million_output_tokens": usd_hr / (agg * 3600) * 1e6, "gpu_hours_per_month": hours,
        "copies_needed": int(copies), "monthly_usd_pay_per_use": hours * usd_hr,
        "monthly_usd_always_on": copies * HOURS_PER_MONTH * usd_hr,
    }


def self_hosted_usd_hr(hardware_usd: float, lifetime_months: float, power_watts: float, usd_per_kwh: float,
                       hosting_usd_month: float = 0.0, utilization: float = 1.0) -> float:
    """Effective $/hour of owned hardware: amortization + power + hosting, divided by the share of time it does useful work."""
    month = hardware_usd / lifetime_months + power_watts / 1000 * HOURS_PER_MONTH * usd_per_kwh + hosting_usd_month
    return month / HOURS_PER_MONTH / utilization


def break_even_months(hardware_usd: float, cloud_usd_hr: float, own_running_usd_hr: float, hours_per_month: float) -> float | None:
    """Months until buying beats renting. None if renting is cheaper per hour than just running the owned box."""
    saving = (cloud_usd_hr - own_running_usd_hr) * hours_per_month
    return hardware_usd / saving if saving > 0 else None
