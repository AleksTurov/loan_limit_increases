"""optimizer.py - allocation optimization utilities for Notebook 04."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def annual_to_monthly(rate_annual):
    return (1.0 + rate_annual) ** (1.0 / 12.0) - 1.0


def prepare_optimization_inputs(
    df,
    scenario="base",
    annual_discount_rate=0.19,
    increase_interval_months=2,
):
    """Build per-customer optimization vectors.

    x_i = number of offered increases to customer i.
    Objective uses expected NPV per offered increase:
        npv_unit_i = p_accept_i * ev_per_increase_i * discount_factor
    Capital usage per offered increase:
        cap_unit_i = p_accept_i * ead_i
    """
    p_col = f"p_accept_{scenario}"
    if p_col not in df.columns:
        raise ValueError(f"Missing scenario column: {p_col}")
    d = df.copy()
    # Support the Notebook 03 export schema where EAD/max caps may be absent.
    if "max_possible_increases" not in d.columns:
        d["max_possible_increases"] = 1
    if "ead" not in d.columns:
        d["ead"] = d["initial_loan"]

    r_m = annual_to_monthly(annual_discount_rate)
    discount_factor = 1.0 / ((1.0 + r_m) ** increase_interval_months)

    d["ub"] = (d["eligible"] * d["max_possible_increases"]).astype(float)
    d["npv_unit"] = d[p_col] * d["ev_per_increase"] * discount_factor
    d["cap_unit"] = d[p_col] * d["ead"]

    return d


def optimize_lp_relaxation(d, capital_limit_ratio=0.30):
    """Continuous LP relaxation with one portfolio capital constraint."""
    n = len(d)
    c = -d["npv_unit"].values  # maximize -> minimize negative

    exposure_limit = capital_limit_ratio * d["initial_loan"].sum()
    A_ub = np.array([d["cap_unit"].values])
    b_ub = np.array([exposure_limit])
    bounds = list(zip(np.zeros(n), d["ub"].values))

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    x = pd.Series(res.x, index=d.index, name="x_lp")
    return x


def optimize_greedy_integer(d, capital_limit_ratio=0.30):
    """Integer greedy allocation by value density (NPV per capital unit)."""
    out = pd.Series(0.0, index=d.index, name="x_greedy")
    exposure_limit = capital_limit_ratio * d["initial_loan"].sum()

    safe_cap = d["cap_unit"].replace(0, np.nan)
    density = d["npv_unit"] / safe_cap
    density = density.fillna(-np.inf)

    order = d.assign(density=density).sort_values(
        ["density", "npv_unit"], ascending=[False, False]
    )

    used = 0.0
    for idx, row in order.iterrows():
        if row["ub"] <= 0 or row["npv_unit"] <= 0:
            continue
        if row["cap_unit"] <= 0:
            continue

        max_by_cap = int(np.floor((exposure_limit - used) / row["cap_unit"]))
        if max_by_cap <= 0:
            continue

        take = min(int(np.floor(row["ub"])), max_by_cap)
        if take <= 0:
            continue

        out.loc[idx] = float(take)
        used += take * row["cap_unit"]
        if used >= exposure_limit:
            break

    return out


def optimize_lp_rounded(d, capital_limit_ratio=0.30):
    """Floor LP solution, then greedy-fill remaining capital."""
    x_lp = optimize_lp_relaxation(d, capital_limit_ratio=capital_limit_ratio)
    x_int = np.floor(x_lp).astype(float)

    exposure_limit = capital_limit_ratio * d["initial_loan"].sum()
    used = float((x_int * d["cap_unit"]).sum())
    rem = exposure_limit - used

    frac = (x_lp - np.floor(x_lp)).clip(lower=0)
    candidates = d.assign(frac=frac).sort_values(["frac", "npv_unit"], ascending=[False, False])

    for idx, row in candidates.iterrows():
        if rem <= 0:
            break
        if row["cap_unit"] <= 0 or row["npv_unit"] <= 0:
            continue
        ub_rem = int(np.floor(row["ub"] - x_int.loc[idx]))
        if ub_rem <= 0:
            continue
        can_take = int(np.floor(rem / row["cap_unit"]))
        take = min(ub_rem, can_take)
        if take <= 0:
            continue
        x_int.loc[idx] += float(take)
        rem -= take * row["cap_unit"]

    x_int.name = "x_lp_round"
    return x_int


def evaluate_plan(d, x, strategy_name, scenario, capital_limit_ratio=0.30):
    """Compute optimization KPIs for a strategy allocation vector x."""
    x = x.clip(lower=0)
    exposure_limit = capital_limit_ratio * d["initial_loan"].sum()

    expected_npv = float((x * d["npv_unit"]).sum())
    expected_cap = float((x * d["cap_unit"]).sum())
    offered_total = float(x.sum())
    offered_customers = int((x > 0).sum())

    tmp = d.copy()
    tmp["x"] = x.values
    mix = tmp.groupby("risk_label", observed=False)["x"].sum()
    mix_share = (mix / max(mix.sum(), 1.0)).to_dict()

    return {
        "scenario": scenario,
        "strategy": strategy_name,
        "expected_npv": expected_npv,
        "capital_used": expected_cap,
        "capital_limit": float(exposure_limit),
        "capital_usage_pct": expected_cap / max(exposure_limit, 1e-9),
        "offered_increases": offered_total,
        "offered_customers": offered_customers,
        "mix_prime": float(mix_share.get("Prime", 0.0)),
        "mix_near": float(mix_share.get("Near-Prime", 0.0)),
        "mix_sub": float(mix_share.get("Sub-Prime", 0.0)),
    }


def run_strategy_pack(d, scenario, capital_limit_ratio=0.30):
    """Run LP relaxation, LP rounded integer, and greedy integer strategies."""
    x_lp = optimize_lp_relaxation(d, capital_limit_ratio=capital_limit_ratio)
    x_lp_round = optimize_lp_rounded(d, capital_limit_ratio=capital_limit_ratio)
    x_greedy = optimize_greedy_integer(d, capital_limit_ratio=capital_limit_ratio)

    rows = [
        evaluate_plan(d, x_lp, "lp_relax", scenario, capital_limit_ratio),
        evaluate_plan(d, x_lp_round, "lp_round", scenario, capital_limit_ratio),
        evaluate_plan(d, x_greedy, "greedy", scenario, capital_limit_ratio),
    ]

    plans = {
        "lp_relax": x_lp,
        "lp_round": x_lp_round,
        "greedy": x_greedy,
    }
    return pd.DataFrame(rows), plans


def run_all_scenarios(df, scenarios=("base", "stress", "benign"), capital_limit_ratio=0.30):
    """Run strategy pack for each scenario and return summary + plans."""
    all_rows = []
    all_plans = {}

    for sc in scenarios:
        d = prepare_optimization_inputs(df, scenario=sc)
        smry, plans = run_strategy_pack(d, scenario=sc, capital_limit_ratio=capital_limit_ratio)
        all_rows.append(smry)
        all_plans[sc] = plans

    summary = pd.concat(all_rows, ignore_index=True)
    return summary, all_plans


def plot_strategy_kpis(summary_df):
    """Plot NPV and capital usage for scenario/strategy combinations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    piv_npv = summary_df.pivot(index="strategy", columns="scenario", values="expected_npv")
    piv_cap = summary_df.pivot(index="strategy", columns="scenario", values="capital_usage_pct") * 100

    piv_npv.plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("Expected NPV by Strategy")
    axes[0].set_ylabel("Expected NPV ($)")
    axes[0].grid(axis="y", alpha=0.25)

    piv_cap.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Capital Usage by Strategy")
    axes[1].set_ylabel("Capital usage (%)")
    axes[1].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "04_strategy_kpis.png", dpi=120, bbox_inches="tight")
    plt.show()
