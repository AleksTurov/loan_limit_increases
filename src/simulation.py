"""simulation.py - Monte Carlo utilities for Notebook 05."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def discount_factor(annual_rate=0.19, months=2):
    monthly = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    return 1.0 / ((1.0 + monthly) ** months)


def build_sim_input(df_demand, df_alloc, scenario, lgd=0.55, ead_pct=0.50):
    """Prepare per-customer simulation table for one scenario."""
    p_col = f"p_accept_{scenario}"
    if p_col not in df_demand.columns:
        raise ValueError(f"Missing column: {p_col}")

    rec = df_alloc[df_alloc["scenario"] == scenario].copy()
    if rec.empty:
        raise ValueError(f"No allocation rows for scenario={scenario}")

    d = df_demand.copy()
    if "ead" not in d.columns:
        d["ead"] = d["initial_loan"] * ead_pct

    keep = [
        "customer_id",
        "risk_label",
        "initial_loan",
        "pd_annual",
        "ead",
        p_col,
    ]
    out = d[keep].merge(
        rec[["customer_id", "x_recommended"]],
        on="customer_id",
        how="left",
    )

    out["x_recommended"] = out["x_recommended"].fillna(0.0)
    out["offers"] = np.floor(out["x_recommended"].clip(lower=0)).astype(int)
    out["eligible_offer"] = (out["offers"] > 0).astype(int)

    out["p_accept"] = out[p_col].clip(0.0, 1.0)
    out["pd_annual"] = out["pd_annual"].clip(0.0, 1.0)
    out["lgd"] = float(lgd)

    return out


def run_monte_carlo(
    sim_input,
    n_sims=5000,
    seed=42,
    annual_rate=0.19,
    months=2,
    p_accept_mult=1.0,
    pd_mult=1.0,
):
    """Run Monte Carlo draws for portfolio NPV and risk metrics."""
    rng = np.random.default_rng(seed)
    d = sim_input.copy()

    offers = d["offers"].to_numpy(dtype=int)
    p_accept = np.clip(d["p_accept"].to_numpy(dtype=float) * p_accept_mult, 0.0, 1.0)
    pd_annual = np.clip(d["pd_annual"].to_numpy(dtype=float) * pd_mult, 0.0, 1.0)
    ead = d["ead"].to_numpy(dtype=float)
    lgd = d["lgd"].to_numpy(dtype=float)

    disc = discount_factor(annual_rate=annual_rate, months=months)

    rows = []
    for i in range(n_sims):
        accepted = rng.binomial(offers, p_accept)
        defaults = rng.binomial(accepted, pd_annual)

        gross_profit = accepted * 40.0
        credit_loss = defaults * lgd * ead
        npv = float((gross_profit - credit_loss).sum() * disc)

        rows.append(
            {
                "sim_id": i,
                "npv": npv,
                "accepted": int(accepted.sum()),
                "defaults": int(defaults.sum()),
                "capital_used": float((accepted * ead).sum()),
            }
        )

    return pd.DataFrame(rows)


def summarize_simulation(sim_df):
    """Summarize simulation distribution including downside metrics."""
    x = sim_df["npv"].to_numpy(dtype=float)
    p05 = float(np.quantile(x, 0.05))
    p50 = float(np.quantile(x, 0.50))
    p95 = float(np.quantile(x, 0.95))
    tail = x[x <= p05]
    cvar95 = float(tail.mean()) if len(tail) else p05

    return {
        "npv_mean": float(np.mean(x)),
        "npv_std": float(np.std(x, ddof=1)),
        "npv_p05": p05,
        "npv_p50": p50,
        "npv_p95": p95,
        "var95": p05,
        "cvar95": cvar95,
        "avg_accepted": float(sim_df["accepted"].mean()),
        "avg_defaults": float(sim_df["defaults"].mean()),
        "avg_capital_used": float(sim_df["capital_used"].mean()),
    }


def run_scenario_pack(df_demand, df_alloc, scenarios=("base", "stress", "benign"), n_sims=5000, seed=42):
    """Run Monte Carlo for each scenario and return summary + raw simulations."""
    out_rows = []
    sims = {}

    for i, sc in enumerate(scenarios):
        inp = build_sim_input(df_demand, df_alloc, scenario=sc)
        sim = run_monte_carlo(inp, n_sims=n_sims, seed=seed + i)
        sm = summarize_simulation(sim)
        sm["scenario"] = sc
        out_rows.append(sm)
        sims[sc] = sim

    summary = pd.DataFrame(out_rows)
    cols = [
        "scenario",
        "npv_mean",
        "npv_std",
        "npv_p05",
        "npv_p50",
        "npv_p95",
        "var95",
        "cvar95",
        "avg_accepted",
        "avg_defaults",
        "avg_capital_used",
    ]
    return summary[cols], sims


def run_sensitivity(
    df_demand,
    df_alloc,
    scenario="base",
    p_accept_multipliers=(0.90, 1.00, 1.10),
    pd_multipliers=(0.90, 1.00, 1.10),
    n_sims=3000,
    seed=123,
):
    """2D sensitivity grid for p_accept and PD multipliers."""
    inp = build_sim_input(df_demand, df_alloc, scenario=scenario)
    rows = []

    k = 0
    for pa in p_accept_multipliers:
        for pdm in pd_multipliers:
            sim = run_monte_carlo(
                inp,
                n_sims=n_sims,
                seed=seed + k,
                p_accept_mult=float(pa),
                pd_mult=float(pdm),
            )
            sm = summarize_simulation(sim)
            rows.append(
                {
                    "scenario": scenario,
                    "p_accept_mult": float(pa),
                    "pd_mult": float(pdm),
                    "npv_mean": sm["npv_mean"],
                    "npv_p05": sm["npv_p05"],
                    "cvar95": sm["cvar95"],
                    "avg_defaults": sm["avg_defaults"],
                }
            )
            k += 1

    return pd.DataFrame(rows)


def plot_npv_distributions(sims_dict):
    """Plot NPV histograms by scenario."""
    fig, axes = plt.subplots(1, len(sims_dict), figsize=(5.8 * len(sims_dict), 4.5))
    if len(sims_dict) == 1:
        axes = [axes]

    colors = {"base": "#1f77b4", "stress": "#d62728", "benign": "#2ca02c"}

    for ax, (sc, sim) in zip(axes, sims_dict.items()):
        c = colors.get(sc, "#4c4c4c")
        ax.hist(sim["npv"], bins=40, color=c, alpha=0.85, edgecolor="white")
        ax.axvline(sim["npv"].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_title(f"NPV Distribution - {sc}")
        ax.set_xlabel("Portfolio NPV")
        ax.set_ylabel("Simulations")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "05_npv_distributions.png", dpi=120, bbox_inches="tight")
    plt.show()


def plot_sensitivity_heatmap(sens_df, value_col="npv_mean"):
    """Plot simple heatmap for sensitivity table."""
    p_vals = sorted(sens_df["p_accept_mult"].unique())
    pd_vals = sorted(sens_df["pd_mult"].unique())

    mat = np.zeros((len(pd_vals), len(p_vals)))
    for i, pdm in enumerate(pd_vals):
        for j, pa in enumerate(p_vals):
            v = sens_df[(sens_df["p_accept_mult"] == pa) & (sens_df["pd_mult"] == pdm)][value_col].iloc[0]
            mat[i, j] = v

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(p_vals)))
    ax.set_xticklabels([f"{x:.2f}" for x in p_vals])
    ax.set_yticks(range(len(pd_vals)))
    ax.set_yticklabels([f"{x:.2f}" for x in pd_vals])
    ax.set_xlabel("p_accept multiplier")
    ax.set_ylabel("PD multiplier")
    ax.set_title(f"Sensitivity Heatmap - {value_col}")

    for i in range(len(pd_vals)):
        for j in range(len(p_vals)):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, label=value_col)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"05_sensitivity_{value_col}.png", dpi=120, bbox_inches="tight")
    plt.show()
