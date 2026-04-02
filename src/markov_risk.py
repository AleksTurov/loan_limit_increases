"""markov_risk.py — Markov-chain risk model and plots for Notebook 02."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

STATES = ["Prime", "Near-Prime", "Sub-Prime", "Default"]
TIER_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]

# Monthly transition matrix template.
# Rows = from-state, Cols = to-state. Default is absorbing.
_T_TEMPLATE = np.array([
    [0.950, 0.030, 0.015, 0.005],   # Prime:      0.5 % monthly PD
    [0.040, 0.910, 0.035, 0.015],   # Near-Prime: 1.5 % monthly PD
    [0.010, 0.030, 0.910, 0.050],   # Sub-Prime:  5.0 % monthly PD
    [0.000, 0.000, 0.000, 1.000],   # Default:    absorbing
])


def _build_from_monthly_pd(pd_triplet, template=None):
    """Build a transition matrix by re-scaling non-default transitions.

    Args:
        pd_triplet: (pd_prime, pd_near, pd_sub) monthly default rates.
        template: base 4x4 matrix used for relative upgrade/downgrade structure.
    """
    if template is None:
        template = _T_TEMPLATE
    T = template.copy()
    for i, pd_m in enumerate(pd_triplet):
        pd_m = float(pd_m)
        scale = (1.0 - pd_m) / template[i, :3].sum()
        T[i, :3] = template[i, :3] * scale
        T[i, 3] = pd_m
    T[3] = np.array([0.0, 0.0, 0.0, 1.0])
    return T


# Reference scenario set (public benchmark anchored, portfolio-adjusted):
# - FRED DRCCLACBS (credit card delinquency rate, Q4 2025 = 2.94%)
# - FRED CORCCACBS (credit card charge-off rate, Q4 2025 = 4.11%)
# We keep monotonic tier spread and vary severity around BASE.
T_SOFT = _build_from_monthly_pd((0.0035, 0.0120, 0.0400))
T_BASE = _build_from_monthly_pd((0.0050, 0.0150, 0.0500))
T_CONSERVATIVE = _build_from_monthly_pd((0.0075, 0.0220, 0.0700))

TRANSITION_MATRICES = {
    "soft": T_SOFT,
    "base": T_BASE,
    "conservative": T_CONSERVATIVE,
}

SCENARIO_REFERENCES = {
    "source_1": "FRED DRCCLACBS (Delinquency Rate on Credit Card Loans)",
    "source_2": "FRED CORCCACBS (Charge-Off Rate on Credit Card Loans)",
    "note": "Scenario matrices are benchmark-anchored and portfolio-adjusted; not direct one-to-one estimates.",
}


def get_transition_matrix(scenario="base"):
    """Return one of: soft, base, conservative."""
    key = scenario.strip().lower()
    if key not in TRANSITION_MATRICES:
        opts = ", ".join(TRANSITION_MATRICES.keys())
        raise ValueError(f"Unknown scenario '{scenario}'. Choose one of: {opts}")
    return TRANSITION_MATRICES[key].copy()


# Backward-compatible default used in notebook code.
DEFAULT_T = T_BASE.copy()


# ── Core computations ─────────────────────────────────────────────────────────

def validate(T):
    """Assert T is a valid 4×4 row-stochastic matrix."""
    assert T.shape == (4, 4), f"Need 4×4, got {T.shape}"
    assert np.allclose(T.sum(axis=1), 1.0), f"Row sums: {T.sum(axis=1)}"
    assert (T >= 0).all(), "Negative entries"


def annual_pd(T, months=12):
    """Annual PD per non-default state via matrix power.

    Formula:
        T_m = T^months
        PD_annual(state) = T_m[state, Default]
    """
    return np.linalg.matrix_power(T, months)[:3, 3]


def stationary(T):
    """Stationary distribution (left eigenvector for eigenvalue 1)."""
    vals, vecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    return pi / pi.sum()


def project(T, pi0, steps=12):
    """Portfolio distribution trajectory for *steps* months."""
    traj = np.zeros((steps + 1, 4))
    traj[0] = pi0
    for t in range(steps):
        traj[t + 1] = traj[t] @ T
    return traj


def assign_params(df, T, lgd=0.55, ead_pct=0.50, profit=40):
    """Attach per-customer risk parameters.

    Returns DataFrame with added columns:
        pd_monthly, pd_annual, lgd, ead, el_per_increase, ev_per_increase
    """
    mpd = T[:3, 3]
    apd = annual_pd(T)
    df = df.copy()
    df["pd_monthly"] = df["risk_state"].map(dict(enumerate(mpd)))
    df["pd_annual"]  = df["risk_state"].map(dict(enumerate(apd)))
    df["lgd"] = lgd
    df["ead"] = df["initial_loan"] * ead_pct
    df["el_per_increase"] = df["pd_monthly"] * df["lgd"] * df["ead"]
    df["ev_per_increase"] = profit - df["el_per_increase"]
    return df


def sensitivity(T_base, lgd=0.55, mean_ead=2550, profit=40,
                shifts=None):
    """Sweep monthly PD ±shift → annual PD & EV table."""
    if shifts is None:
        shifts = np.arange(-0.02, 0.025, 0.005)
    rows = []
    for d in shifts:
        T = T_base.copy()
        for s in range(3):
            new_pd = np.clip(T_base[s, 3] + d, 0.001, 0.50)
            scale = (1 - new_pd) / T_base[s, :3].sum()
            T[s, :3] = T_base[s, :3] * scale
            T[s, 3] = new_pd
        apd = annual_pd(T)
        mpd = T[:3, 3]
        for i in range(3):
            el = mpd[i] * lgd * mean_ead
            rows.append(dict(shift_pp=d * 100, tier=STATES[i],
                             pd_m=mpd[i], pd_a=apd[i], el=el, ev=profit - el))
    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=120, bbox_inches="tight")


def plot_transition(T):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(T, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=STATES, yticklabels=STATES,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title("Monthly Transition Matrix T", fontweight="bold")
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    plt.tight_layout()
    _save(fig, "02_transition.png")
    plt.show()


def plot_annual_pd(T, months=12):
    pds = annual_pd(T, months)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(STATES[:3], pds * 100, color=TIER_COLORS[:3], edgecolor="white")
    for b, v in zip(bars, pds):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual PD (%)")
    ax.set_title(f"Annual PD (from T^{months})", fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_annual_pd.png")
    plt.show()
    return pds


def plot_projection(traj):
    n = len(traj) - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax.plot(range(n + 1), traj[:, i] * 100, "o-", ms=4,
                color=TIER_COLORS[i], label=STATES[i])
    ax.set_xlabel("Month")
    ax.set_ylabel("Share (%)")
    ax.set_title("12-Month Portfolio Projection", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, n + 1, 2))
    plt.tight_layout()
    _save(fig, "02_projection.png")
    plt.show()


def plot_ev_distribution(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, tier, c in zip(axes, STATES[:3], TIER_COLORS[:3]):
        sub = df[df["risk_label"] == tier]
        ax.hist(sub["ev_per_increase"], bins=40, color=c,
                edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", lw=1, ls="--")
        pct = (sub["ev_per_increase"] > 0).mean() * 100
        ax.set_title(f"{tier}  (EV>0: {pct:.0f}%)", fontweight="bold")
        ax.set_xlabel("EV per increase ($)")
        ax.set_ylabel("Customers")
    plt.suptitle("EV per Increase by Risk Tier", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "02_ev_dist.png")
    plt.show()


def plot_sensitivity(sens_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tier, c in zip(STATES[:3], TIER_COLORS[:3]):
        s = sens_df[sens_df["tier"] == tier]
        axes[0].plot(s["shift_pp"], s["pd_a"] * 100, "o-", color=c, label=tier)
        axes[1].plot(s["shift_pp"], s["ev"], "o-", color=c, label=tier)
    axes[0].set_title("Annual PD vs PD Shift", fontweight="bold")
    axes[0].set_xlabel("Monthly PD shift (pp)")
    axes[0].set_ylabel("Annual PD (%)")
    axes[1].set_title("EV per Increase vs PD Shift", fontweight="bold")
    axes[1].set_xlabel("Monthly PD shift (pp)")
    axes[1].set_ylabel("EV ($)")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "02_sensitivity.png")
    plt.show()
