"""demand_model.py - demand forecasting utilities for Notebook 03."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

FEATURES = [
    "risk_label",
    "initial_loan",
    "days_since_last_loan",
    "on_time_pct",
    "utilisation_rate",
    "max_possible_increases",
]

SCENARIOS = {
    "base": {
        "description": "Neutral macro",
        "tier_multipliers": {"Prime": 1.00, "Near-Prime": 1.00, "Sub-Prime": 1.00},
        "global_shift": 0.00,
    },
    "stress": {
        "description": "Higher rates/unemployment/inflation",
        "tier_multipliers": {"Prime": 0.92, "Near-Prime": 0.84, "Sub-Prime": 0.72},
        "global_shift": -0.02,
    },
    "benign": {
        "description": "Easier macro environment",
        "tier_multipliers": {"Prime": 1.04, "Near-Prime": 1.08, "Sub-Prime": 1.12},
        "global_shift": 0.02,
    },
}


def _save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=120, bbox_inches="tight")


def build_proxy_dataset(df):
    """Create modeling dataset with proxy target.

    Proxy target: among eligible customers, `received` is treated as accepted offer.
    This is a practical proxy due missing explicit offer logs in source data.
    """
    data = df.copy()
    if "received" not in data.columns:
        data["received"] = (data["n_increases_2023"] > 0).astype(int)
    if "eligible" not in data.columns:
        data["eligible"] = (data["days_since_last_loan"] >= 60).astype(int)

    model_df = data[data["eligible"] == 1].copy()
    model_df["accepted_proxy"] = model_df["received"].astype(int)
    return model_df


def fit_acceptance_model(model_df, random_state=42):
    """Fit logistic model and return pipeline + holdout metrics."""
    X = model_df[FEATURES]
    y = model_df["accepted_proxy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), [
                "initial_loan",
                "days_since_last_loan",
                "on_time_pct",
                "utilisation_rate",
                "max_possible_increases",
            ]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["risk_label"]),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", LogisticRegression(max_iter=200, solver="lbfgs")),
        ]
    )

    pipe.fit(X_train, y_train)
    p_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test)

    # If the proxy target has near-random separability, use a robust
    # segment-based stochastic model instead of over-interpreting ML output.
    weak_signal = float(auc) < 0.53
    tier_priors = model_df.groupby("risk_label")["accepted_proxy"].mean().to_dict()

    return {
        "pipeline": None if weak_signal else pipe,
        "auc": float(auc),
        "base_rate": float(y.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "weak_signal": weak_signal,
        "mode": "segment_fallback" if weak_signal else "logit",
        "tier_priors": tier_priors,
    }


def apply_scenario(prob_base, risk_label, scenario_name):
    """Apply macro scenario multipliers to base acceptance probability."""
    sc = SCENARIOS[scenario_name]
    mult = risk_label.map(sc["tier_multipliers"]).fillna(1.0)
    out = prob_base * mult + sc["global_shift"]
    return out.clip(0.01, 0.99)


def score_scenarios(df, model_artifact):
    """Score base probability and macro scenarios for full portfolio."""
    data = df.copy()
    pipe = model_artifact["pipeline"]

    if model_artifact.get("mode") == "segment_fallback":
        # Start from empirical tier priors and add small bounded shape adjustments.
        priors = model_artifact.get("tier_priors", {})
        p = data["risk_label"].map(priors).fillna(model_artifact.get("base_rate", 0.55))
        # Gentle deterministic adjustments to preserve heterogeneity.
        recency_adj = ((data["days_since_last_loan"].clip(60, 365) - 60) / (365 - 60)) * 0.03
        util_adj = -(data["utilisation_rate"].fillna(0.5) - 0.5) * 0.04
        data["p_accept_base"] = (p + recency_adj + util_adj).clip(0.01, 0.99)
    else:
        data["p_accept_base"] = pipe.predict_proba(data[FEATURES])[:, 1].clip(0.01, 0.99)

    for sc in SCENARIOS:
        if sc == "base":
            continue
        data[f"p_accept_{sc}"] = apply_scenario(data["p_accept_base"], data["risk_label"], sc)

    # Expected accepted increases for one offer decision
    data["exp_accept_base"] = data["p_accept_base"]
    data["exp_accept_stress"] = data["p_accept_stress"]
    data["exp_accept_benign"] = data["p_accept_benign"]
    return data


def scenario_summary(df_scored):
    """Aggregate scenario KPIs by risk tier and total."""
    rows = []
    for tier, g in df_scored.groupby("risk_label"):
        rows.append({
            "segment": tier,
            "n": len(g),
            "p_base": g["p_accept_base"].mean(),
            "p_stress": g["p_accept_stress"].mean(),
            "p_benign": g["p_accept_benign"].mean(),
        })

    total = {
        "segment": "Total",
        "n": len(df_scored),
        "p_base": df_scored["p_accept_base"].mean(),
        "p_stress": df_scored["p_accept_stress"].mean(),
        "p_benign": df_scored["p_accept_benign"].mean(),
    }
    rows.append(total)
    return pd.DataFrame(rows)


def plot_probability_distributions(df_scored):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cols = ["p_accept_base", "p_accept_stress", "p_accept_benign"]
    titles = ["Base", "Stress", "Benign"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for ax, c, t, clr in zip(axes, cols, titles, colors):
        ax.hist(df_scored[c], bins=40, color=clr, alpha=0.85, edgecolor="white")
        ax.set_title(f"Acceptance Probability - {t}", fontweight="bold")
        ax.set_xlabel("p_accept")
        ax.set_ylabel("Customers")

    plt.tight_layout()
    _save(fig, "03_prob_distributions.png")
    plt.show()


def plot_scenario_by_tier(summary_df):
    core = summary_df[summary_df["segment"] != "Total"].copy()
    x = np.arange(len(core))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, core["p_stress"] * 100, width=w, label="Stress", color="#d62728")
    ax.bar(x, core["p_base"] * 100, width=w, label="Base", color="#1f77b4")
    ax.bar(x + w, core["p_benign"] * 100, width=w, label="Benign", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(core["segment"])
    ax.set_ylabel("Average p_accept (%)")
    ax.set_title("Scenario Impact by Risk Tier", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    _save(fig, "03_scenario_tier.png")
    plt.show()
