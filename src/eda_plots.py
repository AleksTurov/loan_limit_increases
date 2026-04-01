"""eda_plots.py — all plotting helpers for the unified 01_EDA notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest, chi2_contingency, f_oneway


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def _save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=120, bbox_inches="tight")


# ── 1. Distributions ─────────────────────────────────────────────────────────

def plot_distributions(df, eligibility_days=60):
    """2×3 grid: continuous histograms (top) + discrete counts (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")

    specs = [
        ("initial_loan",         "steelblue",  "navy",        "Initial Loan (USD)",     [], []),
        ("days_since_last_loan", "darkorange", "saddlebrown", "Days Since Last Loan",
            [eligibility_days], ["60d eligibility"]),
        ("on_time_pct",          "seagreen",   "darkgreen",   "On-Time Payment (%)",
            [85, 95], ["85% Near-Prime", "95% Prime"]),
    ]
    for ax, (col, fc, lc, title, vlines, vlabels) in zip(axes[0], specs):
        ax.hist(df[col], bins=50, color=fc, edgecolor="white", alpha=0.8, density=True)
        df[col].plot(kind="kde", ax=ax, color=lc, lw=2)
        for v, lbl in zip(vlines, vlabels):
            ax.axvline(v, color="red", linestyle="--", lw=1.5, label=lbl)
        if vlines:
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Density")

    all_n = list(range(7))
    counts = pd.Series(0, index=all_n)
    counts.update(df["n_increases_2023"].value_counts())
    colors_n = ["#e74c3c" if counts[v] == 0 else "#3498db" for v in all_n]
    bars = axes[1][0].bar([str(v) for v in all_n], counts.values, color=colors_n, edgecolor="white")
    axes[1][0].set_title("n_increases_2023  (red = absent)")
    axes[1][0].set_xlabel("n_increases")
    axes[1][0].set_ylabel("Customers")
    for bar, v, c in zip(bars, all_n, counts.values):
        lbl = "ABSENT" if c == 0 else f"{c:,}"
        clr = "#e74c3c" if c == 0 else "black"
        axes[1][0].text(bar.get_x() + bar.get_width() / 2,
                        max(c, counts.max() * 0.02) + counts.max() * 0.01,
                        lbl, ha="center", fontsize=8, color=clr)

    cohort = (df.groupby("n_increases_2023")
                .agg(customers=("customer_id", "count"), total_profit=("total_profit", "sum"))
                .reset_index())
    cohort["pct"] = cohort["total_profit"] / max(cohort["total_profit"].sum(), 1) * 100
    bar2 = axes[1][1].bar(cohort["n_increases_2023"].astype(str),
                           cohort["total_profit"] / 1000,
                           color=["#c0392b", "#2980b9", "#27ae60", "#8e44ad"][:len(cohort)],
                           edgecolor="white")
    axes[1][1].set_title("Portfolio Profit by n_increases ($K)")
    axes[1][1].set_xlabel("n_increases")
    axes[1][1].set_ylabel("Profit ($K)")
    for bar, val in zip(bar2, cohort["pct"]):
        axes[1][1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", fontsize=9, fontweight="bold")

    axes[1][2].hist(df["total_profit"], bins=15, color="crimson", edgecolor="white", alpha=0.85)
    axes[1][2].set_title("Profit Distribution")
    axes[1][2].set_xlabel("Total Profit ($)")
    axes[1][2].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    _save(fig, "01_distributions.png")
    plt.show()


# ── 2. Correlations ──────────────────────────────────────────────────────────

def plot_correlation(df, cols, labels):
    """Pearson heatmap. Returns the correlation matrix."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1)
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "01_correlation.png")
    plt.show()
    return corr


# ── 3. Anomaly plots ─────────────────────────────────────────────────────────

def plot_a1_missing_n(df, max_n=6):
    """A1: only {0,3,4,5} appear in n_increases_2023."""
    all_values = list(range(max_n + 1))
    obs = df["n_increases_2023"].value_counts().reindex(all_values, fill_value=0)
    colors = ["#e74c3c" if obs[v] == 0 else "#3498db" for v in all_values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(all_values, obs.values, color=colors, edgecolor="white", width=0.7)
    for bar, v, c in zip(bars, all_values, obs.values):
        lbl = "ABSENT" if c == 0 else f"{c:,}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(c, obs.max() * 0.02) + obs.max() * 0.01,
                lbl, ha="center", fontsize=9, color="#e74c3c" if c == 0 else "black")
    ax.set_xlabel("n_increases_2023")
    ax.set_ylabel("Count")
    ax.set_title("A1 — n_increases: only {0, 3, 4, 5} ever occur  (red = absent)", fontsize=11)
    ax.set_xticks(all_values)
    plt.tight_layout()
    _save(fig, "01_a1_missing_n.png")
    plt.show()


def plot_a2_otp_analysis(df):
    """A2: floor at 80%, Uniform shape, threshold positions."""
    otp = df["on_time_pct"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("A2 — on_time_pct: Hard Floor at 80%  +  Threshold Position", fontsize=12, fontweight="bold")

    otp.plot(kind="kde", ax=axes[0], color="steelblue", lw=2)
    axes[0].hist(otp, bins=60, density=True, color="steelblue", alpha=0.25, edgecolor="white")
    for v, col, lbl in [(80, "#e74c3c", "80% floor"), (85, "#f39c12", "85% spec"), (95, "#27ae60", "95% spec")]:
        axes[0].axvline(v, color=col, linestyle="--", lw=1.8, label=lbl)
    axes[0].set_xlabel("on_time_pct (%)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Uniform KDE — no natural breaks\nat spec thresholds")
    axes[0].legend(fontsize=8)

    percs = np.arange(0, 101)
    pvals = np.percentile(otp, percs)
    axes[1].plot(percs, pvals, color="steelblue", lw=2)
    for t, col, lbl in [(85, "#f39c12", "85%"), (95, "#27ae60", "95%")]:
        rank = (otp < t).mean() * 100
        axes[1].axhline(t, color=col, linestyle="--", lw=1.5, label=f"{lbl} = {rank:.0f}th pct")
        axes[1].axvline(rank, color=col, linestyle=":", lw=1)
    axes[1].set_xlabel("Percentile")
    axes[1].set_ylabel("on_time_pct value")
    axes[1].set_title("Thresholds cut at 25th / 75th pct\n(arbitrary in Uniform data)")
    axes[1].legend(fontsize=8)

    edges = np.arange(80, 101, 1)
    cnt, _ = np.histogram(otp, bins=edges)
    cols_b = ["#e74c3c" if (b == 85 or b == 95) else "steelblue" for b in edges[:-1]]
    axes[2].bar(range(len(cnt)), cnt, color=cols_b, edgecolor="white")
    axes[2].set_xticks(range(0, len(cnt), 2))
    axes[2].set_xticklabels([str(int(e)) for e in edges[:-1:2]], rotation=45, fontsize=8)
    axes[2].set_xlabel("on_time_pct (1% bucket)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("~1,500 customers per bucket everywhere\n(perfect Uniform)")

    plt.tight_layout()
    _save(fig, "01_a2_otp.png")
    plt.show()


def plot_a3_ks_uniform(df, features):
    """A3: KS test vs Uniform CDF. Returns list of result dicts."""
    fig, axes = plt.subplots(1, len(features), figsize=(15, 4))
    results = []
    for ax, col in zip(axes, features):
        x = df[col].values
        lo, hi = x.min(), x.max()
        stat, p = kstest(x, "uniform", args=(lo, hi - lo))
        results.append({"feature": col, "KS": stat, "p": p})
        xs = np.sort(x)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="#3498db", lw=1.5, label="Empirical CDF")
        ax.plot(xs, (xs - lo) / (hi - lo), color="#e74c3c", lw=1.5, linestyle="--", label="Uniform CDF")
        ax.set_title(f"{col}\nKS={stat:.4f}  p={p:.3f}")
        ax.set_xlabel(col)
        ax.set_ylabel("CDF")
        ax.legend(fontsize=8)
    fig.suptitle("A3 — KS Test: all continuous features are Uniform\n(p > 0.05 → cannot reject Uniform)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_a3_ks.png")
    plt.show()
    return results


def plot_a4_independence(df, features, otp_thresholds=(85, 95)):
    """A4: ANOVA + chi-square — features independent of n_increases."""
    def _risk(pct):
        return 0 if pct >= otp_thresholds[1] else (1 if pct >= otp_thresholds[0] else 2)

    rs = df["on_time_pct"].apply(_risk)
    groups_by = df.groupby("n_increases_2023")

    fig, axes = plt.subplots(1, len(features), figsize=(15, 4))
    for ax, col in zip(axes, features):
        groups = [g[col].values for _, g in groups_by]
        labels = sorted(df["n_increases_2023"].unique())
        F, p = f_oneway(*groups)
        ax.bar(labels, [g.mean() for g in groups], color="#3498db", edgecolor="white", alpha=0.8)
        ax.set_title(f"{col}\nANOVA p={p:.3f}")
        ax.set_xlabel("n_increases")
        ax.set_ylabel("Mean")
        ax.set_xticks(labels)
    fig.suptitle("A4 — Feature means flat across n_increases groups → random assignment", fontsize=11)
    plt.tight_layout()
    _save(fig, "01_a4_independence.png")
    plt.show()

    ct = pd.crosstab(df["n_increases_2023"], rs)
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    print(f"Chi-square (n_increases × risk_state): chi2={chi2:.3f}  dof={dof}  p={p_chi:.4f}  → INDEPENDENT")
    for col in features:
        groups = [g[col].values for _, g in groups_by]
        F, p = f_oneway(*groups)
        print(f"ANOVA {col:<28}: F={F:.3f}  p={p:.4f}  → INDEPENDENT")


def plot_a5_eligibility(df, eligibility_days=60):
    """A5: ineligible customers who received increases anyway."""
    eligible = df["days_since_last_loan"] >= eligibility_days
    received = df["n_increases_2023"] > 0
    inelig = df[~eligible]
    elig   = df[eligible]
    rate_i = received[~eligible].mean()
    rate_e = received[eligible].mean()
    inelig_recv = df[~eligible & received]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("A5 — 60-Day Eligibility Rule Not Enforced", fontsize=12, fontweight="bold")

    axes[0].bar(["Ineligible\n(days<60)", "Eligible\n(days≥60)"],
                [rate_i * 100, rate_e * 100],
                color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.4)
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel("Uptake Rate (%)")
    axes[0].set_title(f"{len(inelig_recv):,} ineligible customers received increases\n(uptake nearly identical to eligible)")
    for i, v in enumerate([rate_i * 100, rate_e * 100]):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ni = inelig_recv["n_increases_2023"].value_counts().sort_index()
    axes[1].bar(ni.index, ni.values, color="#e74c3c", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("n_increases")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Breakdown by n_increases\n(ineligible receivers)")
    axes[1].set_xticks(ni.index)

    plt.tight_layout()
    _save(fig, "01_a5_eligibility.png")
    plt.show()

    print(f"Ineligible (days<60): {len(inelig):,}  |  received: {len(inelig_recv):,}  ({rate_i*100:.1f}%)")
    print(f"Eligible   (days≥60): {len(elig):,}  |  uptake: {rate_e*100:.1f}%")
    print(f"Gap: {abs(rate_e - rate_i)*100:.1f} pp  → rule had no effect")


# ── 4. Profit formula comparison ─────────────────────────────────────────────

def plot_profit_rule_comparison(rule_table):
    """Bar chart comparing 3 candidate profit formulas."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rule_table))
    width = 0.25

    ax.bar(x - width, rule_table["observed_profit"], width=width, color="#8e44ad", label="Observed total_profit")
    ax.bar(x, rule_table["spec_profit"], width=width, color="#27ae60", label="Spec: 40×n")
    ax.bar(x + width, rule_table["activation_profit"], width=width, color="#f39c12", label="Activation: 40×(n−1)")

    ax.set_xticks(x)
    ax.set_xticklabels(rule_table["n_increases_2023"].astype(str))
    ax.set_xlabel("n_increases_2023")
    ax.set_ylabel("Profit per customer ($)")
    ax.set_title("A6 — Observed profit does not match task specification")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "01_a6_profit_rule.png")
    plt.show()


# ── 5. Eligible-zero vs Received comparison ──────────────────────────────────

def plot_received_vs_eligible_zero(df, eligibility_days=60):
    """Compare eligible-but-no-increases vs received groups."""
    eligible_zero = df[(df["days_since_last_loan"] >= eligibility_days) & (df["n_increases_2023"] == 0)].copy()
    received = df[df["n_increases_2023"] > 0].copy()

    summary = pd.DataFrame({
        "group": ["Eligible but no increases", "Received increases"],
        "customers": [len(eligible_zero), len(received)],
        "avg_initial_loan": [eligible_zero["initial_loan"].mean(), received["initial_loan"].mean()],
        "avg_days_since_last_loan": [eligible_zero["days_since_last_loan"].mean(), received["days_since_last_loan"].mean()],
        "avg_on_time_pct": [eligible_zero["on_time_pct"].mean(), received["on_time_pct"].mean()],
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#95a5a6", "#3498db"]
    metrics = [
        ("avg_initial_loan", "Average initial loan"),
        ("avg_days_since_last_loan", "Average days since last loan"),
        ("avg_on_time_pct", "Average on-time payment %"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(summary["group"], summary[col], color=colors, edgecolor="white")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, summary[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "01_group_compare.png")
    plt.show()
    return summary


# ── 6. Risk tier analysis ────────────────────────────────────────────────────

def plot_risk_vs_uptake(df):
    """Uptake and average n by risk tier."""
    risk_state = pd.cut(
        df["on_time_pct"],
        bins=[79.999, 85, 95, 100],
        labels=["Sub-Prime", "Near-Prime", "Prime"],
        include_lowest=True,
        right=False,
    )
    summary = (
        df.assign(risk_state=risk_state, received=df["n_increases_2023"] > 0)
          .groupby("risk_state", observed=False)
          .agg(customers=("customer_id", "count"),
               avg_otp=("on_time_pct", "mean"),
               pct_received=("received", "mean"),
               avg_n=("n_increases_2023", "mean"))
          .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    axes[0].bar(summary["risk_state"].astype(str), summary["pct_received"] * 100, color=colors, edgecolor="white")
    axes[0].set_title("Uptake by risk tier")
    axes[0].set_ylabel("Uptake rate (%)")
    axes[0].set_ylim(0, 70)

    axes[1].bar(summary["risk_state"].astype(str), summary["avg_n"], color=colors, edgecolor="white")
    axes[1].set_title("Average n_increases by risk tier")
    axes[1].set_ylabel("Average n_increases_2023")

    plt.tight_layout()
    _save(fig, "01_risk_uptake.png")
    plt.show()
    return summary


def plot_segments(df, risk_colors, risk_labels):
    """Post-feature-engineering: segment summary heatmap."""
    seg = df.groupby("risk_label").agg(
        n=("customer_id", "count"),
        pct_received=("received", "mean"),
        avg_n=("n_increases_2023", "mean"),
        avg_otp=("on_time_pct", "mean"),
        avg_profit=("incremental_profit", "mean"),
    ).reindex([risk_labels[s] for s in [0, 1, 2]])

    pivot = (
        df.groupby(["risk_state", "n_increases_2023"]).size()
          .unstack(fill_value=0)
          .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100)
          .rename(index=risk_labels)
          .round(1)
    )
    pivot.columns = [f"n={c}" for c in pivot.columns]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Segment Validation — Risk Tiers", fontsize=13, fontweight="bold")

    counts_r = [(df["risk_state"] == s).sum() for s in [0, 1, 2]]
    axes[0].pie(counts_r, labels=[risk_labels[s] for s in [0, 1, 2]],
                colors=[risk_colors[s] for s in [0, 1, 2]], autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Risk Tier Composition")

    vals = seg["pct_received"].values * 100
    bars = axes[1].bar(seg.index, vals, color=[risk_colors[s] for s in [0, 1, 2]], edgecolor="white")
    axes[1].set_ylim(0, 80)
    axes[1].set_ylabel("Uptake Rate (%)")
    axes[1].set_title("Uptake by Risk Tier\n(identical → random assignment)")
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1,
                     f"{v:.1f}%", ha="center", fontsize=10)

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", ax=axes[2],
                linewidths=0.5, cbar_kws={"label": "% of tier"})
    axes[2].set_title("n_increases Distribution by Risk Tier\n(flat rows = no quality signal)")
    axes[2].set_ylabel("")

    plt.tight_layout()
    _save(fig, "01_segments.png")
    plt.show()
    return seg, pivot