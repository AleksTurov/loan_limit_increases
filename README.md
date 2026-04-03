# Loan Limit Increase Optimization

End-to-end credit decision project for deciding who should receive a loan limit increase under response uncertainty, risk migration, and a binding capital constraint.

The project is built as a full modeling pipeline rather than a single scoring model: it starts with portfolio diagnostics, converts risk assumptions into a scenario-based Markov framework, estimates customer acceptance, optimizes allocation under capital limits, and then stress-tests the recommended strategy with Monte Carlo simulation.

The main business conclusion is intentionally decision-oriented: the deterministic optimum looks profitable, but it does not remain attractive once stochastic default risk is simulated under the current calibration.

## Results Snapshot

| Metric | Base | Stress | Benign |
| --- | ---: | ---: | ---: |
| Deterministic expected NPV | $322.5k | $285.6k | $341.4k |
| Monte Carlo mean NPV | -$737.0k | -$784.0k | -$706.7k |
| Monte Carlo VaR95 | -$781.1k | -$833.6k | -$749.3k |
| Monte Carlo CVaR95 | -$792.7k | -$845.6k | -$761.1k |
| Offered customers | 18,769 | 21,038 | 17,685 |

Bottom line: broad rollout is not recommended under the current PD assumptions. A pilot should start with lower-risk segments only.

Pipeline:

- EDA
- Markov-based risk approximation
- Demand forecasting
- Constrained optimization
- Monte Carlo simulation

## Project Objective

Determine which customers should receive a credit limit increase in 2023 in order to maximize expected profitability while controlling downside risk and respecting a capital limit.

## Repository Structure

- [notebooks/01_EDA.ipynb](/data/aturov/loan_limit_increases/notebooks/01_EDA.ipynb): data quality review, feature diagnostics, portfolio patterns
- [notebooks/02_Markov.ipynb](/data/aturov/loan_limit_increases/notebooks/02_Markov.ipynb): scenario-based risk transition model and PD calibration
- [notebooks/03_Demand.ipynb](/data/aturov/loan_limit_increases/notebooks/03_Demand.ipynb): acceptance propensity model
- [notebooks/04_Optimization.ipynb](/data/aturov/loan_limit_increases/notebooks/04_Optimization.ipynb): capital-constrained allocation strategy
- [notebooks/05_Simulation.ipynb](/data/aturov/loan_limit_increases/notebooks/05_Simulation.ipynb): Monte Carlo robustness analysis
- [src/](/data/aturov/loan_limit_increases/src): reusable modeling and plotting modules
- [data/](/data/aturov/loan_limit_increases/data): exported intermediate and final artifacts

## Final Recommendation

Use a risk-adjusted decision rule of the form `grant increase if incremental EV > 0`, implemented through the `lp_round` allocation strategy under the capital constraint.

However, the central business conclusion is that the deterministic optimum does not survive stochastic risk simulation under the current calibration.

As a result, a broad production rollout is not recommended under the current PD assumptions.

## Key Business Takeaways

- The optimized strategy offers increases to about `62.6%` of customers in the base scenario.
- The offer mix is intentionally tilted toward lower-risk tiers: roughly `33.4% Prime`, `55.2% Near-Prime`, and only `11.4% Sub-Prime`.
- Even with that conservative tilt, simulated losses dominate deterministic gains because the calibrated annual PDs are still high.
- A realistic deployment path is a pilot focused on Prime and the upper part of Near-Prime, with Sub-Prime excluded from the first wave.

## Important Modeling Limitation

The Markov risk model is a scenario-based approximation rather than a fully empirical migration model, because the source dataset does not provide a complete temporal panel of customer state transitions.

This means the transition matrices should be interpreted as structured business assumptions calibrated to plausible risk levels, not as directly estimated vintage roll-rate matrices.

## Suggested Production Next Steps

1. Recalibrate PD and transition assumptions on true longitudinal payment and default history.
2. Run a pilot rollout on low-risk segments only.
3. Track realized default rate, contribution profit, capital usage, and acceptance rate against model expectations.
4. Re-estimate strategy thresholds after observing live pilot outcomes.

## Outputs

Main exported artifacts:

- [data/optimization_summary.csv](/data/aturov/loan_limit_increases/data/optimization_summary.csv)
- [data/allocation_recommended.csv](/data/aturov/loan_limit_increases/data/allocation_recommended.csv)
- [data/simulation_summary.csv](/data/aturov/loan_limit_increases/data/simulation_summary.csv)
- [data/simulation_sensitivity.csv](/data/aturov/loan_limit_increases/data/simulation_sensitivity.csv)

## Environment

Python dependencies are listed in [requirements.txt](/data/aturov/loan_limit_increases/requirements.txt).