# v0.2.0 — JSCS major revision computational supplement

- Add population-target experiment: 15 cells, 75,000 samples.
- Add exact counterexamples at n = 4, 8, 16, 32, 64, 128.
- Add pair-only influence covariance recovery and 2,700 comparisons.
- Include replication statistics, CSV summaries, witnesses and environment.
- Correct admissible sizes in legacy inversion helpers.
- Distinguish C from C/n and empirical covariance from exact sampling covariance.
- Retain legacy implementation and document historical results not rerun.

The routines do not implement bootstrap intervals, recover individual observations or improve the MSE of algebraically identical estimators. This release supports reproducibility, not a claim of novelty for classical identities.
