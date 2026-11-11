# Pairwise Shape Recovery v0.2.0

Computational supplement for the major revision of *Finite-Sample Moment Identities for Pairwise Half-Sum Sequences: A Computational Study of Skewness and Kurtosis*, JSCS manuscript 260396436. This is a prepared release, not a claim that a GitHub release has already been published.

## Reproduce the revision

Python 3.12 recommended. Run from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r revision/requirements.txt
python revision/run_experiments.py
python revision/symbolic_check.py
```

NumPy suffices for the experiments; SymPy is used for symbolic verification. The default run generates 5,000 samples per cell (75,000 total). It overwrites `revision/results/`. A short smoke run does not reproduce the manuscript:

```bash
python revision/run_experiments.py --replications 20 --output smoke_results
```

To run all legacy and revision tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Results

| File in revision/results | Correspondence |
| --- | --- |
| table6_counterexamples.csv | Six exceptional sizes; exact equality of pair-sum multisets |
| exact_witnesses.json | Integer samples and pair-sum multiplicities |
| table7_population.csv | Bias, MSE, Monte Carlo SE and arithmetic errors for 15 cells |
| replication_statistics.csv | Direct and recovered statistics for all 75,000 samples |
| covariance_recovery.csv | 27 cells × 100 samples comparing pair-only C with direct score products |
| environment.json | Runtime, generator, seeds and NumPy version |
| SHA256SUMS.txt | Generated result checksums |

Population simulation uses one PCG64 generator, seed 260396436. Family order: Normal, Uniform[-1,1], Exponential(mean 1), Gamma(shape 3, scale 1), Student t(10). Size order: 10, 20, 30 within each family. All source samples in a cell are generated before calculation. Direct statistics use the source data; explicit half-sums use centered and scaled source data. Results concern adjusted Fisher–Pearson skewness and Pearson kurtosis (without subtraction of three).

MCSE(bias) = sample SD(error)/sqrt(R); MCSE(MSE) = sample SD(squared error)/sqrt(R), using divisor R−1. These quantify simulation precision, not uncertainty of an individual observation. No bootstrap is performed. Algebraically identical estimators have identical statistical risks. All printed Table 7 bias/MSE/MCSE values reproduce at three decimals. Last-bit arithmetic errors depend on evaluation order and platform; the manuscript and CSV report the updated implementation's maxima.

## Pair-only uncertainty summaries

`src/pairwise_shape_recovery/inference.py` accepts only pair values and n. It recovers moments through order eight and evaluates influence-function polynomial products. It returns C and C/n: C is the empirical influence covariance, and C/n estimates first-order sampling covariance. Neither is exact finite-sample sampling covariance. The classification uses unadjusted skewness; for adjusted skewness multiply the first row and column by sqrt(n(n−1))/(n−2).

The routine refuses the universal recursion at n = 4, 8, 16, 32, 64, 128. This does not prove ambiguity of a particular input. It rejects wrong lengths, nonfinite and constant data. Correct length alone does not certify that arbitrary inputs form a complete pair multiset: validity is an assumption. No pair labels or source reconstruction are used.

C is positive semidefinite in exact arithmetic. Rounding can produce small negative eigenvalues. No projection silently alters C. Cancellation ratios and minimum eigenvalues are diagnostics, not certified error bounds. Exact integer/rational witnesses verify counterexamples; numerical tests verify implementations. Coverage of confidence intervals and estimation accuracy of C are not assessed. The method does not handle measurement noise or missing pairs.

## Legacy and engineering scope

The original CLI, configs and tests are retained; see `docs/LEGACY_README_v0.1.0.md`. Selected-regime regressions are legacy features, not evidence for the revised statistical conclusions. The new experiment does not call them. Legacy shape-inversion helpers now accept the corrected sizes: skewness n≥3 except 4; kurtosis n≥3 except 8.

The supplied archive contained no raw diesel-engine data or historical output tables. Those are not fabricated here. Obtain engineering data from the source cited in the manuscript. The historical engine pipeline and manuscript Tables 4–5 were not reproduced by this revision run. This package reproduces the new Tables 6–7 and supplementary covariance checks, not every legacy result.

The recurrence, Prouhet construction and influence functions are classical. Computational verification is not proof of priority. The manuscript describes the specialized joint identification result.

## Publishing

See `UPLOAD_RU.md` and `RELEASE_NOTES.md`. The existing license and author metadata are retained. No remote repository, tag or DOI is created by this archive.
