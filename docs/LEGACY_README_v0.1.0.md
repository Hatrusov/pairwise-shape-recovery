# Pairwise Shape Recovery

## Purpose

This repository implements a fully reproducible simulation protocol for studying finite-sample moment relationships between original samples and pairwise-derived sequences.

The package generates small samples, applies pairwise transformations, computes finite-sample shape characteristics, estimates correlations and regressions, checks analytical half-sum identities, and writes results in machine-readable and review-friendly formats.

Module responsibilities are summarized in `docs/architecture.md`.

## Scientific Background

Let

```text
X = {x_1, x_2, ..., x_N}
```

be a finite sample with `N` in `[10, 20, 30, 40, 50]`. The study focuses on how skewness and kurtosis of `X` are related to the corresponding moments of sequences generated from all unordered pairs of observations.

The default protocol treats every generated realization as a finite sample. For distributions such as Cauchy, population moments do not exist, so only finite-sample statistics are computed and interpreted.

## Pairwise Transformations

For each normalized sample, the package computes:

- `ordered_pairwise_differences(x)`: sort `x` descending and compute `x_(i) - x_(j)` for all `i < j`.
- `direct_pairwise_differences(x)`: compute `x_i - x_j` for all `i < j` without sorting.
- `pairwise_half_sums(x)`: compute `(x_i + x_j) / 2` for all `i < j`.

The direct signed pairwise differences are order-dependent. Changing the order of the input sample can change this sequence.

Each transformed sequence has length:

```text
m = N * (N - 1) / 2
```

## Moment Definitions

The default protocol uses:

- population standard deviation;
- adjusted Fisher-Pearson skewness;
- kurtosis as the standardized fourth central moment without subtracting 3.

For an array `x` of length `n`:

```text
mean = mean(x)
m2 = mean((x - mean)^2)
sd = sqrt(m2)
m3 = mean((x - mean)^3)
m4 = mean((x - mean)^4)
g1 = m3 / m2^(3/2)
G1 = sqrt(n * (n - 1)) / (n - 2) * g1
beta2 = m4 / m2^2
```

The half-sum identities are valid only under these default definitions.

## Primary Simulation Protocol

The primary protocol uses the distribution families:

- `uniform`
- `normal`
- `cauchy`
- `student_t5`

For each `N` and each selection regime, the default run selects 100 samples per distribution, producing an aggregate `400 x N` sample table.

Selection regimes:

- `low_skew_selected`: finite samples selected by a threshold on absolute adjusted skewness.
- `moderate_shape_selected`: finite samples selected from a configured decile of the score `abs(skewness) * kurtosis`.

Distribution parameters are explicit in YAML under `distribution_parameters`; they are not embedded as experiment-specific constants in the generator code.

## Extended Validation Protocol

The extended validation protocol repeats the same selection and analysis workflow on additional population families:

- `lognormal`
- `gamma`
- `weibull`
- `skewnormal`

Extended outputs are written under:

```text
outputs/primary_protocol/extended_protocol/
```

When reading extended protocol results, use this directory rather than the primary protocol output root.

## Analytical Half-Sum Identities

For normalized input with population variance and kurtosis `beta2` without subtracting 3, the inverse half-sum kurtosis identity for `N > 8` is:

```text
beta2_x = -3N / (N - 8)
          + [2(N - 2)^2 / ((N - 1)(N - 8))] * beta2_s
```

For adjusted Fisher-Pearson skewness:

```text
m = N(N - 1) / 2
C = N * sqrt((N - 1)(N - 2)) * (m - 2)
    / (m * sqrt(m - 1) * (N - 4))

G1_x = C * G1_half_sum
```

The pipeline validates these identities and records numerical error summaries. If the moment definitions are changed, analytical half-sum checks fail early with a clear `ValueError`.

## Installation

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

The installation step is important because the pipeline writes Parquet files. `pyarrow>=14` is a required dependency; without it, Parquet output raises a clear dependency error.

## Quick Start

Run a small example:

```bash
python -m pairwise_shape_recovery.cli run --config examples/minimal_run.yaml --output outputs/example_minimal
```

Run the primary protocol:

```bash
python -m pairwise_shape_recovery.cli run --config configs/default.yaml
```

Run the primary protocol plus extended validation:

```bash
python -m pairwise_shape_recovery.cli run --config configs/extended_validation.yaml
```

Run fast analytical validation:

```bash
python -m pairwise_shape_recovery.cli validate --config configs/default.yaml
```

## Output Structure

Default output root:

```text
outputs/primary_protocol/
  config_used.yaml
  metadata.json
  samples/
  result_matrices/
    by_distribution/
  correlations/
    by_distribution/
  regressions/
    by_distribution/
  analytic/
  figures/
  workbook/
  logs/
```

Aggregate result matrices are written with exactly 8 statistical columns:

- `original_skewness`
- `original_kurtosis`
- `ordered_difference_skewness`
- `ordered_difference_kurtosis`
- `direct_difference_skewness`
- `direct_difference_kurtosis`
- `half_sum_skewness`
- `half_sum_kurtosis`

For traceability, audit matrices are also saved:

```text
result_matrices/{regime}_N{N}_result_matrix_with_metadata.csv
```

The audit version includes `global_sample_id`, `distribution`, `population_family`, `population_symmetry`, `selection_regime`, and `selection_algorithm` before the statistical columns.

Per-distribution outputs are required for robustness analysis:

```text
result_matrices/by_distribution/{regime}_{distribution}_N{N}_result_matrix.csv
correlations/by_distribution/{regime}_{distribution}_N{N}_correlation.csv
correlations/by_distribution/correlation_summary_by_distribution.csv
regressions/by_distribution/direct_kurtosis_regressions_by_distribution.csv
regressions/by_distribution/inverse_half_sum_regressions_by_distribution.csv
analytic/half_sum_analytic_checks_by_distribution.csv
```

The summary workbook includes:

- `config`
- `direct_kurtosis_regressions`
- `inverse_half_sum_regressions`
- `correlation_summary`
- `analytic_half_sum_checks`
- `selection_metadata`
- `direct_by_dist`
- `inverse_by_dist`
- `analytic_by_dist`
- `corr_by_dist`

## Reproducibility

All random draws use `numpy.random.default_rng(seed)`. The seed and full configuration are saved with the output metadata.

Metadata includes:

- UTC date and time;
- Python version;
- package version;
- seed;
- distribution parameters;
- configuration hash;
- Git commit hash when available;
- selection counts;
- generation rejection counts;
- runtime;
- computational warnings.

The main protocol and extended validation protocol write separate metadata files. The main metadata contains only a compact status block for extended validation.

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Run tests and save a report:

```bash
python -m pairwise_shape_recovery.cli test --report outputs/logs/test_report.txt
```

Continuous integration runs smoke-sized simulations only; full simulation outputs are regenerated from configuration files and are not required for CI.

## Limitations

The package studies finite-sample moment behavior under specified simulation and selection rules. It does not estimate population moments for distributions where they do not exist.

The score used for `moderate_shape_selected` combines sample skewness and kurtosis; it should be interpreted as a finite-sample selection rule, not as a population classification.

Polynomial trend fitting across five `N` values, if added for comparison, should be treated as interpolation rather than as an independent statistical model.

## Citation

Please cite this repository using the metadata in `CITATION.cff`. If a paper or archive DOI is available, cite that record as the preferred scholarly source.

## License

This project is released under the MIT License. See `LICENSE`.
