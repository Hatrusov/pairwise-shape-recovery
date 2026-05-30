# Reproducibility Report

This file records the validation commands for the current package state.

- Package version: `0.1.0`
- Tested Python version: `3.12.13`
- Tested platform: `Windows`
- Primary config: `configs/default.yaml`
- Extended config: `configs/extended_validation.yaml`
- Seed: `20261225`
- Test command: `python -m pytest -q`
- Test result: `48 passed`
- Validation command: `python -m pairwise_shape_recovery.cli validate --config configs/default.yaml`
- Maximum half-sum skew identity error in fast validation: `1.110e-15`
- Maximum half-sum kurtosis identity error in fast validation: `5.773e-15`
- Output archive checksum: to be recorded when packaging release artifacts.
- Git commit hash: recorded in `metadata.json` when available.

Recommended command sequence:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pairwise_shape_recovery.cli validate --config configs/default.yaml
python -m pairwise_shape_recovery.cli run --config examples/minimal_run.yaml --output outputs/example_minimal
python -m pairwise_shape_recovery.cli run --config configs/extended_validation.yaml
```

