# Architecture

The package is organized around small modules with explicit responsibilities.

- `config.py`: configuration dataclasses, YAML loading, and validation.
- `distributions.py`: random sample generation from configured distribution parameters.
- `statistics.py`: moment definitions and sample normalization.
- `selection.py`: finite-sample selection protocols.
- `transforms.py`: pairwise transformations.
- `simulation.py`: end-to-end simulation pipeline.
- `analytic.py`: finite-sample half-sum identities.
- `regression.py`: OLS regression and error metrics.
- `plotting.py`: correlation figures.
- `io.py`: output directories, JSON, Parquet, CSV, and workbook writing.
- `cli.py`: command-line interface.

