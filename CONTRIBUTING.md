# Contributing

Contributions are welcome through issues and pull requests.

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check .
ruff format .
```

## Guidelines

- Keep random generation explicit through `numpy.random.default_rng(seed)`.
- Add tests for changes that affect numerical results or output structure.
- Keep public terminology neutral and scientific.
- Do not commit full simulation outputs; regenerate them from configuration files.

