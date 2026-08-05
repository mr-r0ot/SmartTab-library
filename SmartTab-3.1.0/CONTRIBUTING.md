# Contributing

## Environment

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Required checks

```bash
python -m ruff check src tests
python -m pytest --cov=smarttab
python -m build
python -m twine check dist/*
```

Changes to training, splitting, cleaning, optimization, thresholding, or ensembling require tests that demonstrate absence of target leakage and evaluate behavior on an untouched outer holdout.

Changes to persistence require tests for malformed archives, path traversal, integrity mismatches, format-version mismatches, and trusted loading.

Public parameters must be validated and must have observable behavior. Do not add no-op compatibility flags.
