# Release procedure

1. Start from a clean checkout and verify no generated reports, caches, model weights, or media datasets are tracked.
2. Update `src/smarttab/__init__.py`, `pyproject.toml`, `CHANGELOG.md`, and `CITATION.cff` to the same version.
3. Install and run the core development matrix:

   ```bash
   python -m pip install -e ".[dev]"
   python -m ruff check src tests
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov.plugin --cov=smarttab
   python -m build
   python -m twine check dist/*
   ```

4. Run optional integration groups in clean environments:

   ```bash
   python -m pip install -e ".[all,dev]"
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov.plugin -m optional

   python -m pip install -e ".[multimodal-deep,dev]"
   # Run deployment-specific CPU/GPU deep-backend smoke tests.
   ```

5. Run the isolated CPU benchmark smoke before packaging:

   ```bash
   python benchmarks/run_cpu_benchmarks.py --quick --cases tabular text image --output benchmark_results/release-smoke
   ```

   Full reference runs use the five primary cases and a separate mixed case. Preserve machine-readable results with the release artifacts, but do not place generated model bundles or HTML reports inside the wheel.

6. Inspect wheel contents and verify:

   - `smarttab/reporting/templates/report.html.j2` exists;
   - every module under `smarttab/multimodal/` and `smarttab/datascience/` exists;
   - no `__pycache__`, model weights, media samples, reports, or `catboost_info` exist.

7. Install the wheel in a new virtual environment and run `tests/wheel_smoke.py`. The smoke test must cover tabular fit, raw text fit, report creation, and persistence round-trip.
8. On a codec-enabled environment, run one real file-path audio and video decode smoke test.
9. On a CUDA release-validation runner, test every declared deep backend with the exact PyTorch/CUDA versions documented for that runner. GPU validation is environmental and must not be inferred from CPU unit tests.
10. Create and push an annotated version tag.
11. Create a GitHub Release. The `Publish to PyPI` workflow uses PyPI Trusted Publishing; no long-lived API token is stored in GitHub.
12. Verify PyPI metadata, dependency extras, README rendering, wheel, sdist, and a clean `pip install smarttab`.
13. Never reuse or overwrite a published version number.
