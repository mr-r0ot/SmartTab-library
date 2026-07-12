# SmartTab

**A smart decision engine for tabular machine learning.** Give it a table and a
target column — it figures out the rest.

```bash
pip install smarttab
```

```python
import smarttab

model = smarttab.fit(df, target="churned")
model.predict(new_customers)
```

That's a fully tuned, evaluated, explainable model. No feature engineering
checklist, no hyperparameter grid, no "which algorithm should I use" — SmartTab
looks at your data and your hardware and makes those calls for you, while
still letting you override any single one of them the moment you need to.

📘 **[documents.md](documents.md)** — the full guide: install → first model →
real projects → every parameter, from a beginner's first `fit()` call to
production-grade tuning.

⚙️ **[HowItWorks.md](HowItWorks.md)** — what actually happens inside `fit()`
and why it's built this way, for anyone who wants to look under the hood.

## What you get

- **One function, any kind of tabular problem.** Binary classification,
  multi-class, regression, multi-label, multi-output regression, and
  ranking are all detected automatically from the shape of your target —
  no `task_type=` flag to remember.
- **It cleans up after your data, not after you.** Missing values,
  duplicate rows, constant columns, ID-like columns, leaky columns,
  datetime and free-text fields — handled automatically, conservatively,
  and explained back to you in the report.
- **It knows what's underneath it.** SmartTab profiles your CPU, RAM, and
  GPU and tunes thread counts, batch sizes, and device selection so it
  runs well on a laptop and a workstation alike.
- **It doesn't always reach for the biggest hammer.** Ask for
  `ensemble="auto"` and it will tune a couple of strong candidates,
  compare them, and only pay for a full multi-model ensemble when the
  extra complexity actually earns its keep.
- **Confidence, not just a label.** For decisions where "maybe" matters —
  medical screening, fraud review, anything with a human in the loop —
  `multi_threshold_ensemble=True` gets you a confidence score alongside
  every prediction, so borderline cases can be routed for a second look
  instead of silently guessed at.
- **A report you'd actually want to read.** `model.report()` produces a
  single self-contained, interactive HTML file — metrics, feature
  importance, SHAP, confusion matrices or ROC curves, timing, memory
  usage — plus the same data as clean JSON.
- **Save it, ship it, load it back.** `model.save()` / `smarttab.load()`
  round-trip everything: the trained model(s), the cleaning pipeline, and
  every decision SmartTab made along the way.

## A quick look

```python
import smarttab

model = smarttab.fit(df, target="churned")   # analyzes, cleans, tunes, trains, evaluates

model.predict(new_df)                         # predictions on new data
model.evaluate(X_test, y_test)                # full metric set on a held-out set
model.report("my_report")                     # self-contained HTML + JSON + charts
model.save("model.smarttab")                  # one file, everything included

loaded = smarttab.load("model.smarttab")
```

`data` can be a `pandas.DataFrame` or a path to `.csv` / `.tsv` / `.xlsx` /
`.parquet` / `.json` / `.feather` / `.pickle`.

## Status

SmartTab is pre-1.0 and under active development. The core pipeline —
cleaning, hardware-aware training, hyperparameter search, ensembles, decision
thresholds, confidence scoring, explainability, and reporting — is fully
working end to end for every task type listed above. See the roadmap at the
end of [documents.md](documents.md) for what's intentionally not here yet.

## License

MIT — see [LICENSE](LICENSE).
