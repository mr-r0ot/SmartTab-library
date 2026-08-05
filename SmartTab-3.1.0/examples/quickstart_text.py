"""Minimal raw-text classification."""

import smarttab

texts = [
    "excellent build quality and reliable performance",
    "arrived broken and unusable",
    "works well and feels durable",
    "poor quality, failed immediately",
] * 12
labels = [1, 0, 1, 0] * 12

model = smarttab.fit_text(
    texts,
    labels,
    optimize=False,
    ensemble="none",
    report=False,
)

print(model.predict(["reliable and useful", "broken after one day"]))
print(model.feature_space)
