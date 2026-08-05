"""Mixed tabular + text + image classification without external files."""

import numpy as np
import pandas as pd

import smarttab

rows = 60
images = []
for index in range(rows):
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[..., index % 2] = 120 + index
    images.append(image)

frame = pd.DataFrame(
    {
        "age": np.arange(rows) + 20,
        "review": [
            ("excellent reliable product " if index % 2 else "damaged unreliable product ")
            + str(index)
            for index in range(rows)
        ],
        "photo": images,
        "label": [index % 2 for index in range(rows)],
    }
)

model = smarttab.fit(
    frame,
    target="label",
    modalities={"review": "text", "photo": "image"},
    feature_budget={"total": 128, "review": 80, "photo": 48},
    ensemble="auto",
    fusion="hybrid",
    optimize=False,
    report=False,
)

print(model.predict(frame.drop(columns="label").iloc[:3]))
print(model.feature_space)
