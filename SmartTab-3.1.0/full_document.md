# SmartTab Documentation

## Chapter 1 — Introduction to SmartTab

### 1.1 What is SmartTab?

* Overview
* Problem SmartTab solves
* Comparison with traditional ML workflows

### 1.2 Core Philosophy

* Zero/manual ML engineering
* Data-first modeling
* Hardware-aware AI
* Explainable automation

### 1.3 SmartTab Pipeline Overview

```
Raw Data
   |
   v
Input Validation
   |
   v
Dataset Profiling
   |
   v
Data Quality Audit
   |
   v
Automatic Cleaning
   |
   v
Feature Engineering
   |
   v
Hardware Profiling
   |
   v
Model Selection
   |
   v
Optimization
   |
   v
Ensemble Decision
   |
   v
Calibration + Uncertainty
   |
   v
Evaluation
   |
   v
Explainability
   |
   v
Report
```

---

# Chapter 2 — Installation

### 2.1 Requirements

### 2.2 Installation Methods

### 2.3 Optional Dependencies

### 2.4 Hardware Support

* CPU
* GPU
* RAM limits
* Thread control

---

# Chapter 3 — Quick Start

### 3.1 Your First SmartTab Model

```python
from smarttab import fit

model = fit(
    data=df,
    target="target"
)
```

### 3.2 Prediction

### 3.3 Evaluation

### 3.4 Model Information

### 3.5 Saving and Loading

---

# Chapter 4 — Understanding `fit()`

Complete reference for:

```python
smarttab.fit()
```

## 4.1 Function Signature

## 4.2 Input Data Formats

Supported:

* pandas DataFrame
* CSV path
* Parquet
* raw text
* images
* audio
* videos
* numpy arrays

## 4.3 Parameters

Detailed tables:

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |

Including:

* target
* group_id
* y
* modality
* modalities
* time_limit
* optimize
* ensemble
* model
* metrics
* objective
* explain
* report
* device
* random_state
* etc.

---

# Chapter 5 — Data Input System

## 5.1 Tabular Data

## 5.2 External Target (`y=`)

Example:

```python
fit(
    X,
    y=y
)
```

## 5.3 Multimodal Data

Supported:

* text
* image
* audio
* video

---

# Chapter 6 — Dataset Analysis Engine

How SmartTab understands your dataset.

## 6.1 Automatic Task Detection

Supported tasks:

* Binary Classification
* Multiclass Classification
* Multilabel Classification
* Regression
* Multi-output Regression
* Ranking

## 6.2 Dataset Profile

Explains:

* columns
* datatypes
* cardinality
* missing values
* feature types
* modality detection

## 6.3 Example Profile Output

---

# Chapter 7 — Data Quality System

## 7.1 Automatic Data Audit

Using:

```python
audit()
```

## 7.2 Quality Checks

Includes:

* Missing values
* Duplicate rows
* Conflicting labels
* Invalid media
* Feature problems

## 7.3 Quality Policies

* permissive
* default
* strict

---

# Chapter 8 — Automatic Cleaning Pipeline

## 8.1 Cleaning Architecture

```
Raw Dataset

 ↓

Missing Value Handling

 ↓

Categorical Encoding

 ↓

Scaling

 ↓

Feature Selection

 ↓

Outlier Handling

 ↓

Final Feature Matrix
```

## 8.2 Missing Value Strategies

## 8.3 Categorical Processing

## 8.4 Leakage Protection

## 8.5 Schema Validation

---

# Chapter 9 — Data Splitting Engine

## 9.1 Automatic Split Strategy

Strategies:

* random
* group
* stratified_group
* temporal

## 9.2 Preventing Data Leakage

Examples:

* user IDs
* medical patients
* time series

---

# Chapter 10 — Hardware Intelligence

## 10.1 Hardware Profiling

Explains:

```python
profile_hardware()
```

Detects:

* CPU
* RAM
* GPU
* VRAM

## 10.2 Resource Planner

How SmartTab decides:

* model size
* thread count
* GPU usage

---

# Chapter 11 — Model Selection Engine

## 11.1 Supported Algorithms

* CatBoost
* LightGBM
* XGBoost
* Other boosting models

## 11.2 Automatic Selection Logic

Examples:

Small dataset:

```
CatBoost
```

Large dataset:

```
LightGBM
```

High cardinality:

```
CatBoost
```

---

# Chapter 12 — Optimization Engine

## 12.1 Hyperparameter Optimization

## 12.2 Search Spaces

## 12.3 Time-aware Optimization

## 12.4 Optimization Algorithms

---

# Chapter 13 — Automatic Ensemble System

## 13.1 Ensemble Modes

```python
ensemble="auto"
```

## 13.2 Voting Ensemble

## 13.3 Stacking Ensemble

## 13.4 Meta Models

## 13.5 Diversity Optimization

Architecture:

```
Model A
Model B
Model C

      |
      v

Meta Learner

      |
      v

Final Prediction
```

---

# Chapter 14 — Threshold Optimization

## 14.1 Classification Thresholds

## 14.2 Multilabel Thresholds

## 14.3 Objective Optimization

---

# Chapter 15 — Probability Calibration & Uncertainty

## 15.1 Probability Calibration

Methods:

* sigmoid
* isotonic

## 15.2 Conformal Prediction

## 15.3 Out-of-Distribution Detection

---

# Chapter 16 — Evaluation System

## 16.1 Automatic Metrics

Classification:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* LogLoss

Regression:

* RMSE
* MAE
* R²

Ranking:

* NDCG

---

# Chapter 17 — Explainability

## 17.1 Feature Importance

## 17.2 SHAP Explainability

## 17.3 Understanding Model Decisions

---

# Chapter 18 — SmartTab Reports

## 18.1 Automatic Report Generation

## 18.2 Report Structure

Contains:

* Dataset summary
* Cleaning report
* Hardware report
* Model information
* Metrics
* Explainability
* Charts

---

# Chapter 19 — Multimodal Learning

## 19.1 Text Models

## 19.2 Image Models

## 19.3 Audio Models

## 19.4 Video Models

## 19.5 Modality Fusion

---

# Chapter 20 — Model Object API

Complete reference:

```python
SmartTabModel
```

Methods:

* predict()
* predict_proba()
* evaluate()
* report()
* save()

---

# Chapter 21 — Saving and Loading

## 21.1 save()

## 21.2 load()

## 21.3 Trusted Loading

---

# Chapter 22 — Advanced Configuration

## 22.1 FitConfig

## 22.2 DataScienceConfig

## 22.3 Resource Control

## 22.4 Production Settings

---

# Chapter 23 — Production Usage

## 23.1 API Deployment

## 23.2 Batch Prediction

## 23.3 Monitoring Drift

## 23.4 Retraining

---

# Chapter 24 — Best Practices

* Dataset preparation
* Choosing time limits
* Handling imbalance
* Avoiding leakage
* Production recommendations

---

# Chapter 25 — Troubleshooting

Common errors:

* invalid targets
* insufficient samples
* GPU issues
* memory problems
* split failures

---

# Appendix A — Full API Reference

Detailed:

* fit()
* audit()
* fit_text()
* fit_images()
* fit_audio()
* fit_videos()
* fit_folder()
* load()

---

# Appendix B — SmartTab Architecture Internals

Developer-oriented:

```
smarttab/
│
├── analysis/
├── cleaning/
├── hardware/
├── optimization/
├── training/
├── explainability/
├── persistence/
└── model.py
```

---

# Appendix C — Examples Collection

Real examples:

* Customer churn
* Fraud detection
* House price prediction
* Image classification
* Text classification
* Time-series ranking

---

```md
# Chapter 1 — Introduction to SmartTab

## 1.1 What is SmartTab?

SmartTab is an automated machine learning (AutoML) library designed to transform raw datasets into production-ready machine learning models with minimal manual intervention.

Unlike traditional ML workflows where developers must manually perform:

- Data cleaning
- Feature engineering
- Algorithm selection
- Hyperparameter tuning
- Ensemble design
- Hardware optimization
- Model explanation
- Evaluation

SmartTab provides an end-to-end intelligent pipeline that automatically analyzes the dataset, selects appropriate strategies, trains optimized models, and generates explainable reports.

The main goal of SmartTab is:

> Convert raw structured or multimodal data into reliable, optimized, and explainable machine learning systems automatically.

---

# 1.2 The Problem SmartTab Solves

A typical machine learning workflow requires many independent decisions:

```

Raw Dataset

```
  |
  v
```

"What type of problem is this?"

```
  |
  v
```

"Which features are useful?"

```
  |
  v
```

"How should missing values be handled?"

```
  |
  v
```

"Which algorithm should I use?"

```
  |
  v
```

"Which parameters are optimal?"

```
  |
  v
```

"Should I use an ensemble?"

```
  |
  v
```

"How reliable are predictions?"

```
  |
  v
```

"How can I explain the model?"

```

These decisions usually require:

- Experienced data scientists
- Multiple libraries
- Manual experimentation
- Long development cycles


SmartTab automates these decisions through an intelligent pipeline.

---

# 1.3 SmartTab Core Philosophy

SmartTab is built around five main principles:

| Principle | Description |
|---|---|
| Data First | Understand the dataset before selecting models |
| Automation | Remove unnecessary manual ML decisions |
| Hardware Awareness | Adapt computation to available resources |
| Reliability | Detect data problems before training |
| Explainability | Provide understandable model decisions |

---

# 1.4 SmartTab vs Traditional Machine Learning Workflow

## Traditional Workflow

```

Dataset

↓

Manual Exploration

↓

Manual Cleaning

↓

Manual Feature Engineering

↓

Try Algorithms

↓

Tune Parameters

↓

Compare Results

↓

Build Report

```

Problems:

- Time consuming
- Requires expertise
- Easy to introduce mistakes
- Hard to reproduce


---

## SmartTab Workflow

```

Dataset

↓

Automatic Profiling

↓

Automatic Quality Audit

↓

Automatic Cleaning

↓

Hardware Analysis

↓

Algorithm Selection

↓

Optimization

↓

Ensemble Decision

↓

Calibration

↓

Evaluation

↓

Explainability

↓

Report

```

Advantages:

| Feature | Traditional ML | SmartTab |
|-|-|-|
| Data analysis | Manual | Automatic |
| Cleaning | Manual | Automatic |
| Model selection | Manual | Automatic |
| Optimization | Manual | Automatic |
| Ensemble creation | Manual | Automatic |
| Explainability | Optional | Built-in |
| Hardware adaptation | Manual | Automatic |
| Reporting | Manual | Automatic |

---

# 1.5 SmartTab High-Level Architecture

The complete SmartTab pipeline consists of several intelligent subsystems.

```

```
                     SmartTab

                        |
                        v

          +---------------------------+
          |     Input Processing      |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Dataset Analysis Engine    |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Data Quality System        |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Cleaning Pipeline          |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Feature Engineering        |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Hardware Profiler          |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Model Selection Engine     |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Optimization Engine        |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Ensemble System            |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Calibration & Uncertainty |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Evaluation                 |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Explainability             |
          +---------------------------+

                        |

                        v

          +---------------------------+
          | Report Generation          |
          +---------------------------+
```

````

---

# 1.6 Main Features

## Automatic Dataset Understanding

SmartTab automatically discovers:

- Dataset shape
- Column types
- Feature distributions
- Missing values
- Cardinality
- Possible task type
- Data quality issues


Example:

Input:

```python
import pandas as pd

df = pd.read_csv("customers.csv")
````

SmartTab automatically determines:

```
Task:
Binary Classification

Target:
churn

Features:
25 columns

Numeric:
18

Categorical:
7

Missing Values:
3 columns

Recommended Models:
CatBoost + LightGBM
```

---

# 1.7 Automatic Data Cleaning

SmartTab includes an intelligent cleaning pipeline.

Supported operations:

| Operation              | Description                                     |
| ---------------------- | ----------------------------------------------- |
| Missing value handling | Automatically fills or processes missing values |
| Encoding               | Converts categorical features                   |
| Scaling                | Normalizes numerical features when needed       |
| Feature selection      | Removes unnecessary features                    |
| Outlier handling       | Detects abnormal samples                        |
| Leakage protection     | Prevents invalid feature usage                  |

---

# 1.8 Hardware-Aware Machine Learning

SmartTab does not assume unlimited resources.

It profiles:

* CPU cores
* RAM availability
* GPU availability
* GPU memory

Then adjusts:

* Algorithm choice
* Number of trees
* Parallel threads
* Optimization budget

Example:

Low-resource machine:

```
CPU only
8GB RAM

↓

Light optimization
Smaller models
Limited search
```

High-resource machine:

```
GPU
64GB RAM

↓

More optimization
Larger search space
Ensemble exploration
```

---

# 1.9 Intelligent Model Selection

SmartTab supports automatic selection of boosting-based models.

Supported families include:

| Algorithm | Typical Usage              |
| --------- | -------------------------- |
| CatBoost  | General boosting problems  |
| LightGBM  | Large datasets             |

The selection system considers:

* Dataset size
* Feature types
* Task type
* Hardware
* Time budget

---

# 1.10 Automatic Ensemble Intelligence

SmartTab can automatically decide whether combining multiple models improves performance.

Supported strategies:

## Voting Ensemble

Multiple models vote together.

```
Model A ----\
             \
Model B -------> Final Prediction
             /
Model C ----/
```

---

## Stacking Ensemble

Models become feature generators for a meta-model.

```
             Model A
                |
             Model B
                |
             Model C

                |
                v

          Meta Learner

                |
                v

        Final Prediction
```

---

# 1.11 Explainability by Design

SmartTab does not only produce predictions.

It also explains:

* Which features matter
* Why a prediction happened
* How confident the model is

Available explanations:

* Feature importance
* SHAP values
* Probability calibration
* Uncertainty estimation

Example:

Prediction:

```
Customer will churn

Probability:
87%
```

Explanation:

```
Top factors:

1. Contract type       +35%
2. Monthly charges     +22%
3. Support tickets     +18%
```

---

# 1.12 Supported Machine Learning Tasks

SmartTab supports:

| Task                      | Supported |
| ------------------------- | --------- |
| Binary Classification     | Yes       |
| Multiclass Classification | Yes       |
| Multilabel Classification | Yes       |
| Regression                | Yes       |
| Multi-output Regression   | Yes       |
| Ranking Problems          | Yes       |

---

# 1.13 Supported Data Types

SmartTab can work with:

## Tabular Data

Examples:

* CSV
* Parquet
* pandas DataFrame

## Text Data

Examples:

* Documents
* Reviews
* Messages

## Image Data

Examples:

* Image classification
* Visual features

## Audio Data

Examples:

* Speech
* Sound classification

## Video Data

Examples:

* Frame-based analysis

---

# 1.14 SmartTab Design Goal

The final objective of SmartTab is not only:

> "Train a model."

It is:

> "Automatically build a complete, optimized, explainable machine learning system from raw data."

A SmartTab model contains:

```
Trained Model

+

Cleaning Pipeline

+

Feature Information

+

Optimization Results

+

Hardware Profile

+

Evaluation Metrics

+

Explainability Data

+

Uncertainty Models

+

Drift Monitoring Information
```

This allows the model to be reused reliably in real-world applications.

---

# End of Chapter 1

Next:
**Chapter 2 — Installation and Environment Setup**

```
```

 id="84291"
# Chapter 2 — Installation and Environment Setup

## 2.1 Requirements

SmartTab is designed to work in standard Python machine learning environments.

Minimum requirements:

| Component | Requirement |
|---|---|
| Python | >= 3.10 |
| Operating System | Windows / Linux / macOS |
| RAM | 4GB+ recommended |
| CPU | Any modern x64 processor |
| GPU | Optional |

Recommended production environment:

| Component | Recommendation |
|-|-|
| Python | 3.11+ |
| RAM | 16GB+ |
| CPU | 8+ cores |
| GPU | NVIDIA CUDA GPU (optional) |

---

# 2.2 Installing SmartTab

## Installation from PyPI

The recommended installation method:

```bash
pip install smarttab
````

Verify installation:

```bash
python -c "import smarttab; print(smarttab.__version__)"
```

Expected output:

```
0.x.x
```

---

# 2.3 Development Installation

If you want the latest development version:

```bash
git clone https://github.com/your-org/smarttab.git

cd smarttab

pip install -e .
```

The `-e` flag installs SmartTab in editable mode, allowing source code changes without reinstalling.

---

# 2.4 Recommended Environment Setup

Using virtual environments is recommended.

## Create Environment

```bash
python -m venv smarttab-env
```

Activate:

### Windows

```bash
smarttab-env\Scripts\activate
```

### Linux / macOS

```bash
source smarttab-env/bin/activate
```

Install:

```bash
pip install smarttab
```

---

# 2.5 Installing Optional Dependencies

SmartTab automatically enables additional features when optional packages are installed.

---

## GPU Support

For NVIDIA GPU acceleration:

```bash
pip install torch
```

Verify CUDA:

```python
import torch

print(torch.cuda.is_available())
```

Example output:

```
True
```

---

## SHAP Explainability

For advanced explainability:

```bash
pip install shap
```

Enable:

```python
model = fit(
    data=df,
    target="target",
    explain=True
)
```

SmartTab will generate SHAP explanations.

---

## Visualization Support

For reports and charts:

```bash
pip install matplotlib plotly
```

---

# 2.6 Basic Installation Test

Create:

```
test_smarttab.py
```

Add:

```python
import smarttab

print("SmartTab installed successfully")
```

Run:

```bash
python test_smarttab.py
```

Output:

```
SmartTab installed successfully
```

---

# 2.7 First Working Example

## Dataset Example

Assume we have:

```
customers.csv
```

Content:

| age | income | contract | churn |
| --- | ------ | -------- | ----- |
| 25  | 40000  | monthly  | 1     |
| 45  | 90000  | yearly   | 0     |
| 33  | 55000  | monthly  | 1     |

---

## Load Dataset

```python
import pandas as pd

df = pd.read_csv("customers.csv")

print(df.head())
```

Output:

```
   age  income contract churn

0   25   40000  monthly    1
1   45   90000   yearly    0
2   33   55000  monthly    1
```

---

# 2.8 Training Your First SmartTab Model

```python
from smarttab import fit

model = fit(
    data=df,
    target="churn"
)
```

That single command triggers:

```
Dataset Loading

        ↓

Target Detection

        ↓

Task Detection

        ↓

Data Quality Analysis

        ↓

Cleaning Pipeline

        ↓

Feature Processing

        ↓

Hardware Profiling

        ↓

Model Selection

        ↓

Training

        ↓

Evaluation

        ↓

Explainability

        ↓

Report
```

---

# 2.9 Understanding the Returned Model

`fit()` returns:

```python
SmartTabModel
```

Example:

```python
print(type(model))
```

Output:

```
<class 'smarttab.model.SmartTabModel'>
```

---

Inspect model information:

```python
print(model)
```

Example:

```
SmartTabModel

Task:
binary_classification

Model:
CatBoost

Metric:
roc_auc

Score:
0.91

Features:
24

Training Time:
18.4 seconds
```

---

# 2.10 Accessing Model Metadata

## Selected Algorithm

```python
print(model.model_name)
```

Example:

```
catboost
```

---

## Task Type

```python
print(model.task_type)
```

Example:

```
TaskType.BINARY
```

---

## Best Parameters

```python
print(model.best_params)
```

Example:

```python
{
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3
}
```

---

## Evaluation Metrics

```python
print(model.metrics)
```

Example:

```python
{
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "roc_auc": 0.94
}
```

---

# 2.11 Making Predictions

New data:

```python
new_customers = pd.DataFrame(
    {
        "age": [40],
        "income": [75000],
        "contract": ["yearly"]
    }
)
```

Prediction:

```python
prediction = model.predict(new_customers)

print(prediction)
```

Output:

```
[0]
```

---

# 2.12 Getting Prediction Probabilities

For classification:

```python
probability = model.predict_proba(
    new_customers
)

print(probability)
```

Example:

```
[
 [0.91, 0.09]
]
```

Meaning:

```
Class 0 probability: 91%

Class 1 probability: 9%
```

---

# 2.13 Automatic Report Generation

Enable reports:

```python
model = fit(
    data=df,
    target="churn",
    report=True
)
```

SmartTab creates:

```
smarttab_reports/

 └── catboost_20260805_120000/

      ├── index.html
      ├── metrics.html
      ├── dataset.html
      ├── explainability.html
      └── charts/
```

Open:

```
smarttab_reports/.../index.html
```

in your browser.

---

# 2.14 Controlling Training Time

SmartTab supports time budgets.

Example:

```python
model = fit(
    data=df,
    target="churn",
    time_limit=60
)
```

Meaning:

```
Maximum training time:
60 seconds
```

SmartTab automatically reduces:

* Optimization trials
* Ensemble search
* Expensive explanations

when the deadline approaches.

---

# 2.15 Reproducible Training

For reproducible results:

```python
model = fit(
    data=df,
    target="churn",
    random_state=42
)
```

The same random seed controls:

* Dataset splitting
* Optimization
* Ensemble selection
* Training randomness

---

# 2.16 CPU Only Training

Force CPU:

```python
model = fit(
    data=df,
    target="churn",
    device="cpu"
)
```

---

# 2.17 GPU Training

Allow GPU:

```python
model = fit(
    data=df,
    target="churn",
    device="gpu"
)
```

SmartTab will:

1. Detect GPU
2. Verify compatibility
3. Enable acceleration if possible

---

# 2.18 Complete Beginner Example

A complete script:

```python
import pandas as pd
from smarttab import fit

# Load data
df = pd.read_csv(
    "customers.csv"
)

# Train model
model = fit(
    data=df,
    target="churn",
    report=True,
    explain=True,
    time_limit=300,
    random_state=42
)

# Show result
print(model.metrics)

# Predict new data
sample = df.drop(
    columns=["churn"]
).head(5)

prediction = model.predict(sample)

print(prediction)
```

Output:

```
{
 'accuracy': 0.92,
 'roc_auc': 0.96
}

[0 1 0 0 1]
```

---





# Chapter 3 — Quick Start and Core Workflow

## 3.1 SmartTab Workflow Overview

SmartTab follows an automated machine learning workflow:

```text
Input Data

    |
    v

Data Loading

    |
    v

Input Validation

    |
    v

Dataset Profiling

    |
    v

Data Quality Audit

    |
    v

Automatic Cleaning

    |
    v

Feature Processing

    |
    v

Hardware Profiling

    |
    v

Model Selection

    |
    v

Optimization

    |
    v

Training

    |
    v

Evaluation

    |
    v

Explainability

    |
    v

Report Generation

    |
    v

SmartTabModel
````

The user only needs to provide:

* Data
* Target column
* Optional configuration

Everything else is automated.

---

# 3.2 Your First Complete Machine Learning Pipeline

## Dataset

Example:

`customer_churn.csv`

```text
customer_id, age, income, contract, usage, churn

1, 25, 35000, monthly, 20, 1
2, 45, 85000, yearly, 80, 0
3, 31, 50000, monthly, 40, 1
```

---

## Load Data

```python
import pandas as pd

from smarttab import fit


df = pd.read_csv(
    "customer_churn.csv"
)
```

Check:

```python
print(df.shape)
```

Output:

```text
(10000, 6)
```

---

## Train Model

```python
model = fit(
    data=df,
    target="churn"
)
```

SmartTab automatically discovers:

```text
Problem Type:
Binary Classification

Target:
churn

Features:
customer_id
age
income
contract
usage

Recommended Models:
CatBoost / LightGBM

Metric:
ROC-AUC
```

---

# 3.3 Understanding Automatic Task Detection

SmartTab automatically determines the machine learning task.

Example:

## Binary Classification

Input:

```python
target="churn"
```

Target:

```text
0
1
0
1
```

Detected:

```text
TaskType.BINARY
```

---

## Multiclass Classification

Target:

```text
cat
dog
bird
cat
```

Detected:

```text
TaskType.MULTICLASS
```

---

## Regression

Target:

```text
250000
320000
450000
```

Detected:

```text
TaskType.REGRESSION
```

---

## Multilabel Classification

Example:

```text
[
 [1,0,1],
 [0,1,1],
 [1,1,0]
]
```

Detected:

```text
TaskType.MULTILABEL
```

---

# 3.4 Using External Labels with `y=`

SmartTab supports separating features and labels.

Instead of:

```python
df:

age income target
20  50000 1
30  70000 0
```

You can provide:

```python
X = df_features

y = labels
```

Example:

```python
from smarttab import fit


model = fit(
    data=X,
    y=y
)
```

Internally SmartTab creates:

```text
Feature DataFrame

        +

Target Vector

        |

        v

Training Dataset
```

---

# 3.5 Predicting New Samples

After training:

```python
prediction = model.predict(
    new_data
)
```

Example:

```python
new_customer = pd.DataFrame(
    {
        "age": [35],
        "income": [65000],
        "contract": ["monthly"],
        "usage": [50]
    }
)


result = model.predict(
    new_customer
)

print(result)
```

Output:

```text
[1]
```

Meaning:

```text
Customer is predicted as churn.
```

---

# 3.6 Probability Prediction

For classification problems:

```python
probabilities = model.predict_proba(
    new_customer
)
```

Example output:

```python
[
    [
        0.18,
        0.82
    ]
]
```

Interpretation:

| Class     | Probability |
| --------- | ----------- |
| Not Churn | 18%         |
| Churn     | 82%         |

---

# 3.7 Inspecting the Trained Model

A SmartTab model stores much more than a trained estimator.

Example:

```python
print(model)
```

Output:

```text
SmartTabModel

=====================

Task:
Binary Classification

Model:
CatBoost

Metric:
ROC-AUC

Score:
0.94

Features:
38

Training:
25.3 seconds

Explainability:
Enabled

Ensemble:
Stacking
```

---

# 3.8 Dataset Profile

SmartTab stores dataset analysis results.

Access:

```python
profile = model.dataset_profile
```

Example:

```python
print(profile)
```

Contains:

```text
DatasetProfile

Samples:
50000

Features:
120

Numeric Features:
80

Categorical Features:
40

Missing Values:
Detected

Class Balance:
Checked
```

---

# 3.9 Viewing Feature Importance

SmartTab automatically calculates feature importance.

```python
importance = model.feature_importance

print(importance)
```

Example:

```text
[
 ("monthly_usage", 0.32),
 ("contract_type", 0.21),
 ("age", 0.08)
]
```

Meaning:

```text
monthly_usage
is the most influential feature.
```

---

# 3.10 Enabling Explainability

By default:

```python
model = fit(
    data=df,
    target="churn"
)
```

SmartTab provides native feature importance.

For advanced explanations:

```python
model = fit(
    data=df,
    target="churn",
    explain=True
)
```

This enables:

* SHAP values
* Local explanations
* Feature contribution analysis

---

# 3.11 Enabling Automatic Reports

```python
model = fit(
    data=df,
    target="churn",
    report=True
)
```

Generated report includes:

```text
Report

|
+-- Dataset Summary
|
+-- Data Quality Report
|
+-- Cleaning Report
|
+-- Hardware Information
|
+-- Model Details
|
+-- Metrics
|
+-- Feature Importance
|
+-- SHAP Analysis
|
+-- Charts
```

---

# 3.12 Training with Optimization

Default:

```python
fit(
    data=df,
    target="churn"
)
```

SmartTab may use default parameters.

Enable optimization:

```python
model = fit(
    data=df,
    target="churn",
    optimize=True
)
```

Optimization searches:

```text
Learning Rate

Tree Depth

Regularization

Number of Trees

Feature Parameters

Model Specific Parameters
```

---

# 3.13 Using Time Limits

For large datasets:

```python
model = fit(
    data=df,
    target="churn",
    time_limit=120
)
```

SmartTab creates a resource budget:

```text
Total Budget:
120 seconds


Dataset Analysis:
10 sec


Cleaning:
20 sec


Training:
60 sec


Explainability:
30 sec
```

If time is running out:

```text
Reduce optimization

Skip expensive SHAP

Reduce ensemble search
```

---

# 3.14 Automatic Ensemble Mode

Enable automatic ensemble selection:

```python
model = fit(
    data=df,
    target="churn",
    ensemble="auto"
)
```

SmartTab evaluates:

```text
Single Model

      vs

Voting Ensemble

      vs

Stacking Ensemble
```

Example decision:

```text
CatBoost:
0.91 ROC-AUC


Voting:
0.92 ROC-AUC


Stacking:
0.95 ROC-AUC


Selected:
Stacking Ensemble
```

---

# 3.15 Controlling Verbosity

For detailed logs:

```python
model = fit(
    data=df,
    target="churn",
    verbose=True
)
```

Example:

```text
Stage 1/9:
Dataset analysis

Stage 2/9:
Cleaning pipeline

Stage 3/9:
Hardware profiling

Stage 4/9:
Model selection

Stage 5/9:
Optimization

Stage 6/9:
Training

Stage 7/9:
Evaluation

Stage 8/9:
Explainability

Stage 9/9:
Report generation
```

---

# 3.16 Complete Production-Style Example

```python
import pandas as pd

from smarttab import fit


# Load dataset

data = pd.read_csv(
    "transactions.csv"
)


# Train SmartTab

model = fit(
    data=data,
    target="fraud",
    optimize=True,
    ensemble="auto",
    explain=True,
    report=True,
    time_limit=600,
    random_state=42
)


# Model information

print(
    model.model_name
)


print(
    model.metrics
)


# New prediction

samples = data.drop(
    columns=["fraud"]
).head(10)


predictions = model.predict(
    samples
)


print(predictions)
```

---

# 3.17 What Happens Internally?

When executing:

```python
fit(
    data=df,
    target="target"
)
```

SmartTab internally executes:

```text
_prepare_fit_input()

        ↓

load_data()

        ↓

_handle_missing_targets()

        ↓

_handle_duplicate_rows()

        ↓

resolve_task_and_targets()

        ↓

_split_train_test()

        ↓

analyze_dataset()

        ↓

audit_data_quality()

        ↓

SmartCleaningPipeline.fit_transform()

        ↓

profile_hardware()

        ↓

resolve_resource_plan()

        ↓

select_model()

        ↓

run_optimization()

        ↓

train_model()

        ↓

evaluate()

        ↓

get_feature_importance()

        ↓

SmartTabModel()
```

---

# 3.18 Summary

After this chapter you know:

* How to train your first SmartTab model
* How SmartTab detects tasks
* How prediction works
* How reports and explanations are enabled
* How optimization and ensembles are configured
* What happens internally during `fit()`

Next:

# Chapter 4 — Understanding `fit()` API in Detail

```
```


# Chapter 4 — Understanding the `fit()` API in Detail

## 4.1 Overview

The `fit()` function is the main entry point of SmartTab.

It is responsible for the complete automated machine learning workflow:

```python
from smarttab import fit

model = fit(
    data,
    target="target_column"
)
````

Internally, `fit()` performs:

```text
Input Preparation

        ↓

Data Validation

        ↓

Task Detection

        ↓

Train/Test Split

        ↓

Dataset Analysis

        ↓

Data Cleaning

        ↓

Feature Transformation

        ↓

Hardware Profiling

        ↓

Model Selection

        ↓

Optimization

        ↓

Training

        ↓

Calibration

        ↓

Evaluation

        ↓

Explainability

        ↓

Report Generation

        ↓

SmartTabModel
```

---

# 4.2 Function Signature

Complete signature:

```python
fit(
    data,
    target=None,
    group_id=None,
    *,
    y=None,
    modality="auto",
    modalities=None,
    **kwargs
)
```

---

# 4.3 Basic Parameters

## `data`

### Type

```python
DataFrame | str | Path | list | numpy.ndarray
```

### Description

The input dataset.

Supported inputs:

| Input            | Example              |
| ---------------- | -------------------- |
| pandas DataFrame | `pd.DataFrame()`     |
| CSV file         | `"data.csv"`         |
| Parquet file     | `"data.parquet"`     |
| Text samples     | `["hello world"]`    |
| Images           | image paths / arrays |
| Audio            | WAV arrays / files   |
| Video            | video paths          |

---

## Example: DataFrame

```python
import pandas as pd

df = pd.read_csv(
    "customers.csv"
)


model = fit(
    data=df,
    target="churn"
)
```

---

## Example: CSV Path

SmartTab can load files directly:

```python
model = fit(
    data="customers.csv",
    target="churn"
)
```

Internally:

```python
load_data("customers.csv")
```

is called.

---

# 4.4 `target`

## Type

```python
str | list[str] | None
```

## Description

Defines the prediction target.

The target is the value SmartTab learns to predict.

---

## Single Target

Example:

Dataset:

| age | income | churn |
| --- | ------ | ----- |
| 20  | 30000  | 1     |
| 40  | 80000  | 0     |

Use:

```python
model = fit(
    df,
    target="churn"
)
```

---

## Multiple Targets

Used for:

* Multi-output regression
* Multi-label classification

Example:

```python
model = fit(
    df,
    target=[
        "class_a",
        "class_b",
        "class_c"
    ]
)
```

---

# 4.5 `y`

## Type

```python
array-like
```

## Description

External target labels.

Used when features and labels are separated.

---

Example:

```python
X = pd.DataFrame(
    {
        "age":[20,30,40],
        "income":[30000,50000,90000]
    }
)


y = [
    1,
    0,
    1
]


model = fit(
    data=X,
    y=y
)
```

SmartTab internally creates:

```text
X

+

y

↓

Training Dataset
```

---

# 4.6 `group_id`

## Type

```python
str | None
```

## Description

Defines a grouping column.

Used to prevent leakage when multiple rows belong to the same entity.

Common examples:

| Domain         | Group       |
| -------------- | ----------- |
| Medical        | Patient ID  |
| Banking        | Customer ID |
| Recommendation | User ID     |
| Manufacturing  | Machine ID  |

---

Example:

Dataset:

| patient_id | age | disease |
| ---------- | --- | ------- |
| 1          | 45  | yes     |
| 1          | 46  | yes     |
| 2          | 30  | no      |

Training:

```python
model = fit(
    data=df,
    target="disease",
    group_id="patient_id"
)
```

SmartTab ensures:

```text
Patient 1

cannot appear in:

Training AND Testing
```

This prevents unrealistic evaluation.

---

# 4.7 `modality`

## Type

```python
str
```

## Default

```python
"auto"
```

---

## Description

Defines the input data modality.

Available:

| Value | Data Type           |
| ----- | ------------------- |
| auto  | Automatic detection |
| text  | Text                |
| image | Images              |
| audio | Audio               |
| video | Video               |

---

## Example: Text Classification

```python
texts = [
    "This movie is great",
    "Very bad experience"
]


labels = [
    1,
    0
]


model = fit(
    texts,
    y=labels,
    modality="text"
)
```

---

## Example: Image Classification

```python
images = [
    "cat1.jpg",
    "dog1.jpg"
]


labels = [
    "cat",
    "dog"
]


model = fit(
    images,
    y=labels,
    modality="image"
)
```

---

# 4.8 `modalities`

## Type

```python
dict | str | None
```

## Description

Used for datasets containing multiple feature types.

Example:

```python
modalities = {
    "description": "text",
    "photo": "image",
    "audio_file": "audio"
}
```

Usage:

```python
model = fit(
    data=df,
    target="category",
    modalities=modalities
)
```

SmartTab creates:

```text
Text Features

        +

Image Features

        +

Audio Features

        ↓

Unified Feature Space
```

---

# 4.9 Configuration Parameters

Most advanced settings are provided through `kwargs`.

These control:

* Model selection
* Optimization
* Hardware
* Cleaning
* Explainability
* Reports

---

# 4.10 Training Control Parameters

## `time_limit`

### Type

```python
float | None
```

Maximum training duration.

Example:

```python
model = fit(
    df,
    target="target",
    time_limit=300
)
```

Meaning:

```text
Maximum:
300 seconds
```

SmartTab dynamically allocates:

```text
Dataset Analysis

+

Training

+

Optimization

+

Explainability
```

---

## `random_state`

Controls reproducibility.

Example:

```python
model = fit(
    df,
    target="target",
    random_state=42
)
```

Controls:

* Dataset split
* Optimization randomness
* Ensemble selection

---

## `verbose`

Controls logging.

Example:

```python
model = fit(
    df,
    target="target",
    verbose=True
)
```

Output:

```text
Stage 1/9:
Dataset analysis

Stage 2/9:
Cleaning

Stage 3/9:
Hardware profiling
...
```

---

# 4.11 Model Selection Parameters

## `model`

Force a specific model.

Example:

```python
model = fit(
    df,
    target="target",
    model="catboost"
)
```

Possible values:

```text
catboost
lightgbm
xgboost
auto
```

---

Default:

```python
model="auto"
```

SmartTab decides.

---

# 4.12 Optimization Parameters

## `optimize`

Enable hyperparameter search.

Example:

```python
model = fit(
    df,
    target="target",
    optimize=True
)
```

Without optimization:

```text
One model
Default parameters
Fast training
```

With optimization:

```text
Multiple trials

Parameter search

Best configuration selected
```

---

## `optimizer`

Controls optimization backend.

Example:

```python
model = fit(
    df,
    target="target",
    optimizer="optuna"
)
```

---

## `n_trials`

Number of optimization attempts.

Example:

```python
model = fit(
    df,
    target="target",
    n_trials=100
)
```

Meaning:

```text
Try 100 parameter combinations
```

---

# 4.13 Ensemble Parameters

## `ensemble`

Controls ensemble behavior.

Values:

| Value    | Behavior          |
| -------- | ----------------- |
| none     | Single model      |
| auto     | Smart decision    |
| voting   | Voting ensemble   |
| stacking | Stacking ensemble |

---

Example:

```python
model = fit(
    df,
    target="target",
    ensemble="auto"
)
```

SmartTab compares:

```text
Single Model

vs

Voting

vs

Stacking
```

and selects the best.

---

## `ensemble_models_limit`

Maximum ensemble members.

Example:

```python
model = fit(
    df,
    target="target",
    ensemble="auto",
    ensemble_models_limit=5
)
```

Possible architecture:

```text
Model 1
Model 2
Model 3
Model 4
Model 5

      ↓

Meta Model

      ↓

Prediction
```

---

# 4.14 Explainability Parameters

## `explain`

Controls explanation generation.

Values:

| Value | Meaning                       |
| ----- | ----------------------------- |
| True  | Always calculate explanations |
| False | Disable                       |
| auto  | Smart decision                |

Example:

```python
model = fit(
    df,
    target="target",
    explain=True
)
```

---

## `report`

Enable HTML reports.

Example:

```python
model = fit(
    df,
    target="target",
    report=True
)
```

---

# 4.15 Hardware Parameters

## `device`

Control computation device.

Example:

CPU:

```python
model = fit(
    df,
    target="target",
    device="cpu"
)
```

GPU:

```python
model = fit(
    df,
    target="target",
    device="gpu"
)
```

---

## `cpu_threads`

Limit CPU usage.

Example:

```python
model = fit(
    df,
    target="target",
    cpu_threads=4
)
```

---

## `ram_limit`

Limit memory usage.

Example:

```python
model = fit(
    df,
    target="target",
    ram_limit="8GB"
)
```

---

# 4.16 Complete Advanced Example

```python
from smarttab import fit
import pandas as pd


df = pd.read_csv(
    "transactions.csv"
)


model = fit(
    data=df,

    target="fraud",

    group_id="customer_id",

    optimize=True,

    n_trials=50,

    ensemble="auto",

    ensemble_models_limit=5,

    explain=True,

    report=True,

    time_limit=600,

    device="gpu",

    random_state=42
)
```

This configuration enables:

```text
✓ Leakage-safe splitting

✓ Automatic cleaning

✓ Hyperparameter optimization

✓ Ensemble search

✓ GPU acceleration

✓ Explainability

✓ HTML reporting
```

---

# 4.17 Summary

The `fit()` function is the complete SmartTab automation interface.

Main inputs:

| Parameter  | Purpose               |
| ---------- | --------------------- |
| data       | Dataset               |
| target     | Prediction target     |
| y          | External labels       |
| group_id   | Leakage prevention    |
| modality   | Input type            |
| optimize   | Hyperparameter search |
| ensemble   | Model combination     |
| explain    | Explainability        |
| report     | Reports               |
| time_limit | Resource control      |


```