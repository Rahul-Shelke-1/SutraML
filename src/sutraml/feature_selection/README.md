# Feature Selection Framework

## Objective

Build a reusable feature-selection framework that allows users to:

1. Select a feature-selection strategy.
2. Select one or more ML models.
3. Evaluate candidate feature subsets against a configurable metric.
4. Identify the feature subset that provides the best model performance.
5. Compare feature-selection strategies and model combinations.
6. Return a reproducible feature-selection report.

The framework will support three major feature-selection families:

* **Filter Methods**
* **Wrapper Methods**
* **Embedded Methods**

The goal is not only to answer:

> "Which features are statistically important?"

but also:

> "Which subset of features gives the best predictive performance for a given model and evaluation objective?"

---

# 1. High-Level Architecture

```text
                    Dataset
                       │
                       ▼
              Feature / Target Split
                       │
                       ▼
             Problem Type Detection
              ┌────────┴────────┐
              │                 │
       Classification       Regression
              │                 │
              └────────┬────────┘
                       ▼
              Feature Selection
              Strategy Selection
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Filter         Wrapper        Embedded
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                Candidate Features
                       │
                       ▼
                 Model Selection
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Model A      Model B      Model C
          │            │            │
          └────────────┼────────────┘
                       ▼
                Cross Validation
                       │
                       ▼
                 Evaluation Metric
                       │
                       ▼
               Feature Subset Ranking
                       │
                       ▼
                Best Feature Set
                       │
                       ▼
                  Final Report
```

---

# 2. Feature Selection Strategies

## 2.1 Filter Methods

Filter methods evaluate features independently of a specific ML estimator.

### Planned methods

#### Statistical methods

* Pearson correlation
* Spearman correlation
* Kendall correlation
* Chi-square test
* ANOVA
* t-test
* Mann-Whitney U test

#### Statistical scoring

* F-test

  * `f_classif`
  * `f_regression`

#### Dependency-based methods

* Mutual Information

  * `mutual_info_classif`
  * `mutual_info_regression`

#### Unsupervised filtering

* Variance threshold
* Near-zero variance
* Highly correlated feature removal

### Output

```text
Feature
Score
P-Value
Method
Rank
Selected
```

Filter methods should produce a ranked candidate feature set that can subsequently be passed to model training.

---

# 3. Wrapper Methods

Wrapper methods evaluate feature subsets using an actual ML model.

The model becomes part of the feature-selection process.

## 3.1 Forward Selection

Start with no features.

```text
[]
 │
 ├── + Feature A → Score
 ├── + Feature B → Score
 ├── + Feature C → Score
 └── + Feature D → Score

Select best feature
        │
        ▼
      [A]

Repeat:
[A] + B
[A] + C
[A] + D

Select best combination
        │
        ▼
      [A,C]
```

Continue until:

* all features are selected, or
* maximum feature count is reached, or
* performance improvement falls below threshold.

---

# 3.2 Backward Elimination

Start with all features.

```text
[A,B,C,D,E]
      │
      ├── Remove A → Score
      ├── Remove B → Score
      ├── Remove C → Score
      ├── Remove D → Score
      └── Remove E → Score

Remove feature producing the best result
             │
             ▼
          [A,C,D,E]

Repeat
```

Stopping conditions:

* minimum feature count reached
* no meaningful improvement
* maximum iterations reached

---

# 3.3 Recursive Feature Elimination

Use model-estimated feature importance to recursively eliminate features.

Potential implementations:

* RFE
* RFECV

The framework should expose:

```text
estimator
step
min_features
cv
scoring
```

---

# 3.4 Sequential Feature Selection

Provide a general interface for:

* Forward Selection
* Backward Selection

Potential implementation using:

```text
SequentialFeatureSelector
```

but wrapped behind our own project API.

---

# 4. Embedded Methods

Embedded methods perform feature selection during model training.

## Planned methods

### Linear models

* Lasso
* Elastic Net
* Logistic Regression with L1
* Linear models with L1/L2 regularization

### Tree-based models

* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

### Importance-based selection

Support:

* model coefficients
* impurity-based importance
* permutation importance

Potential abstraction:

```text
Estimator
    │
    ▼
Fit Model
    │
    ▼
Extract Importance
    │
    ▼
Rank Features
    │
    ▼
Select Top-K
```

---

# 5. Model Configuration

The framework should allow the user to provide multiple models.

Example concept:

```python
model_configs = {
    "logistic_regression": {...},
    "random_forest": {...},
    "xgboost": {...},
    "lightgbm": {...},
    "catboost": {...},
}
```

The feature-selection framework should not hard-code a single estimator.

Instead:

```text
Feature Selection
        +
Model Configuration
        +
Evaluation Strategy
        =
Feature Selection Experiment
```

---

# 6. Evaluation Strategy

Feature selection should be driven by an evaluation metric.

For classification:

* ROC-AUC
* PR-AUC
* F1
* Precision
* Recall
* Accuracy
* Balanced Accuracy
* Log Loss

For regression:

* MAE
* MSE
* RMSE
* R²
* MAPE

The API should allow:

```python
scoring="roc_auc"
```

or a custom scoring function.

---

# 7. Cross Validation

Feature-selection evaluation should use cross-validation rather than relying on a single train/test split.

Conceptually:

```text
Feature Subset
      │
      ▼
Cross Validation
 ┌────┼────┐
 ▼    ▼    ▼
CV1  CV2  CV3 ...
 │    │    │
 └────┼────┘
      ▼
 Mean CV Score
      │
      ▼
Feature Subset Score
```

For the heart-stroke problem, stratified CV should be preferred for classification because of class imbalance.

---

# 8. Feature Selection Experiment

The central abstraction should eventually look conceptually like:

```python
FeatureSelectionExperiment(
    X=X,
    y=y,
    models=model_configs,
    method="wrapper",
    scoring="roc_auc",
    cv=5,
)
```

The experiment evaluates:

```text
                    Feature Selection
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
        Filter           Wrapper         Embedded
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                    Candidate Subsets
                           │
                           ▼
                     Model Evaluation
                           │
                           ▼
                     CV Score
                           │
                           ▼
                      Ranking
```

---

# 9. Experiment Result

The framework should return structured results rather than only printing the selected features.

Example conceptual result:

```text
Method: Wrapper
Strategy: Forward Selection
Model: Logistic Regression
Metric: ROC-AUC
CV: StratifiedKFold(5)

Selected Features:
    age
    hypertension
    heart_disease
    avg_glucose_level
    bmi

Number of Features:
    5

CV Score:
    0.842

Baseline Score:
    0.811

Improvement:
    +0.031
```

---

# 10. Comparison Across Models

A major capability of the framework should be comparing the same feature-selection strategy across different models.

Example:

| Method   | Model                  | Features | CV ROC-AUC |
| -------- | ---------------------- | -------: | ---------: |
| Filter   | Logistic Regression    |        8 |       0.81 |
| Filter   | Random Forest          |        8 |       0.83 |
| Wrapper  | Logistic Regression    |        5 |       0.84 |
| Wrapper  | Random Forest          |        6 |       0.85 |
| Embedded | Logistic Regression L1 |        7 |       0.82 |
| Embedded | Random Forest          |        6 |       0.85 |

This makes the framework useful for **model + feature-subset discovery**, rather than merely feature ranking.

---

# 11. Baseline Comparison

Every feature-selection experiment should optionally compare against the baseline model using all available features.

```text
All Features
     │
     ▼
Baseline Model
     │
     ▼
Baseline Score
     │
     │
     ├─────────────────────┐
     │                     │
     ▼                     ▼
Feature Selection      Feature Selection
     │                     │
     ▼                     ▼
Reduced Features       Reduced Features
     │                     │
     ▼                     ▼
Selected Model         Selected Model
     │                     │
     └──────────┬──────────┘
                ▼
          Compare Scores
```

This prevents selecting a smaller feature set merely because it looks statistically attractive.

---

# 12. Feature Selection Report

The final report should capture:

### Experiment metadata

* feature-selection method
* selection strategy
* estimator
* scoring metric
* CV strategy
* random state
* parameters

### Feature information

* original feature count
* selected feature count
* selected features
* removed features

### Performance

* baseline score
* selected-feature score
* score improvement
* mean CV score
* standard deviation

### Ranking

```text
Rank
Feature
Score
Selected
```

---

# 13. Reproducibility

Every experiment should preserve enough configuration to reproduce the result.

```text
Dataset
   +
Feature Selection Configuration
   +
Model Configuration
   +
CV Configuration
   +
Random State
   =
Reproducible Experiment
```

Where possible, random states should be explicitly controlled.

---

# 14. Proposed Package Structure

```text
src/heart_stroke_prediction/analyze/feature_selection/

├── __init__.py
│
├── base.py
│
├── filter/
│   ├── __init__.py
│   ├── correlation.py
│   ├── statistical.py
│   ├── information.py
│   └── variance.py
│
├── wrapper/
│   ├── __init__.py
│   ├── sequential.py
│   ├── forward.py
│   ├── backward.py
│   └── recursive.py
│
├── embedded/
│   ├── __init__.py
│   ├── regularization.py
│   ├── tree_importance.py
│   └── permutation.py
│
├── experiment.py
├── result.py
└── utils.py
```

The exact structure is intentionally not final yet.

The existing `filter_methods.py` can be refactored gradually rather than rewriting everything at once. The current implementation already contains several useful filter primitives.

---

# 15. Development Phases

## Phase 1 — Common Interface

* [ ] Define `BaseFeatureSelector`
* [ ] Define common `fit()`
* [ ] Define common `transform()`
* [ ] Define `fit_transform()`
* [ ] Define `get_support()`
* [ ] Define `get_feature_names_out()`
* [ ] Define common configuration/result representation

## Phase 2 — Filter Methods

* [ ] Refactor existing filter implementation
* [ ] Correlation filtering
* [ ] Variance filtering
* [ ] Statistical filtering
* [ ] F-test
* [ ] Mutual Information
* [ ] Feature ranking
* [ ] Threshold / top-k selection

## Phase 3 — Wrapper Methods

* [ ] Forward selection
* [ ] Backward elimination
* [ ] Sequential selection
* [ ] RFE
* [ ] RFECV
* [ ] CV-based scoring
* [ ] configurable stopping criteria

## Phase 4 — Embedded Methods

* [ ] L1-based selection
* [ ] Elastic Net
* [ ] tree-based importance
* [ ] permutation importance
* [ ] top-k importance selection
* [ ] threshold-based selection

## Phase 5 — Model Evaluation

* [ ] Model registry/configuration
* [ ] Multiple estimators
* [ ] Scoring abstraction
* [ ] Classification metrics
* [ ] Regression metrics
* [ ] Cross-validation abstraction

## Phase 6 — Experiment Engine

* [ ] Run feature selection against multiple models
* [ ] Compare feature subsets
* [ ] Compare baseline vs selected features
* [ ] Rank experiments
* [ ] Store experiment configuration
* [ ] Return structured result

## Phase 7 — Reporting

* [ ] Feature ranking report
* [ ] Model comparison report
* [ ] Feature-count vs performance report
* [ ] Baseline comparison
* [ ] Selection history for wrapper methods
* [ ] Export results

## Phase 8 — Testing

* [ ] Unit tests for every selector
* [ ] Edge-case tests
* [ ] Classification tests
* [ ] Regression tests
* [ ] Small synthetic datasets
* [ ] Integration tests
* [ ] Reproducibility tests

---

# 16. First MVP

Do not implement the entire framework at once.

The first useful milestone should be:

```text
                    Feature Selection MVP

                           Dataset
                              │
                              ▼
                       Feature Selector
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
              Filter        Wrapper       Embedded
                │             │             │
                ▼             ▼             ▼
             Ranking      Subset Search   Importance
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                        Model Evaluation
                              │
                              ▼
                         CV ROC-AUC
                              │
                              ▼
                       Best Feature Set
```

### MVP scope

1. One clean common selector interface.
2. Filter:

   * F-test
   * Mutual Information
   * correlation
3. Wrapper:

   * Forward Selection
   * Backward Selection
4. Embedded:

   * L1 Logistic Regression
   * Tree-based importance
5. Multiple configurable models.
6. Cross-validation.
7. Configurable scoring metric.
8. Baseline comparison.
9. Structured result object.

Everything else can build on top of this.

---

# 17. Design Principle

The important architectural distinction is:

```text
                 Feature Selection Method
                          │
                          ▼
                   Candidate Features
                          │
                          ▼
                    Model + Metric
                          │
                          ▼
                    Cross Validation
                          │
                          ▼
                      Score
                          │
                          ▼
                  Feature Subset Ranking
```

The framework should therefore keep these concerns separate:

```text
Selection Algorithm
        ≠
Model
        ≠
Evaluation Metric
        ≠
Cross Validation
        ≠
Reporting
```

This separation will make the feature-selection system extensible instead of turning it into one large class containing every selection algorithm.
