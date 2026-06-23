---
name: effective-machine-learning
description: Use this skill when Codex needs to build, debug, evaluate, tune, or explain machine learning models during fast coding or data-science work. Trigger for supervised ML, classification, regression, churn prediction, tabular modeling, model comparison, cross-validation, imbalanced data, feature engineering, hyperparameter tuning, feature importance, SHAP, metric interpretation, model reports, or production-minded ML workflows.
---

# Effective Machine Learning

## Core Posture

Build useful models quickly, but keep the science honest. Move fast through setup and implementation, then slow down at the places where ML projects usually go wrong: leakage, bad splits, misleading metrics, imbalanced targets, untested assumptions, and overclaiming weak results.

Prefer a complete, reproducible workflow over a clever one-off notebook. When the user gives a dataset, instruction file, business context, or target variable, treat those as the source of truth and make all modeling choices traceable to them.

## Quick Start

For most ML requests, follow this loop:

1. Read the dataset and any instruction/context files.
2. Identify the target, prediction task, row count, feature types, missing values, and class/target distribution.
3. Create a leakage-safe train/test split; use stratification for classification when possible.
4. Build a baseline model first.
5. Train 2-4 sensible candidate models for comparison.
6. Use cross-validation on the training data to estimate stability.
7. Tune the best model family using an appropriate metric.
8. Evaluate once on the held-out test set.
9. Explain the model with feature importance, coefficients, permutation importance, or SHAP as appropriate.
10. Save artifacts: code, model, metrics tables, plots, predictions/risk scores, and a short report.

## Problem Framing

Before modeling, clarify or infer:

- **Task type:** classification, regression, ranking, forecasting, clustering, or anomaly detection.
- **Prediction target:** exact column name, positive class, and business meaning.
- **Prediction timing:** what information would be available at scoring time.
- **Success metric:** business-relevant primary metric plus secondary guardrail metrics.
- **Deployment use:** decision automation, prioritization list, analyst insight, dashboard, experiment, or prototype.

If the target column is named differently from the user's wording, use the real dataset column and state the mapping clearly.

## Leakage Checks

Actively search for leakage before training:

- Remove row identifiers, names, account numbers, timestamps that encode future outcomes, post-event labels, and direct target aliases.
- Check suspiciously strong features with names like `status`, `closed`, `cancelled`, `churn_reason`, `outcome`, `resolved_after`, or `future_*`.
- Split before resampling, scaling, feature selection, target encoding, or imputation learned from data.
- For time-dependent data, prefer time-based validation over random splitting.
- Never use the test set for feature selection, threshold selection, hyperparameter tuning, or repeated decision-making.

If leakage is possible but uncertain, document it and run a conservative version without suspicious features.

## Data Review

Always report:

- Dataset shape.
- Target distribution or target summary.
- Missing values.
- Numerical, categorical, binary, date/time, and identifier features.
- Duplicate rows or duplicate IDs when relevant.
- Obvious outliers or impossible values.
- Whether preprocessing appears already completed.

For tabular work, avoid ad hoc parsing when structured tools are available. Use pandas or an equivalent dataframe library.

## Splitting and Validation

Choose validation based on the data:

- **Classification:** `train_test_split(..., stratify=y)` and `StratifiedKFold`.
- **Regression:** random split or K-fold unless time/order/group structure exists.
- **Grouped data:** `GroupKFold` or group-aware train/test split.
- **Time series / temporal data:** chronological split and rolling or expanding validation.
- **Small data:** use cross-validation, but keep one final holdout if feasible.

Use a fixed random seed for reproducibility. Fit preprocessing only inside a pipeline to avoid leakage.

## Preprocessing

Use pipelines whenever transformations are learned from data.

Recommended defaults:

- Numeric missing values: median imputation.
- Categorical missing values: most frequent or explicit `"Missing"` category.
- Categorical encoding: one-hot encoding for low/medium cardinality; target encoding only inside cross-validation.
- Scaling: required for linear models, SVMs, KNN, neural networks; usually unnecessary for trees.
- Feature selection: perform inside cross-validation if it is learned.

If the dataset is already cleaned and encoded, do not invent extra preprocessing. State that no additional encoding/scaling is required for tree models.

## Model Selection

Start simple, then compare against stronger models.

For tabular classification:

- Logistic Regression: interpretable baseline; good for linear signal and coefficient explanations.
- Decision Tree: easy to explain; prone to overfitting.
- Random Forest: strong default for non-linear tabular data; robust; supports feature importance.
- Gradient Boosting / HistGradientBoosting / XGBoost / LightGBM / CatBoost: high performance; tuning-sensitive.
- Neural Network: use only when data volume and complexity justify lower interpretability.

For tabular regression:

- Linear/Ridge/Lasso baseline.
- Random Forest or Extra Trees for robust non-linear baseline.
- Gradient boosting for higher accuracy.

For text, images, audio, or embeddings, prefer proven domain models instead of forcing tabular techniques.

Honor user model requirements. If the user says "use Random Forest", still compare a baseline when useful, but make Random Forest the final or primary model unless results clearly prove it unsuitable.

## Imbalanced Classification

Do not rely on accuracy alone when classes are imbalanced.

Use some combination of:

- Stratified split and stratified cross-validation.
- Class weights.
- Resampling inside the training pipeline only.
- SMOTE or variants only after splitting and only inside cross-validation/training pipelines.
- Threshold tuning based on out-of-fold training predictions, not the test set.

Report precision, recall, F1, ROC-AUC, confusion matrix, and often PR-AUC. Explain false positives and false negatives in business terms.

## Tuning

Tune only after establishing baselines.

Use:

- `RandomizedSearchCV` for broad search.
- `GridSearchCV` for small, focused grids.
- Stratified CV for classification.
- A primary scoring metric aligned to the business objective.

Good tuning targets:

- Churn/fraud/risk screening: recall, F2, PR-AUC, or recall at constrained precision.
- Costly outreach: precision, F1, or expected value.
- Ranking: ROC-AUC, PR-AUC, lift, or top-decile capture.
- Regression: MAE/RMSE/R2 based on business loss.

Keep the search space realistic. Do not spend long tuning a model if data quality or leakage uncertainty is the main issue.

## Evaluation

Evaluate every trained model with a comparison table.

For classification include:

- Accuracy.
- Precision.
- Recall.
- F1 or F-beta.
- ROC-AUC and/or PR-AUC.
- Confusion matrix.
- Cross-validation mean and standard deviation.
- Threshold analysis when predictions drive action.

For regression include:

- MAE.
- RMSE.
- R2.
- Residual diagnostics.
- Error by important segment if available.

Interpret metrics plainly:

- Accuracy can look high while missing the minority class.
- Recall answers "how many real positives did we catch?"
- Precision answers "how many flagged positives were actually positive?"
- ROC-AUC measures ranking quality across thresholds.
- Confusion matrix turns model errors into operational consequences.

If the model performs poorly, say so directly and recommend data/model improvements. Do not dress weak metrics up as production-ready.

## Explainability

Choose explanation tools by model type:

- Linear models: coefficients, odds ratios, standardized coefficients.
- Tree ensembles: impurity importance plus permutation importance when possible.
- Boosted trees and Random Forest: SHAP when available and computationally reasonable.
- Any model: partial dependence or simple segment analysis for stakeholder-friendly interpretation.

Translate technical drivers into actions:

- "Low engagement" becomes reactivation campaign.
- "Unresolved complaints" becomes service recovery.
- "Declining transactions" becomes relationship manager outreach.
- "Tenure risk" becomes onboarding or loyalty treatment.

State that feature importance is predictive association, not causal proof.

## Deliverables

For a full ML task, produce as many of these as fit the request:

- Reproducible script or notebook.
- Saved model artifact.
- Metrics comparison table.
- Cross-validation results.
- Hyperparameter search results.
- Confusion matrix or residual plots.
- ROC/PR curves for classification.
- Feature importance or SHAP charts.
- Prediction/risk-score output for operational use.
- Business-focused report with limitations and next steps.
- Requirements file when new dependencies are used.

Organize outputs in a clear folder such as `outputs/figures`, `outputs/tables`, and `outputs/models`.

## Reporting Style

Write reports for both data scientists and business stakeholders:

- Start with the recommendation and the evidence.
- Include dataset facts and validation method.
- Explain why the selected model fits the use case.
- Compare models clearly.
- Discuss class imbalance or skewed targets.
- Explain metrics in business terms.
- Highlight the most important drivers.
- Include limitations and next steps.
- Avoid claiming causality unless the design supports it.

For business-critical predictions, include operational guidance: risk bands, thresholds, campaign capacity, monitoring, retraining cadence, drift checks, and human review points.

## Production Guardrails

Before calling a model production-ready, check:

- The split reflects future use.
- No target leakage is present.
- Metrics are stable across folds and key segments.
- Probability scores are calibrated if used as probabilities.
- Thresholds are chosen on validation or out-of-fold predictions.
- Monitoring is defined for input drift, prediction drift, performance decay, and data quality.
- Retraining criteria are documented.
- Model artifacts and code are reproducible.

When in doubt, label the result as a prototype, pilot, or analysis model rather than a production model.

## Vibe Coding Rules

Move quickly, but keep these rules:

- Never skip the baseline.
- Never tune on the test set.
- Never resample before splitting.
- Never use accuracy alone for imbalanced classification.
- Never hide poor performance.
- Never interpret feature importance as causality.
- Always save the code that generated the result.
- Always leave the user with the next practical action.

## Useful Prompt Patterns

Use prompts like:

- "Use `$effective-machine-learning` to build a churn model from this CSV, compare models, tune Random Forest, and write a business report."
- "Use `$effective-machine-learning` to review this notebook for leakage, weak validation, and misleading metrics."
- "Use `$effective-machine-learning` to turn this quick model into a reproducible training script with saved artifacts."
- "Use `$effective-machine-learning` to improve recall for this imbalanced classifier and explain the precision trade-off."
- "Use `$effective-machine-learning` to check if this time-series split is leakage-free and suggest improvements."
