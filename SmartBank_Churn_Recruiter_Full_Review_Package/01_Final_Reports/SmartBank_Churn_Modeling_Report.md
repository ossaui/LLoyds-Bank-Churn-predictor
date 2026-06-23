# SmartBank Customer Churn Prediction Report

Generated: 2026-05-25 01:08

## Executive Summary

This analysis used the cleaned dataset `Advanced_Churn_Analysis_Dataset.csv` to build and evaluate binary churn prediction models for SmartBank / Lloyds Banking Group. The file uses `ChurnStatus` as the churn target; this is treated as the required binary churn indicator.

The recommended model family is **Random Forest** because it balances tabular predictive modeling, feature-importance explainability, class-weight support, and practical scalability for customer-retention workflows. However, the supplied dataset shows weak churn separability: the tuned Random Forest default threshold achieved **Accuracy 0.670**, **Precision 0.143**, **Recall 0.122**, **F1 0.132**, and **ROC-AUC 0.483** on the held-out test set. This should be treated as a pilot model, not an automated production decision engine, until richer behavioral features improve rank-ordering quality.

Because false negatives are costly in churn prevention, a recall-focused operating threshold of **0.370** was selected from out-of-fold training predictions with a campaign-capacity guardrail. At that action threshold the held-out test metrics were **Precision 0.188**, **Recall 0.634**, and **F1 0.291**. This threshold is useful for screening, but the low precision means retention teams should combine the score with business rules and segment-level prioritization.

## 1. Dataset Review and Problem Understanding

- Dataset rows: **1000**
- Dataset columns: **27**
- Target variable used: **ChurnStatus**
- Problem type: **Binary classification**
- Missing values detected: **0**
- Feature columns used for modeling: **25**
- Numerical features: **26**
- Already encoded binary indicator features: **10**
- Non-numeric categorical features requiring encoding: **0**

### Churn Class Distribution

| Class | Customer Count | Share |
| ----- | -------------- | ----- |
| 0     | 796            | 0.796 |
| 1     | 204            | 0.204 |

The churn distribution shows a **meaningful class imbalance**. Stratified train-test splitting and stratified cross-validation were used so each evaluation fold preserves the churn/non-churn ratio. This matters because ordinary random splitting can accidentally under-represent churners, causing unstable recall and misleading accuracy.

![Churn distribution](figures/churn_distribution.png)

## 2. Feature Preparation

The workflow separated features `X` from the target `y`, removed `CustomerID` from model training because it is an identifier rather than a behavioral predictor, and used an 80/20 train-test split with stratification.

All model features were already numeric or binary encoded, so no additional one-hot encoding was required. Feature scaling was applied in the Logistic Regression pipeline because linear models are sensitive to feature scale. Tree-based models such as Random Forest, Gradient Boosting, and XGBoost do not require scaling because they split on feature thresholds.

## 3. Algorithm Selection

| Algorithm           | Strengths                                                          | Limitations                                                           | Business Fit                                                                |
| ------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Logistic Regression | Simple, transparent coefficients, fast baseline.                   | Linear boundary; may miss non-linear churn interactions.              | Useful benchmark and explainable scorecard-style model.                     |
| Decision Tree       | Easy to explain with rule paths.                                   | Can overfit and be unstable without pruning.                          | Good for workshops, weaker as sole enterprise model.                        |
| Random Forest       | Strong accuracy, handles non-linearity, robust feature importance. | Less transparent than one tree, but explainable with importance/SHAP. | Recommended balance of performance, reliability, and stakeholder usability. |
| Gradient Boosting   | High predictive accuracy on tabular data.                          | More tuning-sensitive and can be less intuitive.                      | Strong challenger model when maximum lift is required.                      |
| XGBoost             | Advanced boosted trees, regularization, high tuning flexibility.   | More operational complexity and explanation burden.                   | Excellent challenger; use when lift outweighs simplicity.                   |
| Neural Networks     | Can learn complex patterns in very large datasets.                 | Less interpretable and needs more data/monitoring.                    | Not preferred here because stakeholder explainability is important.         |

**Final selection:** Random Forest is the best fit for this banking churn use case because it captures non-linear customer behavior, supports class weighting, and gives feature-importance outputs that can be translated into retention actions. Given the current test performance, it should be used as a transparent pilot model while the data science team improves the signal with stronger behavioral and historical features.

## 4. Model Building and Cross-Validation

The following models were trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost. Logistic Regression served as the baseline. Random Forest was used as the final model family and was tuned.

Stratified 5-fold cross-validation was used to check model stability:

| Model               | accuracy_mean | precision_mean | recall_mean | f1_mean | roc_auc_mean |
| ------------------- | ------------- | -------------- | ----------- | ------- | ------------ |
| Logistic Regression | 0.532         | 0.204          | 0.448       | 0.280   | 0.500        |
| Decision Tree       | 0.535         | 0.231          | 0.539       | 0.320   | 0.550        |
| Random Forest       | 0.795         | 0              | 0           | 0       | 0.591        |
| Gradient Boosting   | 0.791         | 0.520          | 0.092       | 0.152   | 0.543        |
| XGBoost             | 0.700         | 0.271          | 0.282       | 0.275   | 0.575        |
| Tuned Random Forest | 0.734         | 0.283          | 0.178       | 0.216   | 0.552        |

![Cross-validation comparison](figures/cross_validation_comparison.png)

## 5. Class Imbalance Handling

The imbalance strategy compared three Random Forest approaches: no balancing, class weighting, and SMOTE oversampling applied only within the training data. This avoids leaking synthetic examples into the test set.

| Model             | Accuracy | Precision | Recall | F1 Score | ROC-AUC | False Negatives | False Positives |
| ----------------- | -------- | --------- | ------ | -------- | ------- | --------------- | --------------- |
| RF - No Balancing | 0.795    | 0         | 0      | 0        | 0.502   | 41              | 0               |
| RF - Class Weight | 0.795    | 0         | 0      | 0        | 0.522   | 41              | 0               |
| RF - SMOTE        | 0.770    | 0.222     | 0.049  | 0.080    | 0.443   | 39              | 7               |

Class weighting is operationally simple and keeps the original customer population intact, while SMOTE can improve minority-class learning when churners are under-represented. For banking deployment, class weighting is often the first practical option because it is transparent, reproducible, and easy to monitor.

## 6. Hyperparameter Tuning

RandomizedSearchCV was used with stratified 5-fold cross-validation. The search optimized Random Forest parameters including number of trees, maximum depth, minimum split size, minimum leaf size, feature sampling, and class weighting. The refit metric was **recall** because the business priority is identifying as many true churn-risk customers as possible.

Best Random Forest parameters:

```json
{
  "n_estimators": 200,
  "min_samples_split": 10,
  "min_samples_leaf": 4,
  "max_features": null,
  "max_depth": 5,
  "class_weight": "balanced_subsample"
}
```

Top tuning results by recall:

| rank_test_recall | mean_test_recall | mean_test_roc_auc | mean_test_f1 | mean_test_precision | param_n_estimators | param_max_depth | param_min_samples_split | param_min_samples_leaf | param_max_features | param_class_weight |
| ---------------- | ---------------- | ----------------- | ------------ | ------------------- | ------------------ | --------------- | ----------------------- | ---------------------- | ------------------ | ------------------ |
| 1                | 0.178            | 0.552             | 0.216        | 0.283               | 200                | 5               | 10                      | 4                      | nan                | balanced_subsample |
| 2                | 0.154            | 0.545             | 0.196        | 0.279               | 120                | 5               | 5                       | 2                      | log2               | balanced_subsample |
| 3                | 0.154            | 0.551             | 0.200        | 0.298               | 300                | 5               | 10                      | 1                      | sqrt               | balanced           |
| 4                | 0.049            | 0.558             | 0.083        | 0.315               | 120                | 8               | 10                      | 1                      | log2               | balanced_subsample |
| 5                | 0.037            | 0.572             | 0.065        | 0.309               | 300                | 16              | 10                      | 4                      | nan                | balanced           |
| 6                | 0.031            | 0.562             | 0.053        | 0.277               | 300                | 8               | 5                       | 4                      | sqrt               | balanced           |
| 6                | 0.031            | 0.575             | 0.055        | 0.275               | 200                | None            | 2                       | 4                      | nan                | balanced           |
| 8                | 0.025            | 0.565             | 0.042        | 0.211               | 200                | 8               | 5                       | 4                      | sqrt               | balanced           |
| 9                | 0.018            | 0.578             | 0.034        | 0.333               | 200                | 8               | 5                       | 1                      | nan                | nan                |
| 10               | 0.018            | 0.563             | 0.032        | 0.140               | 300                | 12              | 2                       | 4                      | log2               | balanced_subsample |

## 7. Final Model Evaluation

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC | False Negatives | False Positives |
| ------------------- | -------- | --------- | ------ | -------- | ------- | --------------- | --------------- |
| Logistic Regression | 0.465    | 0.189     | 0.488  | 0.272    | 0.507   | 21              | 86              |
| Decision Tree       | 0.500    | 0.183     | 0.415  | 0.254    | 0.494   | 24              | 76              |
| Random Forest       | 0.790    | 0.333     | 0.024  | 0.045    | 0.493   | 40              | 2               |
| Gradient Boosting   | 0.780    | 0.286     | 0.049  | 0.083    | 0.492   | 39              | 5               |
| XGBoost             | 0.555    | 0.147     | 0.244  | 0.183    | 0.438   | 31              | 58              |
| Tuned Random Forest | 0.670    | 0.143     | 0.122  | 0.132    | 0.483   | 36              | 30              |

- Best model by recall: **Logistic Regression** with recall **0.488**
- Best model by ROC-AUC: **Logistic Regression** with ROC-AUC **0.507**
- Recommended model: **Tuned Random Forest**

### Recall-Focused Operating Threshold

The standard 0.50 threshold is not always appropriate for churn prevention because it can miss too many churners. A threshold was selected using out-of-fold training predictions, prioritizing recall while keeping the flagged population near a manageable campaign size where possible.

Selected operating threshold: **0.370**

Held-out performance at selected threshold:

| Model                                     | Threshold | Accuracy | Precision | Recall | F1 Score | F2 Score | ROC-AUC | False Negatives | False Positives |
| ----------------------------------------- | --------- | -------- | --------- | ------ | -------- | -------- | ------- | --------------- | --------------- |
| Tuned Random Forest - Operating Threshold | 0.370     | 0.365    | 0.188     | 0.634  | 0.291    | 0.430    | 0.483   | 15              | 112             |

This operating threshold reduces false negatives from **36** to **15**, but increases false positives from **30** to **112**. That trade-off is acceptable only if campaign capacity and intervention cost can support a broad screening list.

Top candidate thresholds from cross-validated training predictions:

| Threshold | Accuracy | Precision | Recall | F1 Score | F2 Score | Flagged Share | Selected |
| --------- | -------- | --------- | ------ | -------- | -------- | ------------- | -------- |
| 0.370     | 0.477    | 0.226     | 0.644  | 0.334    | 0.470    | 0.581         | True     |
| 0.240     | 0.244    | 0.205     | 0.945  | 0.337    | 0.549    | 0.938         | False    |
| 0.250     | 0.251    | 0.205     | 0.933  | 0.337    | 0.546    | 0.925         | False    |
| 0.320     | 0.354    | 0.213     | 0.804  | 0.336    | 0.517    | 0.770         | False    |
| 0.260     | 0.260    | 0.206     | 0.920  | 0.336    | 0.543    | 0.911         | False    |
| 0.270     | 0.278    | 0.207     | 0.896  | 0.336    | 0.537    | 0.884         | False    |
| 0.200     | 0.220    | 0.203     | 0.963  | 0.335    | 0.550    | 0.969         | False    |
| 0.360     | 0.448    | 0.222     | 0.681  | 0.334    | 0.481    | 0.626         | False    |

### Confusion Matrix Interpretation

For the Tuned Random Forest at the default 0.50 threshold:

- True negatives: **129** customers correctly predicted as retained.
- False positives: **30** customers flagged as churn-risk who did not churn. These may receive retention outreach unnecessarily.
- False negatives: **36** churners missed by the model. These are the highest business concern because retention teams may not intervene.
- True positives: **5** churners correctly identified for proactive retention.

![Confusion matrix](figures/confusion_matrix_random_forest.png)

### ROC-AUC Interpretation

ROC-AUC measures the model's ability to rank churners above non-churners across decision thresholds. It is more informative than accuracy when churn classes are imbalanced.

![ROC curves](figures/roc_curves.png)

## 8. Feature Importance and Explainability

Top Random Forest churn drivers:

| Feature                  | Importance | Business Interpretation                                                                                              |
| ------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------- |
| LoginFrequency           | 0.136      | Digital engagement patterns can show disengagement or channel preference shifts before churn.                        |
| AvgAmountSpent           | 0.133      | Spending and transaction activity reflect relationship depth and declining banking usage.                            |
| TotalAmountSpent         | 0.101      | Spending and transaction activity reflect relationship depth and declining banking usage.                            |
| CustomerTenureDays       | 0.099      | Tenure helps distinguish newly acquired customers from established relationships needing different retention offers. |
| DaysSinceLastLogin       | 0.095      | Digital engagement patterns can show disengagement or channel preference shifts before churn.                        |
| DaysSinceLastTransaction | 0.089      | Spending and transaction activity reflect relationship depth and declining banking usage.                            |
| Age                      | 0.077      | Age helps tailor communication channels and product recommendations.                                                 |
| MaxAmountSpent           | 0.062      | Spending and transaction activity reflect relationship depth and declining banking usage.                            |

These are predictive importance signals, not proof of causality. They should guide retention diagnostics, segmentation, and further feature engineering rather than be interpreted as direct causes of churn.

![Random Forest feature importance](figures/random_forest_feature_importance.png)

### SHAP Explainability

SHAP values estimate how strongly each feature contributes to individual churn-risk predictions. The chart below summarizes the largest average impacts for the positive churn class.

| Feature                  | MeanAbsSHAP | Business Interpretation                                                                                              |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------- |
| CustomerTenureDays       | 0.034       | Tenure helps distinguish newly acquired customers from established relationships needing different retention offers. |
| LoginFrequency           | 0.023       | Digital engagement patterns can show disengagement or channel preference shifts before churn.                        |
| TotalAmountSpent         | 0.020       | Spending and transaction activity reflect relationship depth and declining banking usage.                            |
| AvgAmountSpent           | 0.016       | Spending and transaction activity reflect relationship depth and declining banking usage.                            |
| DaysSinceLastLogin       | 0.016       | Digital engagement patterns can show disengagement or channel preference shifts before churn.                        |
| DaysSinceLastTransaction | 0.015       | Spending and transaction activity reflect relationship depth and declining banking usage.                            |
| Age                      | 0.012       | Age helps tailor communication channels and product recommendations.                                                 |
| MaxAmountSpent           | 0.011       | Spending and transaction activity reflect relationship depth and declining banking usage.                            |

![SHAP summary](figures/shap_summary_random_forest.png)

## 9. Model Performance Improvement Suggestions

- Add richer behavioral features such as month-over-month transaction decline, salary-credit changes, product usage drops, complaint aging, and channel migration.
- Engineer trend features rather than relying only on current-state values; churn risk often appears as a change in behavior.
- Test probability-threshold optimization so retention teams can choose higher recall or higher precision based on campaign capacity.
- Compare calibrated probabilities using Platt scaling or isotonic calibration before integrating scores into customer-facing workflows.
- Monitor model drift, churn-rate drift, and segment-level performance after deployment.
- Retrain on a fixed schedule, such as quarterly, or sooner if churn rate, feature distributions, or campaign response patterns materially shift.

## 10. Business Utilisation Recommendations

- Score customers weekly or monthly and create risk bands such as high, medium, and low churn risk.
- Route high-risk customers to retention teams with the top contributing risk factors, not just a raw score.
- Prioritize unresolved complaints, declining engagement, low recent transaction activity, and high-risk segments for proactive outreach.
- Use churn probabilities to guide marketing campaigns, but cap campaign volume based on retention-team capacity.
- Track intervention outcomes so future model versions can learn which retention actions actually reduce churn.

## Deliverables Created

- Final model: `models/final_tuned_random_forest.joblib`
- Performance comparison table: `tables/model_performance_comparison.csv`
- Cross-validation table: `tables/cross_validation_summary.csv`
- Imbalance comparison table: `tables/imbalance_strategy_comparison.csv`
- Hyperparameter tuning results: `tables/random_forest_hyperparameter_results.csv`
- Operating threshold results: `tables/random_forest_threshold_analysis.csv`
- Feature importance table: `tables/random_forest_feature_importance.csv`
- Test-set churn risk scores: `tables/test_set_churn_risk_scores.csv`
- Top 50 high-risk customers: `tables/top_50_high_risk_customers.csv`
