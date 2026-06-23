# Recruiter Review Guide

## Project Summary

SmartBank Customer Churn Prediction is an end-to-end machine learning project for identifying banking customers at risk of churn. The project includes dataset review, model comparison, imbalance handling, hyperparameter tuning, explainability, risk scoring, and retention recommendations.

## Best Files To Check First

1. `01_Final_Reports/SmartBank_Churn_Modeling_Report.docx`
2. `05_Visual_Evidence/random_forest_feature_importance.png`
3. `05_Visual_Evidence/shap_summary_random_forest.png`
4. `04_Results_Tables/model_performance_comparison.csv`
5. `04_Results_Tables/top_50_high_risk_customers.csv`
6. `03_Code_and_Reproducibility/smartbank_churn_modeling.py`

## Skills Demonstrated

- Exploratory data analysis for a banking churn problem.
- Classification modeling with Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
- Stratified train/test split and cross-validation.
- Handling class imbalance with class weights and SMOTE comparison.
- Hyperparameter tuning with recall as the business-oriented priority.
- Feature importance and SHAP-style model explanation.
- Churn-risk scoring and top-customer prioritization for retention teams.
- Clear business communication through reports, visuals, and recommendations.

## Key Evidence

- Dataset: 1,000 rows, 27 columns.
- Target: `ChurnStatus`.
- Churn rate: 20.4%.
- Recommended model family: Tuned Random Forest.
- Default threshold test result: Accuracy 0.670, Precision 0.143, Recall 0.122, F1 0.132, ROC-AUC 0.483.
- Recall-focused threshold: 0.370.
- Recall-focused result: Precision 0.188, Recall 0.634, F1 0.291.

## Honest Model Note

The report clearly states that this is a pilot model rather than a production decision engine. The current dataset has weak churn separability, so the project is strongest as evidence of a complete data science workflow, evaluation discipline, explainability, and business translation.

## Suggested Review Path

1. Read the Word report in `01_Final_Reports`.
2. Open the charts in `05_Visual_Evidence`.
3. Check the metrics and high-risk customer outputs in `04_Results_Tables`.
4. Review the modeling code and requirements in `03_Code_and_Reproducibility`.
5. Check the saved model and metadata in `06_Model_Artifacts`.
6. Review optional workflow context in `07_Project_Context` only if needed.
