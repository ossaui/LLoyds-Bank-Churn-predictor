# File Index

## 01_Final_Reports

- `Customer_Churn_Analysis_Report.docx` - original polished churn analysis report.
- `SmartBank_Churn_Modeling_Report.docx` - polished modeling report for review.
- `SmartBank_Churn_Modeling_Report.md` - Markdown version of the modeling report.
- `SmartBank_Churn_Modeling_Report_from_outputs.md` - report copy generated in the outputs folder.

## 02_Data

- `Advanced_Churn_Analysis_Dataset.csv` - main modeling dataset.
- `Customer_Churn_Data_Large.xlsx` - Excel dataset file.
- `Data_Notes.md` - summary of dataset structure and columns.

## 03_Code_and_Reproducibility

- `smartbank_churn_modeling.py` - main modeling workflow script.
- `requirements.txt` - Python packages needed to run the script.
- `data_modeling_notes.txt` - original modeling notes.
- `Run_Instructions.md` - short reproducibility guide.

## 04_Results_Tables

- `cross_validation_summary.csv` - cross-validation metrics by model.
- `imbalance_strategy_comparison.csv` - comparison of imbalance handling methods.
- `model_performance_comparison.csv` - held-out test metrics by model.
- `random_forest_feature_importance.csv` - Random Forest feature importance.
- `random_forest_hyperparameter_results.csv` - tuning search results.
- `random_forest_operating_threshold_metrics.csv` - final threshold metrics.
- `random_forest_shap_importance.csv` - SHAP-style importance table.
- `random_forest_threshold_analysis.csv` - threshold candidates and trade-offs.
- `test_set_churn_risk_scores.csv` - churn-risk scores for test customers.
- `top_50_high_risk_customers.csv` - prioritized retention list.

## 05_Visual_Evidence

- `churn_distribution.png` - target class balance.
- `confusion_matrix_random_forest.png` - Random Forest confusion matrix.
- `cross_validation_comparison.png` - model comparison visual.
- `random_forest_feature_importance.png` - top model features.
- `roc_curves.png` - ROC curves across models.
- `shap_summary_random_forest.png` - SHAP-style explanation chart.

## 06_Model_Artifacts

- `final_tuned_random_forest.joblib` - saved model artifact from outputs.
- `final_tuned_random_forest_modeling_copy.joblib` - saved model copy from modeling folder.
- `modeling_metadata.json` - target, class balance, selected threshold, random seed, and tuning metadata.

## 07_Project_Context

- `Exclusions_And_Notes.md` - notes about omitted machine-local dependency folders.
- `skills/effective-machine-learning` - optional workflow context included from the original project.
