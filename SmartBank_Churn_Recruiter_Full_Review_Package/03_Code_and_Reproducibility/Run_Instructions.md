# Run Instructions

These steps are for a technical reviewer who wants to reproduce the modeling workflow.

## 1. Install Python Packages

From this folder, install the required packages:

```bash
pip install -r requirements.txt
```

## 2. Check Data Path

The script expects the churn dataset to be available as:

`../02_Data/Advanced_Churn_Analysis_Dataset.csv`

This path already works inside this recruiter package.

## 3. Run The Modeling Script

```bash
python smartbank_churn_modeling.py
```

## 4. Expected Outputs

The workflow creates model metrics, feature importance, risk scores, charts, reports, and a saved model artifact. In this recruiter package, those outputs are already included in:

- `04_Results_Tables`
- `05_Visual_Evidence`
- `06_Model_Artifacts`

If the script is rerun, new outputs are written to:

- `08_Reproduced_Outputs`

## Reproducibility Notes

- Random seed: 42
- Target column: `ChurnStatus`
- Selected operating threshold: 0.370
- Final recommended model family: Random Forest
