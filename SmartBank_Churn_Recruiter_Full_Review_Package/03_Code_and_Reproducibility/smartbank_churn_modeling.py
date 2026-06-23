"""SmartBank customer churn modeling workflow.

This script builds a reproducible binary-classification workflow for the
provided cleaned churn dataset. It trains comparison models, handles class
imbalance, tunes a Random Forest, creates evaluation charts, and writes a
business-focused Markdown report.
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_DIR.parent
LOCAL_PACKAGES = PROJECT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    fbeta_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None

try:
    import shap
except Exception:  # pragma: no cover - optional dependency fallback
    shap = None


RANDOM_STATE = 42
DATASET_PATH = PACKAGE_DIR / "02_Data" / "Advanced_Churn_Analysis_Dataset.csv"
OUTPUT_DIR = PACKAGE_DIR / "08_Reproduced_Outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "SmartBank_Churn_Modeling_Report.md"


def ensure_dirs() -> None:
    for directory in [OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    target_col = "Churn" if "Churn" in df.columns else "ChurnStatus"
    if target_col not in df.columns:
        raise ValueError("Expected a binary churn target column named 'Churn' or 'ChurnStatus'.")

    return df, target_col


def describe_features(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    feature_cols = [col for col in df.columns if col != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    binary_cols = [
        col
        for col in numeric_cols
        if set(pd.Series(df[col].dropna().unique()).astype(float)).issubset({0.0, 1.0})
    ]
    continuous_cols = [col for col in numeric_cols if col not in binary_cols]

    return {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "binary_cols": binary_cols,
        "continuous_cols": continuous_cols,
        "missing_total": int(df.isna().sum().sum()),
    }


def prepare_data(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    id_col = "CustomerID" if "CustomerID" in df.columns else None
    y = df[target_col].astype(int)
    drop_cols = [target_col]
    if id_col:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols)
    customer_ids = df[id_col] if id_col else pd.Series(range(1, len(df) + 1), name="CustomerID")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_test": ids_test,
        "id_col": id_col,
    }


def build_models(scale_pos_weight: float) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    return models


def predict_positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / max(scores.max() - scores.min(), 1e-9)
    raise ValueError("Model does not support probability or decision score predictions.")


def metric_row(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_prob = predict_positive_probability(model, X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }


def metric_row_at_threshold(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    y_prob = predict_positive_probability(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "F2 Score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }


def optimize_operating_threshold(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    rows = []
    for threshold in np.round(np.arange(0.20, 0.61, 0.01), 2):
        y_pred = (oof_prob >= threshold).astype(int)
        rows.append(
            {
                "Threshold": float(threshold),
                "Accuracy": accuracy_score(y_train, y_pred),
                "Precision": precision_score(y_train, y_pred, zero_division=0),
                "Recall": recall_score(y_train, y_pred, zero_division=0),
                "F1 Score": f1_score(y_train, y_pred, zero_division=0),
                "F2 Score": fbeta_score(y_train, y_pred, beta=2, zero_division=0),
                "Flagged Share": float(np.mean(y_pred)),
            }
        )

    threshold_df = pd.DataFrame(rows)
    base_churn_rate = float(y_train.mean())
    eligible = threshold_df[
        (threshold_df["Recall"] >= 0.60)
        & (threshold_df["Precision"] >= base_churn_rate)
        & (threshold_df["Flagged Share"] <= 0.60)
    ]
    if eligible.empty:
        eligible = threshold_df[
            (threshold_df["Recall"] >= 0.50) & (threshold_df["Flagged Share"] <= 0.60)
        ]
    if eligible.empty:
        eligible = threshold_df

    threshold_df["Selected"] = False
    selected_idx = eligible.sort_values(
        ["F1 Score", "F2 Score", "Precision"],
        ascending=False,
    ).index[0]
    threshold_df.loc[selected_idx, "Selected"] = True
    return threshold_df


def run_cross_validation(models: dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_models(
    models: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fitted_models: dict[str, Any] = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        rows.append(metric_row(name, model, X_test, y_test))
    return fitted_models, pd.DataFrame(rows)


def compare_imbalance_methods(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    strategies = {
        "RF - No Balancing": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "RF - Class Weight": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "RF - SMOTE": ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    rows = []
    for name, model in strategies.items():
        model.fit(X_train, y_train)
        rows.append(metric_row(name, model, X_test, y_test))
    return pd.DataFrame(rows)


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    params = {
        "n_estimators": [120, 200, 300],
        "max_depth": [None, 5, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    scoring = {
        "recall": "recall",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "accuracy": "accuracy",
    }
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=24,
        scoring=scoring,
        refit="recall",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def plot_churn_distribution(y: pd.Series, class_counts: pd.Series) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette=["#4C78A8", "#F58518"])
    ax.set_title("Churn Class Distribution")
    ax.set_xlabel("Churn Status (0 = retained, 1 = churned)")
    ax.set_ylabel("Customer Count")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "churn_distribution.png", dpi=180)
    plt.close()


def plot_cv_comparison(cv_df: pd.DataFrame) -> None:
    plot_df = cv_df[["Model", "recall_mean", "roc_auc_mean", "f1_mean"]].melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Mean Score",
    )
    plot_df["Metric"] = plot_df["Metric"].str.replace("_mean", "", regex=False).str.upper()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="Model", y="Mean Score", hue="Metric")
    ax.set_title("Stratified 5-Fold Cross-Validation Comparison")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cross_validation_comparison.png", dpi=180)
    plt.close()


def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Retained", "Predicted Churn"],
        yticklabels=["Actual Retained", "Actual Churn"],
    )
    ax.set_title("Tuned Random Forest Confusion Matrix")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix_random_forest.png", dpi=180)
    plt.close()


def plot_roc_curves(models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = predict_positive_probability(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#555555", lw=1.5, linestyle="--", label="Random guess")
    plt.title("ROC Curves by Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "roc_curves.png", dpi=180)
    plt.close()


def plot_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    importances = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top = importances.head(15)
    plt.figure(figsize=(9, 7))
    ax = sns.barplot(data=top, y="Feature", x="Importance", palette="viridis")
    ax.set_title("Tuned Random Forest Feature Importance")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=180)
    plt.close()

    return importances


def get_positive_class_shap_values(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def plot_shap_summary(model: RandomForestClassifier, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame | None:
    if shap is None:
        return None

    try:
        sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
        background = X_train.sample(n=min(250, len(X_train)), random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        shap_values = get_positive_class_shap_values(explainer.shap_values(sample))
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"Feature": sample.columns, "MeanAbsSHAP": mean_abs}).sort_values(
            "MeanAbsSHAP",
            ascending=False,
        )

        top = shap_df.head(15)
        plt.figure(figsize=(9, 7))
        ax = sns.barplot(data=top, y="Feature", x="MeanAbsSHAP", palette="mako")
        ax.set_title("SHAP Summary - Tuned Random Forest")
        ax.set_xlabel("Mean absolute SHAP value for churn risk")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "shap_summary_random_forest.png", dpi=180)
        plt.close()

        return shap_df
    except Exception as exc:
        (OUTPUT_DIR / "shap_warning.txt").write_text(f"SHAP plot could not be generated: {exc}", encoding="utf-8")
        return None


def save_high_risk_customers(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ids_test: pd.Series,
    operating_threshold: float,
) -> pd.DataFrame:
    y_prob = predict_positive_probability(model, X_test)
    y_pred = (y_prob >= operating_threshold).astype(int)
    risk_df = pd.DataFrame(
        {
            "CustomerID": ids_test.values,
            "ActualChurn": y_test.values,
            "PredictedChurn": y_pred,
            "ChurnProbability": y_prob,
            "OperatingThreshold": operating_threshold,
        }
    ).sort_values("ChurnProbability", ascending=False)
    risk_df.to_csv(TABLE_DIR / "test_set_churn_risk_scores.csv", index=False)
    risk_df.head(50).to_csv(TABLE_DIR / "top_50_high_risk_customers.csv", index=False)
    return risk_df


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def df_to_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    display = df.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda x: f"{x:.{decimals}f}" if not float(x).is_integer() else f"{int(x)}")

    headers = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def business_meaning(feature: str) -> str:
    feature_l = feature.lower()
    if "complaint" in feature_l or "resolution" in feature_l:
        return "Service friction is a direct retention signal; unresolved issues should trigger urgent recovery outreach."
    if "login" in feature_l or "mobile" in feature_l or "website" in feature_l:
        return "Digital engagement patterns can show disengagement or channel preference shifts before churn."
    if "transaction" in feature_l or "amount" in feature_l or "spent" in feature_l:
        return "Spending and transaction activity reflect relationship depth and declining banking usage."
    if "tenure" in feature_l:
        return "Tenure helps distinguish newly acquired customers from established relationships needing different retention offers."
    if "income" in feature_l:
        return "Income segment can support differentiated offers and retention economics."
    if "age" in feature_l:
        return "Age helps tailor communication channels and product recommendations."
    if "category" in feature_l or "prefers" in feature_l:
        return "Product/category preferences help personalize retention campaigns."
    return "This feature contributes to churn-risk ranking and should be monitored with segment-level diagnostics."


def write_report(
    df: pd.DataFrame,
    target_col: str,
    feature_info: dict[str, Any],
    class_counts: pd.Series,
    class_rates: pd.Series,
    cv_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    imbalance_df: pd.DataFrame,
    search: RandomizedSearchCV,
    threshold_df: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    shap_importance: pd.DataFrame | None,
    risk_df: pd.DataFrame,
) -> None:
    minority_rate = float(class_rates.min())
    imbalance_label = "meaningful" if minority_rate < 0.40 else "mild"
    final_row = metrics_df.loc[metrics_df["Model"].eq("Tuned Random Forest")].iloc[0]
    threshold_row = threshold_metrics.iloc[0]
    selected_threshold = float(threshold_row["Threshold"])
    cm_values = final_row[
        ["True Negatives", "False Positives", "False Negatives", "True Positives"]
    ].astype(int)
    threshold_cm_values = threshold_row[
        ["True Negatives", "False Positives", "False Negatives", "True Positives"]
    ].astype(int)
    best_model_by_recall = metrics_df.sort_values(["Recall", "ROC-AUC"], ascending=False).iloc[0]
    best_model_by_auc = metrics_df.sort_values(["ROC-AUC", "Recall"], ascending=False).iloc[0]

    top_features = feature_importance.head(8).copy()
    top_features["Business Interpretation"] = top_features["Feature"].map(business_meaning)
    top_features_report = top_features[["Feature", "Importance", "Business Interpretation"]]

    shap_section = "SHAP was attempted, but a compatible SHAP output was not available in this run."
    if shap_importance is not None:
        shap_top = shap_importance.head(8).copy()
        shap_top["Business Interpretation"] = shap_top["Feature"].map(business_meaning)
        shap_section = df_to_markdown(shap_top[["Feature", "MeanAbsSHAP", "Business Interpretation"]])

    hyperparameter_summary = pd.DataFrame(search.cv_results_).sort_values("rank_test_recall").head(10)
    hyperparameter_summary = hyperparameter_summary[
        [
            "rank_test_recall",
            "mean_test_recall",
            "mean_test_roc_auc",
            "mean_test_f1",
            "mean_test_precision",
            "param_n_estimators",
            "param_max_depth",
            "param_min_samples_split",
            "param_min_samples_leaf",
            "param_max_features",
            "param_class_weight",
        ]
    ]

    model_selection = pd.DataFrame(
        [
            {
                "Algorithm": "Logistic Regression",
                "Strengths": "Simple, transparent coefficients, fast baseline.",
                "Limitations": "Linear boundary; may miss non-linear churn interactions.",
                "Business Fit": "Useful benchmark and explainable scorecard-style model.",
            },
            {
                "Algorithm": "Decision Tree",
                "Strengths": "Easy to explain with rule paths.",
                "Limitations": "Can overfit and be unstable without pruning.",
                "Business Fit": "Good for workshops, weaker as sole enterprise model.",
            },
            {
                "Algorithm": "Random Forest",
                "Strengths": "Strong accuracy, handles non-linearity, robust feature importance.",
                "Limitations": "Less transparent than one tree, but explainable with importance/SHAP.",
                "Business Fit": "Recommended balance of performance, reliability, and stakeholder usability.",
            },
            {
                "Algorithm": "Gradient Boosting",
                "Strengths": "High predictive accuracy on tabular data.",
                "Limitations": "More tuning-sensitive and can be less intuitive.",
                "Business Fit": "Strong challenger model when maximum lift is required.",
            },
            {
                "Algorithm": "XGBoost",
                "Strengths": "Advanced boosted trees, regularization, high tuning flexibility.",
                "Limitations": "More operational complexity and explanation burden.",
                "Business Fit": "Excellent challenger; use when lift outweighs simplicity.",
            },
            {
                "Algorithm": "Neural Networks",
                "Strengths": "Can learn complex patterns in very large datasets.",
                "Limitations": "Less interpretable and needs more data/monitoring.",
                "Business Fit": "Not preferred here because stakeholder explainability is important.",
            },
        ]
    )

    report = f"""# SmartBank Customer Churn Prediction Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Executive Summary

This analysis used the cleaned dataset `{DATASET_PATH.name}` to build and evaluate binary churn prediction models for SmartBank / Lloyds Banking Group. The file uses `{target_col}` as the churn target; this is treated as the required binary churn indicator.

The recommended model family is **Random Forest** because it balances tabular predictive modeling, feature-importance explainability, class-weight support, and practical scalability for customer-retention workflows. However, the supplied dataset shows weak churn separability: the tuned Random Forest default threshold achieved **Accuracy {fmt(final_row["Accuracy"])}**, **Precision {fmt(final_row["Precision"])}**, **Recall {fmt(final_row["Recall"])}**, **F1 {fmt(final_row["F1 Score"])}**, and **ROC-AUC {fmt(final_row["ROC-AUC"])}** on the held-out test set. This should be treated as a pilot model, not an automated production decision engine, until richer behavioral features improve rank-ordering quality.

Because false negatives are costly in churn prevention, a recall-focused operating threshold of **{fmt(selected_threshold)}** was selected from out-of-fold training predictions with a campaign-capacity guardrail. At that action threshold the held-out test metrics were **Precision {fmt(threshold_row["Precision"])}**, **Recall {fmt(threshold_row["Recall"])}**, and **F1 {fmt(threshold_row["F1 Score"])}**. This threshold is useful for screening, but the low precision means retention teams should combine the score with business rules and segment-level prioritization.

## 1. Dataset Review and Problem Understanding

- Dataset rows: **{df.shape[0]}**
- Dataset columns: **{df.shape[1]}**
- Target variable used: **{target_col}**
- Problem type: **Binary classification**
- Missing values detected: **{feature_info["missing_total"]}**
- Feature columns used for modeling: **{len(feature_info["feature_cols"]) - (1 if "CustomerID" in feature_info["feature_cols"] else 0)}**
- Numerical features: **{len(feature_info["numeric_cols"])}**
- Already encoded binary indicator features: **{len(feature_info["binary_cols"])}**
- Non-numeric categorical features requiring encoding: **{len(feature_info["categorical_cols"])}**

### Churn Class Distribution

{df_to_markdown(pd.DataFrame({"Class": class_counts.index, "Customer Count": class_counts.values, "Share": class_rates.values}))}

The churn distribution shows a **{imbalance_label} class imbalance**. Stratified train-test splitting and stratified cross-validation were used so each evaluation fold preserves the churn/non-churn ratio. This matters because ordinary random splitting can accidentally under-represent churners, causing unstable recall and misleading accuracy.

![Churn distribution](figures/churn_distribution.png)

## 2. Feature Preparation

The workflow separated features `X` from the target `y`, removed `CustomerID` from model training because it is an identifier rather than a behavioral predictor, and used an 80/20 train-test split with stratification.

All model features were already numeric or binary encoded, so no additional one-hot encoding was required. Feature scaling was applied in the Logistic Regression pipeline because linear models are sensitive to feature scale. Tree-based models such as Random Forest, Gradient Boosting, and XGBoost do not require scaling because they split on feature thresholds.

## 3. Algorithm Selection

{df_to_markdown(model_selection, decimals=3)}

**Final selection:** Random Forest is the best fit for this banking churn use case because it captures non-linear customer behavior, supports class weighting, and gives feature-importance outputs that can be translated into retention actions. Given the current test performance, it should be used as a transparent pilot model while the data science team improves the signal with stronger behavioral and historical features.

## 4. Model Building and Cross-Validation

The following models were trained: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost. Logistic Regression served as the baseline. Random Forest was used as the final model family and was tuned.

Stratified 5-fold cross-validation was used to check model stability:

{df_to_markdown(cv_df[["Model", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]])}

![Cross-validation comparison](figures/cross_validation_comparison.png)

## 5. Class Imbalance Handling

The imbalance strategy compared three Random Forest approaches: no balancing, class weighting, and SMOTE oversampling applied only within the training data. This avoids leaking synthetic examples into the test set.

{df_to_markdown(imbalance_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "False Negatives", "False Positives"]])}

Class weighting is operationally simple and keeps the original customer population intact, while SMOTE can improve minority-class learning when churners are under-represented. For banking deployment, class weighting is often the first practical option because it is transparent, reproducible, and easy to monitor.

## 6. Hyperparameter Tuning

RandomizedSearchCV was used with stratified 5-fold cross-validation. The search optimized Random Forest parameters including number of trees, maximum depth, minimum split size, minimum leaf size, feature sampling, and class weighting. The refit metric was **recall** because the business priority is identifying as many true churn-risk customers as possible.

Best Random Forest parameters:

```json
{json.dumps(search.best_params_, indent=2)}
```

Top tuning results by recall:

{df_to_markdown(hyperparameter_summary)}

## 7. Final Model Evaluation

{df_to_markdown(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "False Negatives", "False Positives"]])}

- Best model by recall: **{best_model_by_recall["Model"]}** with recall **{fmt(best_model_by_recall["Recall"])}**
- Best model by ROC-AUC: **{best_model_by_auc["Model"]}** with ROC-AUC **{fmt(best_model_by_auc["ROC-AUC"])}**
- Recommended model: **Tuned Random Forest**

### Recall-Focused Operating Threshold

The standard 0.50 threshold is not always appropriate for churn prevention because it can miss too many churners. A threshold was selected using out-of-fold training predictions, prioritizing recall while keeping the flagged population near a manageable campaign size where possible.

Selected operating threshold: **{fmt(selected_threshold)}**

Held-out performance at selected threshold:

{df_to_markdown(threshold_metrics[["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1 Score", "F2 Score", "ROC-AUC", "False Negatives", "False Positives"]])}

This operating threshold reduces false negatives from **{cm_values["False Negatives"]}** to **{threshold_cm_values["False Negatives"]}**, but increases false positives from **{cm_values["False Positives"]}** to **{threshold_cm_values["False Positives"]}**. That trade-off is acceptable only if campaign capacity and intervention cost can support a broad screening list.

Top candidate thresholds from cross-validated training predictions:

{df_to_markdown(threshold_df.sort_values(["Selected", "F1 Score"], ascending=False).head(8))}

### Confusion Matrix Interpretation

For the Tuned Random Forest at the default 0.50 threshold:

- True negatives: **{cm_values["True Negatives"]}** customers correctly predicted as retained.
- False positives: **{cm_values["False Positives"]}** customers flagged as churn-risk who did not churn. These may receive retention outreach unnecessarily.
- False negatives: **{cm_values["False Negatives"]}** churners missed by the model. These are the highest business concern because retention teams may not intervene.
- True positives: **{cm_values["True Positives"]}** churners correctly identified for proactive retention.

![Confusion matrix](figures/confusion_matrix_random_forest.png)

### ROC-AUC Interpretation

ROC-AUC measures the model's ability to rank churners above non-churners across decision thresholds. It is more informative than accuracy when churn classes are imbalanced.

![ROC curves](figures/roc_curves.png)

## 8. Feature Importance and Explainability

Top Random Forest churn drivers:

{df_to_markdown(top_features_report)}

These are predictive importance signals, not proof of causality. They should guide retention diagnostics, segmentation, and further feature engineering rather than be interpreted as direct causes of churn.

![Random Forest feature importance](figures/random_forest_feature_importance.png)

### SHAP Explainability

SHAP values estimate how strongly each feature contributes to individual churn-risk predictions. The chart below summarizes the largest average impacts for the positive churn class.

{shap_section}

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
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", palette="deep")

    df, target_col = load_dataset(DATASET_PATH)
    feature_info = describe_features(df, target_col)
    prepared = prepare_data(df, target_col)
    X = prepared["X"]
    y = prepared["y"]
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    class_counts = y.value_counts().sort_index()
    class_rates = y.value_counts(normalize=True).sort_index()
    neg_count = int(class_counts.get(0, 0))
    pos_count = int(class_counts.get(1, 1))
    scale_pos_weight = neg_count / max(pos_count, 1)

    plot_churn_distribution(y, class_counts)

    models = build_models(scale_pos_weight=scale_pos_weight)
    cv_df = run_cross_validation(models, X_train, y_train)
    fitted_models, metrics_df = evaluate_models(models, X_train, X_test, y_train, y_test)

    imbalance_df = compare_imbalance_methods(X_train, X_test, y_train, y_test)
    search = tune_random_forest(X_train, y_train)
    tuned_rf = search.best_estimator_
    fitted_models["Tuned Random Forest"] = tuned_rf

    tuned_cv = run_cross_validation({"Tuned Random Forest": tuned_rf}, X_train, y_train)
    cv_df = pd.concat([cv_df, tuned_cv], ignore_index=True)

    tuned_metrics = pd.DataFrame([metric_row("Tuned Random Forest", tuned_rf, X_test, y_test)])
    metrics_df = pd.concat([metrics_df, tuned_metrics], ignore_index=True)

    threshold_df = optimize_operating_threshold(tuned_rf, X_train, y_train)
    selected_threshold = float(threshold_df.loc[threshold_df["Selected"], "Threshold"].iloc[0])
    threshold_metrics = pd.DataFrame(
        [
            metric_row_at_threshold(
                "Tuned Random Forest - Operating Threshold",
                tuned_rf,
                X_test,
                y_test,
                selected_threshold,
            )
        ]
    )

    cv_df.to_csv(TABLE_DIR / "cross_validation_summary.csv", index=False)
    metrics_df.to_csv(TABLE_DIR / "model_performance_comparison.csv", index=False)
    imbalance_df.to_csv(TABLE_DIR / "imbalance_strategy_comparison.csv", index=False)
    threshold_df.to_csv(TABLE_DIR / "random_forest_threshold_analysis.csv", index=False)
    threshold_metrics.to_csv(TABLE_DIR / "random_forest_operating_threshold_metrics.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(TABLE_DIR / "random_forest_hyperparameter_results.csv", index=False)

    plot_cv_comparison(cv_df)
    plot_confusion_matrix(tuned_rf, X_test, y_test)
    plot_roc_curves(fitted_models, X_test, y_test)
    feature_importance = plot_feature_importance(tuned_rf, list(X.columns))
    feature_importance.to_csv(TABLE_DIR / "random_forest_feature_importance.csv", index=False)

    shap_importance = plot_shap_summary(tuned_rf, X_train, X_test)
    if shap_importance is not None:
        shap_importance.to_csv(TABLE_DIR / "random_forest_shap_importance.csv", index=False)

    risk_df = save_high_risk_customers(tuned_rf, X_test, y_test, prepared["ids_test"], selected_threshold)
    dump(tuned_rf, MODEL_DIR / "final_tuned_random_forest.joblib")

    metadata = {
        "dataset_path": str(DATASET_PATH),
        "target_column": target_col,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_rates": {str(k): float(v) for k, v in class_rates.items()},
        "best_random_forest_params": search.best_params_,
        "selected_operating_threshold": selected_threshold,
        "random_state": RANDOM_STATE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUTPUT_DIR / "modeling_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(
        df=df,
        target_col=target_col,
        feature_info=feature_info,
        class_counts=class_counts,
        class_rates=class_rates,
        cv_df=cv_df,
        metrics_df=metrics_df,
        imbalance_df=imbalance_df,
        search=search,
        threshold_df=threshold_df,
        threshold_metrics=threshold_metrics,
        feature_importance=feature_importance,
        shap_importance=shap_importance,
        risk_df=risk_df,
    )

    print("SmartBank churn modeling complete.")
    print(f"Report: {REPORT_PATH}")
    print(f"Final model: {MODEL_DIR / 'final_tuned_random_forest.joblib'}")
    print(f"Performance table: {TABLE_DIR / 'model_performance_comparison.csv'}")


if __name__ == "__main__":
    main()
