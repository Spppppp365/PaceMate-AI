import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


TARGET_COLUMN = "fainting_risk"
THRESHOLD = 0.75

EXCLUDED_COLUMNS = [
    "participant_id",
    "day",
    "dizziness",
    "fatigue",
    "brain_fog",
    "symptom_severity",
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest",
]


def get_project_root():
    return (
        Path(__file__)
        .resolve()
        .parent.parent
    )


def load_dataset(filename):
    path = (
        get_project_root()
        / "data"
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def prepare_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    return data.drop(
        columns=excluded
    )


def load_model():
    path = (
        get_project_root()
        / "models"
        / "fainting_risk_model.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


def main():
    print("=" * 70)
    print("PaceMate-AI Fainting-Risk Test Threshold Evaluation")
    print("=" * 70)

    test_data = load_dataset(
        "test_dataset.csv"
    )

    X_test = prepare_features(
        test_data
    )

    y_test = test_data[
        TARGET_COLUMN
    ]

    model = load_model()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print(
        f"Test rows: {len(test_data)}"
    )

    print(
        f"Test participants: "
        f"{test_data['participant_id'].nunique()}"
    )

    print(
        f"Positive cases: "
        f"{int(y_test.sum())}"
    )

    print(
        f"Positive rate: "
        f"{y_test.mean():.4f}"
    )

    print()
    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(matrix)

    print()
    print("=" * 70)
    print("TEST THRESHOLD EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()