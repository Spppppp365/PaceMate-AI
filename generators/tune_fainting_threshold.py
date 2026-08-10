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

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]


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


def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    true_negatives = matrix[0, 0]
    false_positives = matrix[0, 1]
    false_negatives = matrix[1, 0]
    true_positives = matrix[1, 1]

    predicted_positive = (
        false_positives
        + true_positives
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "predicted_positive": predicted_positive,
    }


def main():
    print("=" * 70)
    print("PaceMate-AI Fainting-Risk Threshold Analysis")
    print("=" * 70)

    validation_data = load_dataset(
        "validation_dataset.csv"
    )

    X_validation = prepare_features(
        validation_data
    )

    y_validation = validation_data[
        TARGET_COLUMN
    ]

    model = load_model()

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    print()
    print(
        f"Validation rows: "
        f"{len(validation_data)}"
    )

    print(
        f"Positive cases: "
        f"{int(y_validation.sum())}"
    )

    print(
        f"Positive rate: "
        f"{y_validation.mean():.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{pr_auc:.4f}"
    )

    results = []

    for threshold in THRESHOLDS:

        metrics = evaluate_threshold(
            y_validation,
            probabilities,
            threshold,
        )

        results.append(
            metrics
        )

    results_dataframe = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    print(
        results_dataframe[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positives",
                "false_negatives",
                "predicted_positive",
            ]
        ].round(4).to_string(
            index=False
        )
    )

    best_f1 = results_dataframe.loc[
        results_dataframe["f1"].idxmax()
    ]

    print()
    print("=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"False positives: "
        f"{int(best_f1['false_positives'])}"
    )

    print(
        f"False negatives: "
        f"{int(best_f1['false_negatives'])}"
    )

    output_path = (
        get_project_root()
        / "results"
        / "fainting_threshold_analysis.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()