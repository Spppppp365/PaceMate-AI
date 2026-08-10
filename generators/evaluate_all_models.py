import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


TARGET_COLUMNS = [
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest",
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


def load_test_dataset():
    path = (
        get_project_root()
        / "data"
        / "test_dataset.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
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


def load_model(target_column):
    path = (
        get_project_root()
        / "models"
        / f"{target_column}_model.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


def evaluate_model(
    model,
    X_test,
    y_test,
):
    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": matrix,
    }


def save_results(results):
    output_path = (
        get_project_root()
        / "results"
        / "all_model_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_to_save = results.copy()

    results_to_save = results_to_save.drop(
        columns=["confusion_matrix"]
    )

    results_to_save.to_csv(
        output_path,
        index=False
    )

    return output_path


def main():
    print("=" * 60)
    print("PaceMate-AI Multi-Target Model Evaluation")
    print("=" * 60)

    print()
    print("Loading unseen test participants...")

    test_data = load_test_dataset()

    X_test = prepare_features(
        test_data
    )

    print(
        f"Test rows: {len(test_data)}"
    )

    print(
        f"Test participants: "
        f"{test_data['participant_id'].nunique()}"
    )

    print(
        f"Number of features: "
        f"{X_test.shape[1]}"
    )

    results = []

    for target_column in TARGET_COLUMNS:

        print()
        print("-" * 60)
        print(
            f"Evaluating: {target_column}"
        )
        print("-" * 60)

        model = load_model(
            target_column
        )

        y_test = test_data[
            target_column
        ]

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall:    {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score:  {metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC:   {metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:    {metrics['pr_auc']:.4f}"
        )

        print()
        print("Confusion Matrix:")
        print(
            metrics["confusion_matrix"]
        )

        results.append({
            "target": target_column,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "confusion_matrix": metrics[
                "confusion_matrix"
            ],
        })

    results_dataframe = pd.DataFrame(
        results
    )

    print()
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_dataframe[
            [
                "target",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].round(4).to_string(
            index=False
        )
    )

    output_path = save_results(
        results_dataframe
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 60)
    print("ALL MODELS EVALUATED")
    print("=" * 60)


if __name__ == "__main__":
    main()