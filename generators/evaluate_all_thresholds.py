import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

TARGETS = [
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest",
]

MODEL_FILES = {
    "flare_risk": "flare_risk_calibrated_model.joblib",
    "dizziness_risk": "dizziness_risk_calibrated_model.joblib",
    "fatigue_risk": "fatigue_risk_calibrated_model.joblib",
    "fainting_risk": "fainting_risk_calibrated_model.joblib",
    "need_to_hydrate": "need_to_hydrate_calibrated_model.joblib",
    "need_to_rest": "need_to_rest_calibrated_model.joblib",
}

THRESHOLDS = {
    "flare_risk": 0.55,
    "dizziness_risk": 0.25,
    "fatigue_risk": 0.30,
    "fainting_risk": 0.20,
    "need_to_hydrate": 0.35,
    "need_to_rest": 0.35,
}

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
    return Path(__file__).resolve().parent.parent


def prepare_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    return data.drop(columns=excluded)


def load_model(target):
    path = (
        get_project_root()
        / "models"
        / MODEL_FILES[target]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


def main():
    print("=" * 80)
    print("PaceMate-AI Multi-Target Threshold Test Evaluation")
    print("=" * 80)

    root = get_project_root()

    test_path = (
        root
        / "data"
        / "test_dataset.csv"
    )

    test_data = pd.read_csv(test_path)

    X_test = prepare_features(test_data)

    print()
    print(f"Test rows: {len(test_data)}")
    print(
        f"Test participants: "
        f"{test_data['participant_id'].nunique()}"
    )
    print(
        f"Number of features: "
        f"{X_test.shape[1]}"
    )

    results = []

    for target in TARGETS:
        print()
        print("=" * 80)
        print(f"Evaluating: {target}")
        print("=" * 80)

        model = load_model(target)

        y_test = test_data[target]

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        threshold = THRESHOLDS[target]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

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

        brier = brier_score_loss(
            y_test,
            probabilities,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        print(
            f"Threshold:    {threshold:.2f}"
        )
        print(
            f"Accuracy:     {accuracy:.4f}"
        )
        print(
            f"Precision:    {precision:.4f}"
        )
        print(
            f"Recall:       {recall:.4f}"
        )
        print(
            f"F1 Score:     {f1:.4f}"
        )
        print(
            f"ROC-AUC:      {roc_auc:.4f}"
        )
        print(
            f"PR-AUC:       {pr_auc:.4f}"
        )
        print(
            f"Brier Score:  {brier:.6f}"
        )
        print(
            f"Mean Predicted: {probabilities.mean():.4f}"
        )

        print()
        print("Confusion Matrix:")
        print(matrix)

        results.append({
            "target": target,
            "threshold": threshold,
            "positive_rate": y_test.mean(),
            "mean_predicted_probability": probabilities.mean(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "true_negatives": matrix[0, 0],
            "false_positives": matrix[0, 1],
            "false_negatives": matrix[1, 0],
            "true_positives": matrix[1, 1],
        })

    results_dataframe = pd.DataFrame(results)

    output_path = (
        root
        / "results"
        / "final_threshold_evaluation.csv"
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
    print("=" * 80)
    print("FINAL THRESHOLD EVALUATION")
    print("=" * 80)

    print(
        results_dataframe.round(4).to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 80)
    print("THRESHOLD TEST EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
