import pandas as pd
from pathlib import Path
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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
    "flare_risk": "flare_risk_model.joblib",
    "dizziness_risk": "dizziness_risk_calibrated_model.joblib",
    "fatigue_risk": "fatigue_risk_calibrated_model.joblib",
    "fainting_risk": "fainting_risk_calibrated_model.joblib",
    "need_to_hydrate": "need_to_hydrate_calibrated_model.joblib",
    "need_to_rest": "need_to_rest_calibrated_model.joblib",
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
    0.85,
    0.90,
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
    print("PaceMate-AI Multi-Target Threshold Tuning")
    print("=" * 80)

    root = get_project_root()

    validation_path = (
        root
        / "data"
        / "validation_dataset.csv"
    )

    validation_data = pd.read_csv(
        validation_path
    )

    X_validation = prepare_features(
        validation_data
    )

    print()
    print(
        f"Validation rows: {len(validation_data)}"
    )

    print(
        f"Validation participants: "
        f"{validation_data['participant_id'].nunique()}"
    )

    print(
        f"Number of features: "
        f"{X_validation.shape[1]}"
    )

    all_results = []
    best_results = []

    for target in TARGETS:
        print()
        print("=" * 80)
        print(f"Target: {target}")
        print("=" * 80)

        model = load_model(target)

        y_validation = validation_data[target]

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

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC:  {pr_auc:.4f}"
        )

        best_f1 = -1
        best_row = None

        for threshold in THRESHOLDS:
            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            row = {
                "target": target,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

            all_results.append(row)

            if f1 > best_f1:
                best_f1 = f1
                best_row = row

        best_results.append(best_row)

        print()
        print(
            f"Best threshold: "
            f"{best_row['threshold']:.2f}"
        )

        print(
            f"Precision:      "
            f"{best_row['precision']:.4f}"
        )

        print(
            f"Recall:         "
            f"{best_row['recall']:.4f}"
        )

        print(
            f"F1 Score:       "
            f"{best_row['f1']:.4f}"
        )

    all_results_df = pd.DataFrame(
        all_results
    )

    best_results_df = pd.DataFrame(
        best_results
    )

    results_dir = (
        root
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_path = (
        results_dir
        / "all_threshold_results.csv"
    )

    best_path = (
        results_dir
        / "best_thresholds.csv"
    )

    all_results_df.to_csv(
        all_path,
        index=False,
    )

    best_results_df.to_csv(
        best_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("BEST THRESHOLDS")
    print("=" * 80)

    print(
        best_results_df.round(4).to_string(
            index=False
        )
    )

    print()
    print(
        f"All results saved to: {all_path}"
    )

    print(
        f"Best thresholds saved to: {best_path}"
    )

    print()
    print("=" * 80)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()