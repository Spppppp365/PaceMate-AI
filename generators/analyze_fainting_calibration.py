import pandas as pd
from pathlib import Path
import joblib

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)


TARGET_COLUMN = "fainting_risk"

N_BINS = 10


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


def analyze_split(
    data,
    model,
    split_name,
):
    X = prepare_features(data)

    y = data[
        TARGET_COLUMN
    ]

    probabilities = model.predict_proba(
        X
    )[:, 1]

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    brier = brier_score_loss(
        y,
        probabilities,
    )

    fraction_positive, mean_predicted = (
        calibration_curve(
            y,
            probabilities,
            n_bins=N_BINS,
            strategy="uniform",
        )
    )

    print()
    print("=" * 70)
    print(
        f"CALIBRATION: {split_name}"
    )
    print("=" * 70)

    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Positive rate: {y.mean():.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC: {pr_auc:.4f}"
    )

    print(
        f"Brier score: {brier:.6f}"
    )

    print()
    print(
        "Calibration bins:"
    )

    print(
        f"{'Mean predicted':>18} "
        f"{'Observed rate':>18}"
    )

    rows = []

    for predicted, observed in zip(
        mean_predicted,
        fraction_positive,
    ):
        print(
            f"{predicted:18.4f} "
            f"{observed:18.4f}"
        )

        rows.append({
            "split": split_name,
            "mean_predicted": predicted,
            "observed_rate": observed,
            "absolute_error": abs(
                predicted - observed
            ),
        })

    return rows


def main():
    print("=" * 70)
    print("PaceMate-AI Fainting-Risk Calibration Analysis")
    print("=" * 70)

    model = load_model()

    validation_data = load_dataset(
        "validation_dataset.csv"
    )

    test_data = load_dataset(
        "test_dataset.csv"
    )

    results = []

    results.extend(
        analyze_split(
            validation_data,
            model,
            "Validation",
        )
    )

    results.extend(
        analyze_split(
            test_data,
            model,
            "Test",
        )
    )

    results_dataframe = pd.DataFrame(
        results
    )

    output_path = (
        get_project_root()
        / "results"
        / "fainting_calibration.csv"
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
    print("=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()