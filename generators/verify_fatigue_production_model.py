import pandas as pd
from pathlib import Path
import joblib


TARGET_COLUMN = "fatigue_risk"

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


def main():
    print("=" * 70)
    print("PaceMate-AI Production Fatigue-Risk Model Verification")
    print("=" * 70)

    project_root = get_project_root()

    model_path = (
        project_root
        / "models"
        / "fatigue_risk_calibrated_model.joblib"
    )

    test_path = (
        project_root
        / "data"
        / "test_dataset.csv"
    )

    print()
    print("Loading production model...")

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    test_data = pd.read_csv(test_path)

    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in test_data.columns
    ]

    X_test = test_data.drop(
        columns=excluded
    )

    y_test = test_data[TARGET_COLUMN]

    print()
    print(f"Test rows: {len(test_data)}")
    print(f"Number of features: {X_test.shape[1]}")
    print(f"Positive cases: {y_test.sum()}")
    print(f"Positive rate: {y_test.mean():.4f}")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print()
    print("Probability verification:")
    print(
        f"Minimum probability: {probabilities.min():.6f}"
    )
    print(
        f"Maximum probability: {probabilities.max():.6f}"
    )
    print(
        f"Mean probability:    {probabilities.mean():.6f}"
    )

    print()
    print("Sample predictions:")

    for index in range(min(10, len(test_data))):
        print(
            f"Test row {index + 1:2d}: "
            f"predicted={probabilities[index]:.4f} "
            f"actual={y_test.iloc[index]}"
        )

    print()
    print("=" * 70)
    print("FATIGUE PRODUCTION MODEL VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
