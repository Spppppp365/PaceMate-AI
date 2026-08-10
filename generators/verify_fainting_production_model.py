import pandas as pd
from pathlib import Path
import joblib


TARGET_COLUMN = "fainting_risk"

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


def main():
    print("=" * 70)
    print("PaceMate-AI Production Fainting-Risk Model Verification")
    print("=" * 70)

    project_root = get_project_root()

    model_path = (
        project_root
        / "models"
        / "fainting_risk_calibrated_model.joblib"
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

    X_test = prepare_features(test_data)
    y_test = test_data[TARGET_COLUMN]

    print()
    print(f"Test rows: {len(test_data)}")
    print(f"Number of features: {X_test.shape[1]}")

    probabilities = model.predict_proba(X_test)[:, 1]

    print()
    print("Probability verification:")
    print(f"Minimum probability: {probabilities.min():.6f}")
    print(f"Maximum probability: {probabilities.max():.6f}")
    print(f"Mean probability:    {probabilities.mean():.6f}")

    assert len(probabilities) == len(test_data)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0

    print()
    print("Sample predictions:")

    for index in range(10):
        print(
            f"Test row {index + 1:2d}: "
            f"predicted={probabilities[index]:.4f} "
            f"actual={y_test.iloc[index]}"
        )

    print()
    print("=" * 70)
    print("PRODUCTION MODEL VERIFICATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
