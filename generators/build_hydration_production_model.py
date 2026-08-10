import pandas as pd
from pathlib import Path
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


TARGET_COLUMN = "need_to_hydrate"

RANDOM_SEED = 42

N_ESTIMATORS = 300
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 5

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
    print("PaceMate-AI Production Hydration-Risk Model")
    print("=" * 70)

    project_root = get_project_root()

    data_path = (
        project_root
        / "data"
        / "train_dataset.csv"
    )

    data = pd.read_csv(data_path)

    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    X = data.drop(columns=excluded)
    y = data[TARGET_COLUMN]

    print()
    print(f"Training rows: {len(data)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Positive cases: {y.sum()}")
    print(f"Positive rate: {y.mean():.4f}")

    print()
    print(
        "Training sigmoid-calibrated "
        "Random Forest..."
    )

    base_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    model.fit(X, y)

    output_path = (
        project_root
        / "models"
        / "need_to_hydrate_calibrated_model.joblib"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
        compress=3,
    )

    print()
    print("Saved production model:")
    print(output_path)

    print()
    print("=" * 70)
    print("PRODUCTION MODEL BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
