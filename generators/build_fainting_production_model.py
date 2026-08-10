import pandas as pd
from pathlib import Path
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


TARGET_COLUMN = "fainting_risk"

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


def load_dataset(filename):
    path = get_project_root() / "data" / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def prepare_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    return data.drop(columns=excluded)


def create_base_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )


def main():
    print("=" * 70)
    print("PaceMate-AI Production Fainting-Risk Model")
    print("=" * 70)

    train_data = load_dataset("train_dataset.csv")

    X_train = prepare_features(train_data)
    y_train = train_data[TARGET_COLUMN]

    print()
    print(f"Training rows: {len(train_data)}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Positive cases: {y_train.sum()}")
    print(f"Positive rate: {y_train.mean():.4f}")

    print()
    print("Training isotonic-calibrated Random Forest...")

    model = CalibratedClassifierCV(
        estimator=create_base_model(),
        method="isotonic",
        cv=5,
    )

    model.fit(
        X_train,
        y_train,
    )

    models_directory = (
        get_project_root()
        / "models"
    )

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        models_directory
        / "fainting_risk_calibrated_model.joblib"
    )

    joblib.dump(
        model,
        output_path,
    )

    print()
    print(f"Saved production model:")
    print(output_path)

    print()
    print("=" * 70)
    print("PRODUCTION MODEL BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
