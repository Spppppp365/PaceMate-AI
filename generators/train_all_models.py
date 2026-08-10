import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier


RANDOM_SEED = 42

N_ESTIMATORS = 300
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 5


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


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def save_model(model, target_column):
    models_directory = (
        get_project_root()
        / "models"
    )

    models_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        models_directory
        / f"{target_column}_model.joblib"
    )

    joblib.dump(
        model,
        output_path
    )

    return output_path


def main():
    print("=" * 60)
    print("PaceMate-AI Multi-Target Model Training")
    print("=" * 60)

    print()
    print("Loading datasets...")

    train_data = load_dataset(
        "train_dataset.csv"
    )

    validation_data = load_dataset(
        "validation_dataset.csv"
    )

    test_data = load_dataset(
        "test_dataset.csv"
    )

    print(
        f"Training rows: {len(train_data)}"
    )

    print(
        f"Validation rows: {len(validation_data)}"
    )

    print(
        f"Test rows: {len(test_data)}"
    )

    X_train = prepare_features(
        train_data
    )

    X_validation = prepare_features(
        validation_data
    )

    X_test = prepare_features(
        test_data
    )

    print()
    print(
        f"Number of features: {X_train.shape[1]}"
    )

    print()
    print("Training all target models...")

    for target_column in TARGET_COLUMNS:

        print()
        print("-" * 60)
        print(
            f"Training: {target_column}"
        )
        print("-" * 60)

        y_train = train_data[
            target_column
        ]

        model = train_model(
            X_train,
            y_train
        )

        output_path = save_model(
            model,
            target_column
        )

        print(
            f"Saved: {output_path}"
        )

    print()
    print("=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()