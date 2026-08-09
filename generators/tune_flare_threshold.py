import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


TARGET_COLUMN = "flare_risk"

RANDOM_SEED = 42

N_ESTIMATORS = 300
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 5

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
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


def prepare_target(data):
    return data[TARGET_COLUMN]


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


def evaluate_threshold(
    probabilities,
    y_true,
    threshold
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    print("=" * 60)
    print(
        "PaceMate-AI Flare Risk Threshold Tuning"
    )
    print("=" * 60)

    train_data = load_dataset(
        "train_dataset.csv"
    )

    validation_data = load_dataset(
        "validation_dataset.csv"
    )

    X_train = prepare_features(
        train_data
    )

    X_validation = prepare_features(
        validation_data
    )

    y_train = prepare_target(
        train_data
    )

    y_validation = prepare_target(
        validation_data
    )

    print()
    print(
        f"Training rows: {len(train_data)}"
    )

    print(
        f"Validation rows: "
        f"{len(validation_data)}"
    )

    print(
        f"Number of features: "
        f"{X_train.shape[1]}"
    )

    print()
    print("Training model...")

    model = train_model(
        X_train,
        y_train
    )

    print("Training complete.")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    results = []

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            probabilities,
            y_validation,
            threshold
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 60)
    print("VALIDATION THRESHOLD RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    best = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print()
    print(
        "Best threshold by validation F1:"
    )

    print(
        f"Threshold: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
