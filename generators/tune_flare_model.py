import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


RANDOM_SEED = 42

TARGET_COLUMN = "flare_risk"

EXCLUDED_COLUMNS = [
    "participant_id",
    "day",

    # Current-day symptom outcomes
    "dizziness",
    "fatigue",
    "brain_fog",
    "symptom_severity",

    # Target columns
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest",
]


MODEL_CONFIGURATIONS = [
    {
        "name": "baseline",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
    },
    {
        "name": "deeper",
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 5,
    },
    {
        "name": "more_trees",
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 5,
    },
    {
        "name": "smaller_leaf",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 3,
    },
    {
        "name": "larger_leaf",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 8,
    },
    {
        "name": "deeper_smaller_leaf",
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 3,
    },
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


def train_model(
    X_train,
    y_train,
    configuration
):
    model = RandomForestClassifier(
        n_estimators=configuration[
            "n_estimators"
        ],
        max_depth=configuration[
            "max_depth"
        ],
        min_samples_leaf=configuration[
            "min_samples_leaf"
        ],
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_validation,
    y_validation
):
    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    return {
        "precision": precision_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_validation,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_validation,
            probabilities,
        ),
    }


def main():
    print("=" * 70)
    print("PaceMate-AI Flare Risk Model Tuning")
    print("=" * 70)

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
    print(
        "Testing model configurations..."
    )

    results = []

    for configuration in MODEL_CONFIGURATIONS:

        print()
        print(
            f"Training: "
            f"{configuration['name']}"
        )

        model = train_model(
            X_train,
            y_train,
            configuration
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation
        )

        result = {
            "name": configuration["name"],
            "n_estimators": configuration[
                "n_estimators"
            ],
            "max_depth": configuration[
                "max_depth"
            ],
            "min_samples_leaf": configuration[
                "min_samples_leaf"
            ],
            **metrics,
        }

        results.append(
            result
        )

        print(
            f"F1:     {metrics['f1']:.4f}"
        )

        print(
            f"PR-AUC: {metrics['pr_auc']:.4f}"
        )

        print(
            f"ROC-AUC:{metrics['roc_auc']:.4f}"
        )

        print(
            f"Recall: {metrics['recall']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "f1",
        ascending=False
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    print()
    print(
        "Best configuration by validation F1:"
    )

    best = results_df.iloc[0]

    print(
        f"  {best['name']}"
    )

    print(
        f"  F1: {best['f1']:.4f}"
    )

    print(
        f"  PR-AUC: {best['pr_auc']:.4f}"
    )


if __name__ == "__main__":
    main()