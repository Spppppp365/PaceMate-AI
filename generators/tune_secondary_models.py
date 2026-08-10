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

TARGET_COLUMNS = [
    "dizziness_risk",
    "fatigue_risk",
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


CONFIGURATIONS = {
    "baseline": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    "deeper": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    "more_trees": {
        "n_estimators": 500,
        "max_depth": 16,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    },
    "smaller_leaf": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
    },
    "larger_leaf": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 8,
        "class_weight": "balanced",
    },
    "no_class_weight": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 5,
        "class_weight": None,
    },
}


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


def train_model(
    X_train,
    y_train,
    configuration,
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
        class_weight=configuration[
            "class_weight"
        ],
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

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

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():
    print("=" * 70)
    print("PaceMate-AI Secondary Target Model Tuning")
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

    print()
    print(
        f"Training rows: {len(train_data)}"
    )

    print(
        f"Validation rows: {len(validation_data)}"
    )

    print(
        f"Number of features: {X_train.shape[1]}"
    )

    all_results = []

    for target in TARGET_COLUMNS:

        print()
        print("=" * 70)
        print(
            f"TARGET: {target}"
        )
        print("=" * 70)

        y_train = train_data[
            target
        ]

        y_validation = validation_data[
            target
        ]

        for name, configuration in CONFIGURATIONS.items():

            print()
            print(
                f"Training: {name}"
            )

            model = train_model(
                X_train,
                y_train,
                configuration,
            )

            metrics = evaluate_model(
                model,
                X_validation,
                y_validation,
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
                f"Precision: {metrics['precision']:.4f}"
            )

            all_results.append({
                "target": target,
                "name": name,
                **configuration,
                **metrics,
            })

    results = pd.DataFrame(
        all_results
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        results[
            [
                "target",
                "name",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].round(4).to_string(
            index=False
        )
    )

    output_path = (
        get_project_root()
        / "results"
        / "secondary_model_tuning.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()