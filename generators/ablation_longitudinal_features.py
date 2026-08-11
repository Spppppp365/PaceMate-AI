import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


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


LONGITUDINAL_PREFIXES = [
    "previous_",
    "sleep_3day_",
    "sleep_7day_",
    "water_3day_",
    "water_7day_",
    "hrv_3day_",
    "hrv_7day_",
    "symptom_3day_",
    "symptom_7day_",
]


LONGITUDINAL_EXACT = [
    "hrv_change",
    "resting_hr_change",
    "symptom_change",
    "sleep_deficit",
    "hydration_deficit",
]


def get_project_root():
    return Path(__file__).resolve().parent.parent


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


def get_same_day_features(data):
    features = prepare_features(data)

    longitudinal_columns = []

    for column in features.columns:
        if any(
            column.startswith(prefix)
            for prefix in LONGITUDINAL_PREFIXES
        ):
            longitudinal_columns.append(column)

        elif column in LONGITUDINAL_EXACT:
            longitudinal_columns.append(column)

    return features.drop(
        columns=longitudinal_columns
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    target,
    feature_set,
):
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "target": target,
        "feature_set": feature_set,
        "number_of_features": X_test.shape[1],
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_test,
            probabilities,
        ),
    }


def main():
    print("=" * 70)
    print("PaceMate-AI Longitudinal Feature Ablation Study")
    print("=" * 70)

    print()
    print("Loading participant-level datasets...")

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
        f"Training rows:   {len(train_data)}"
    )

    print(
        f"Validation rows: {len(validation_data)}"
    )

    print(
        f"Test rows:       {len(test_data)}"
    )

    X_train_full = prepare_features(
        train_data
    )

    X_test_full = prepare_features(
        test_data
    )

    X_train_same_day = get_same_day_features(
        train_data
    )

    X_test_same_day = get_same_day_features(
        test_data
    )

    print()
    print(
        f"Full feature count: "
        f"{X_train_full.shape[1]}"
    )

    print(
        f"Same-day feature count: "
        f"{X_train_same_day.shape[1]}"
    )

    results = []

    for target in TARGET_COLUMNS:

        print()
        print("=" * 70)
        print(
            f"ABLATION: {target}"
        )
        print("=" * 70)

        y_train = train_data[target]
        y_test = test_data[target]

        print()
        print("Training full-feature model...")

        full_model = train_model(
            X_train_full,
            y_train,
        )

        full_result = evaluate_model(
            full_model,
            X_test_full,
            y_test,
            target,
            "full",
        )

        results.append(
            full_result
        )

        print(
            f"Full model ROC-AUC: "
            f"{full_result['roc_auc']:.4f}"
        )

        print(
            f"Full model PR-AUC:   "
            f"{full_result['pr_auc']:.4f}"
        )

        print()
        print("Training same-day-only model...")

        same_day_model = train_model(
            X_train_same_day,
            y_train,
        )

        same_day_result = evaluate_model(
            same_day_model,
            X_test_same_day,
            y_test,
            target,
            "same_day_only",
        )

        results.append(
            same_day_result
        )

        print(
            f"Same-day ROC-AUC: "
            f"{same_day_result['roc_auc']:.4f}"
        )

        print(
            f"Same-day PR-AUC:   "
            f"{same_day_result['pr_auc']:.4f}"
        )

        roc_difference = (
            full_result["roc_auc"]
            - same_day_result["roc_auc"]
        )

        pr_difference = (
            full_result["pr_auc"]
            - same_day_result["pr_auc"]
        )

        print()
        print(
            f"Longitudinal ROC-AUC improvement: "
            f"{roc_difference:+.4f}"
        )

        print(
            f"Longitudinal PR-AUC improvement:   "
            f"{pr_difference:+.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        get_project_root()
        / "results"
        / "longitudinal_feature_ablation.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 70)
    print("LONGITUDINAL FEATURE ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()