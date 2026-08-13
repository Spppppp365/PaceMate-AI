import pandas as pd
from pathlib import Path
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


RANDOM_SEED = 42
NUMBER_OF_REPEATS = 5

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
    return (
        Path(__file__)
        .resolve()
        .parent.parent
    )


def load_dataset():
    path = (
        get_project_root()
        / "data"
        / "training_dataset.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {path}"
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


def split_participants(data, seed):

    participants = sorted(
        data["participant_id"].unique()
    )

    rng = random.Random(seed)

    rng.shuffle(participants)

    total = len(participants)

    train_count = int(
        total * 0.70
    )

    validation_count = int(
        total * 0.15
    )

    train_participants = participants[
        :train_count
    ]

    validation_participants = participants[
        train_count:
        train_count + validation_count
    ]

    test_participants = participants[
        train_count + validation_count:
    ]

    return (
        train_participants,
        validation_participants,
        test_participants,
    )


def create_dataset(
    data,
    participants,
):

    return data[
        data["participant_id"].isin(
            participants
        )
    ].copy()


def train_model(
    X_train,
    y_train,
):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
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
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
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

    print("=" * 80)
    print(
        "PaceMate-AI Repeated Longitudinal "
        "Feature Ablation Robustness"
    )
    print("=" * 80)

    data = load_dataset()

    print()
    print(
        f"Total rows: {len(data)}"
    )

    print(
        f"Total participants: "
        f"{data['participant_id'].nunique()}"
    )

    all_results = []

    for repeat in range(
        1,
        NUMBER_OF_REPEATS + 1,
    ):

        print()
        print("#" * 80)
        print(
            f"REPEAT {repeat} / "
            f"{NUMBER_OF_REPEATS}"
        )
        print("#" * 80)

        seed = (
            RANDOM_SEED + repeat - 1
        )

        (
            train_participants,
            validation_participants,
            test_participants,
        ) = split_participants(
            data,
            seed,
        )

        train_data = create_dataset(
            data,
            train_participants,
        )

        validation_data = create_dataset(
            data,
            validation_participants,
        )

        test_data = create_dataset(
            data,
            test_participants,
        )

        print()
        print(
            f"Training participants: "
            f"{len(train_participants)}"
        )

        print(
            f"Validation participants: "
            f"{len(validation_participants)}"
        )

        print(
            f"Test participants: "
            f"{len(test_participants)}"
        )

        print()
        print(
            f"Training rows: "
            f"{len(train_data)}"
        )

        print(
            f"Validation rows: "
            f"{len(validation_data)}"
        )

        print(
            f"Test rows: "
            f"{len(test_data)}"
        )

        X_train_full = prepare_features(
            train_data
        )

        X_test_full = prepare_features(
            test_data
        )

        X_train_same_day = (
            get_same_day_features(
                train_data
            )
        )

        X_test_same_day = (
            get_same_day_features(
                test_data
            )
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

        for target in TARGET_COLUMNS:

            print()
            print("-" * 80)
            print(
                f"Target: {target}"
            )
            print("-" * 80)

            y_train = train_data[
                target
            ]

            y_test = test_data[
                target
            ]

            print(
                "Training full-feature model..."
            )

            full_model = train_model(
                X_train_full,
                y_train,
            )

            full_metrics = evaluate_model(
                full_model,
                X_test_full,
                y_test,
            )

            print(
                f"Full ROC-AUC: "
                f"{full_metrics['roc_auc']:.4f}"
            )

            print(
                f"Full PR-AUC:   "
                f"{full_metrics['pr_auc']:.4f}"
            )

            print(
                f"Full Brier:    "
                f"{full_metrics['brier_score']:.6f}"
            )

            print()
            print(
                "Training same-day-only model..."
            )

            same_day_model = train_model(
                X_train_same_day,
                y_train,
            )

            same_day_metrics = evaluate_model(
                same_day_model,
                X_test_same_day,
                y_test,
            )

            print(
                f"Same-day ROC-AUC: "
                f"{same_day_metrics['roc_auc']:.4f}"
            )

            print(
                f"Same-day PR-AUC:   "
                f"{same_day_metrics['pr_auc']:.4f}"
            )

            print(
                f"Same-day Brier:    "
                f"{same_day_metrics['brier_score']:.6f}"
            )

            roc_difference = (
                full_metrics["roc_auc"]
                - same_day_metrics["roc_auc"]
            )

            pr_difference = (
                full_metrics["pr_auc"]
                - same_day_metrics["pr_auc"]
            )

            brier_difference = (
                full_metrics["brier_score"]
                - same_day_metrics["brier_score"]
            )

            print()
            print(
                f"ROC-AUC improvement: "
                f"{roc_difference:+.4f}"
            )

            print(
                f"PR-AUC improvement:   "
                f"{pr_difference:+.4f}"
            )

            print(
                f"Brier difference:      "
                f"{brier_difference:+.6f}"
            )

            all_results.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "target": target,

                    "full_roc_auc":
                        full_metrics["roc_auc"],

                    "same_day_roc_auc":
                        same_day_metrics["roc_auc"],

                    "roc_auc_improvement":
                        roc_difference,

                    "full_pr_auc":
                        full_metrics["pr_auc"],

                    "same_day_pr_auc":
                        same_day_metrics["pr_auc"],

                    "pr_auc_improvement":
                        pr_difference,

                    "full_brier_score":
                        full_metrics["brier_score"],

                    "same_day_brier_score":
                        same_day_metrics["brier_score"],

                    "brier_difference":
                        brier_difference,

                    "full_feature_count":
                        X_train_full.shape[1],

                    "same_day_feature_count":
                        X_train_same_day.shape[1],
                }
            )

    results = pd.DataFrame(
        all_results
    )

    results_dir = (
        get_project_root()
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / "repeated_longitudinal_ablation.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "REPEATED ABLATION SUMMARY"
    )
    print("=" * 80)

    summary = (
        results
        .groupby("target")
        [
            [
                "roc_auc_improvement",
                "pr_auc_improvement",
                "brier_difference",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        summary.round(4).to_string()
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print()
    print("=" * 80)
    print(
        "REPEATED LONGITUDINAL "
        "ABLATION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()