import pandas as pd
from pathlib import Path
import random
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


RANDOM_SEEDS = [
    42,
    123,
    456,
    789,
    1000,
]

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

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


def create_participant_split(
    data,
    seed
):
    participants = sorted(
        data["participant_id"].unique()
    )

    random.seed(seed)

    random.shuffle(participants)

    total = len(participants)

    train_count = int(
        total * TRAIN_FRACTION
    )

    validation_count = int(
        total * VALIDATION_FRACTION
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
    participants
):
    return data[
        data["participant_id"].isin(
            participants
        )
    ].copy()


def verify_no_overlap(
    train_participants,
    validation_participants,
    test_participants
):
    train_set = set(
        train_participants
    )

    validation_set = set(
        validation_participants
    )

    test_set = set(
        test_participants
    )

    if train_set & validation_set:
        return False

    if train_set & test_set:
        return False

    if validation_set & test_set:
        return False

    return True


def train_model(
    X_train,
    y_train
):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_discrimination(
    model,
    X_data,
    y_data
):
    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    roc_auc = roc_auc_score(
        y_data,
        probabilities
    )

    pr_auc = average_precision_score(
        y_data,
        probabilities
    )

    return (
        roc_auc,
        pr_auc
    )


def main():
    print("=" * 70)
    print(
        "PaceMate-AI Experiment 1A"
    )
    print(
        "Repeated Discrimination Robustness"
    )
    print("=" * 70)

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

    for split_number, seed in enumerate(
        RANDOM_SEEDS,
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"Participant Split {split_number}"
        )
        print(
            f"Random seed: {seed}"
        )
        print("=" * 70)

        (
            train_participants,
            validation_participants,
            test_participants,
        ) = create_participant_split(
            data,
            seed
        )

        overlap_ok = verify_no_overlap(
            train_participants,
            validation_participants,
            test_participants
        )

        if not overlap_ok:
            raise RuntimeError(
                "Participant overlap detected."
            )

        train_data = create_dataset(
            data,
            train_participants
        )

        validation_data = create_dataset(
            data,
            validation_participants
        )

        test_data = create_dataset(
            data,
            test_participants
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

        X_train = prepare_features(
            train_data
        )

        X_validation = prepare_features(
            validation_data
        )

        X_test = prepare_features(
            test_data
        )

        for target_column in TARGET_COLUMNS:

            print()
            print(
                f"Target: {target_column}"
            )

            y_train = train_data[
                target_column
            ]

            y_validation = validation_data[
                target_column
            ]

            y_test = test_data[
                target_column
            ]

            model = train_model(
                X_train,
                y_train
            )

            validation_roc_auc, validation_pr_auc = (
                evaluate_discrimination(
                    model,
                    X_validation,
                    y_validation
                )
            )

            test_roc_auc, test_pr_auc = (
                evaluate_discrimination(
                    model,
                    X_test,
                    y_test
                )
            )

            print(
                f"Validation ROC-AUC: "
                f"{validation_roc_auc:.4f}"
            )

            print(
                f"Validation PR-AUC: "
                f"{validation_pr_auc:.4f}"
            )

            print(
                f"Test ROC-AUC: "
                f"{test_roc_auc:.4f}"
            )

            print(
                f"Test PR-AUC: "
                f"{test_pr_auc:.4f}"
            )

            all_results.append({
                "split": split_number,
                "seed": seed,
                "target": target_column,
                "validation_roc_auc": validation_roc_auc,
                "validation_pr_auc": validation_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_pr_auc": test_pr_auc,
            })

    results = pd.DataFrame(
        all_results
    )

    output_path = (
        get_project_root()
        / "results"
        / "repeated_discrimination_robustness.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print()
    print("=" * 70)
    print(
        "EXPERIMENT 1A COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Results saved to:"
    )

    print(output_path)

    print()
    print(
        "Overall results:"
    )

    summary = (
        results
        .groupby("target")
        [
            [
                "validation_roc_auc",
                "validation_pr_auc",
                "test_roc_auc",
                "test_pr_auc",
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

    print(summary)


if __name__ == "__main__":
    main()