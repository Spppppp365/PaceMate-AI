import pandas as pd
from pathlib import Path
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


RANDOM_SEED = 42

N_REPEATS = 5

N_ESTIMATORS = 300
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 5

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

TARGETS = [
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

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
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


def split_participants(data, seed):
    participants = sorted(
        data["participant_id"].unique()
    )

    random.seed(seed)

    random.shuffle(
        participants
    )

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
    participants,
):
    return data[
        data["participant_id"].isin(
            participants
        )
    ].copy()


def verify_no_overlap(
    train_participants,
    validation_participants,
    test_participants,
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
    y_train,
):
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
        y_train,
    )

    return model


def select_best_threshold(
    probabilities,
    y_validation,
):
    best_threshold = None
    best_precision = None
    best_recall = None
    best_f1 = -1

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return {
        "threshold": best_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "f1": best_f1,
    }


def evaluate_at_threshold(
    probabilities,
    y_test,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    print("=" * 80)
    print(
        "PaceMate-AI Repeated Threshold/F1 Robustness"
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
        N_REPEATS + 1,
    ):

        print()
        print("#" * 80)
        print(
            f"REPEAT {repeat} / {N_REPEATS}"
        )
        print("#" * 80)

        split_seed = (
            RANDOM_SEED + repeat - 1
        )

        (
            train_participants,
            validation_participants,
            test_participants,
        ) = split_participants(
            data,
            split_seed,
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

        overlap_ok = verify_no_overlap(
            train_participants,
            validation_participants,
            test_participants,
        )

        if not overlap_ok:
            raise RuntimeError(
                "Participant overlap detected."
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

        for target in TARGETS:

            print()
            print("-" * 80)
            print(
                f"Target: {target}"
            )
            print("-" * 80)

            y_train = train_data[
                target
            ]

            y_validation = validation_data[
                target
            ]

            y_test = test_data[
                target
            ]

            model = train_model(
                X_train,
                y_train,
            )

            validation_probabilities = (
                model.predict_proba(
                    X_validation
                )[:, 1]
            )

            test_probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            best = select_best_threshold(
                validation_probabilities,
                y_validation,
            )

            test_metrics = (
                evaluate_at_threshold(
                    test_probabilities,
                    y_test,
                    best["threshold"],
                )
            )

            print(
                f"Validation threshold: "
                f"{best['threshold']:.2f}"
            )

            print(
                f"Validation precision: "
                f"{best['precision']:.4f}"
            )

            print(
                f"Validation recall: "
                f"{best['recall']:.4f}"
            )

            print(
                f"Validation F1: "
                f"{best['f1']:.4f}"
            )

            print(
                f"Test precision: "
                f"{test_metrics['precision']:.4f}"
            )

            print(
                f"Test recall: "
                f"{test_metrics['recall']:.4f}"
            )

            print(
                f"Test F1: "
                f"{test_metrics['f1']:.4f}"
            )

            all_results.append(
                {
                    "repeat": repeat,
                    "split_seed": split_seed,
                    "target": target,
                    "train_participants": len(
                        train_participants
                    ),
                    "validation_participants": len(
                        validation_participants
                    ),
                    "test_participants": len(
                        test_participants
                    ),
                    "validation_threshold": (
                        best["threshold"]
                    ),
                    "validation_precision": (
                        best["precision"]
                    ),
                    "validation_recall": (
                        best["recall"]
                    ),
                    "validation_f1": (
                        best["f1"]
                    ),
                    "test_precision": (
                        test_metrics["precision"]
                    ),
                    "test_recall": (
                        test_metrics["recall"]
                    ),
                    "test_f1": (
                        test_metrics["f1"]
                    ),
                }
            )

    results_dataframe = pd.DataFrame(
        all_results
    )

    output_path = (
        get_project_root()
        / "results"
        / "repeated_threshold_f1_robustness.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "REPEATED THRESHOLD/F1 ROBUSTNESS SUMMARY"
    )
    print("=" * 80)

    summary = (
        results_dataframe
        .groupby("target")
        [
            [
                "validation_threshold",
                "validation_f1",
                "test_precision",
                "test_recall",
                "test_f1",
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
        f"Results saved to:"
    )

    print(
        output_path
    )

    print()
    print("=" * 80)
    print(
        "REPEATED THRESHOLD/F1 ROBUSTNESS COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()