import pandas as pd
from pathlib import Path
import random


RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_training_data():
    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    data_path = (
        project_root
        / "data"
        / "training_dataset.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at: {data_path}"
        )

    return pd.read_csv(
        data_path
    )


def split_participants(data):

    participants = sorted(
        data["participant_id"]
        .unique()
    )

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        participants
    )

    total = len(
        participants
    )

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
        test_participants
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

    train_validation_overlap = (
        train_set
        & validation_set
    )

    train_test_overlap = (
        train_set
        & test_set
    )

    validation_test_overlap = (
        validation_set
        & test_set
    )

    if train_validation_overlap:
        return False

    if train_test_overlap:
        return False

    if validation_test_overlap:
        return False

    return True


def save_dataset(
    data,
    filename
):

    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    output_path = (
        project_root
        / "data"
        / filename
    )

    data.to_csv(
        output_path,
        index=False
    )

    return output_path


def main():

    print("=" * 60)

    print(
        "PaceMate-AI Participant-Level Dataset Split"
    )

    print("=" * 60)

    data = load_training_data()

    print()
    print(
        f"Total rows: {len(data)}"
    )

    print(
        f"Total participants: "
        f"{data['participant_id'].nunique()}"
    )

    (
        train_participants,
        validation_participants,
        test_participants
    ) = split_participants(
        data
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

    overlap_ok = verify_no_overlap(
        train_participants,
        validation_participants,
        test_participants
    )

    print()
    print("--- Participant Split ---")

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
    print("--- Row Counts ---")

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

    print()
    print("--- Participant Overlap Check ---")

    if overlap_ok:

        print(
            "No participant overlap detected."
        )

    else:

        print(
            "ERROR: Participant overlap detected."
        )

        raise RuntimeError(
            "Dataset split contains participant overlap."
        )

    train_path = save_dataset(
        train_data,
        "train_dataset.csv"
    )

    validation_path = save_dataset(
        validation_data,
        "validation_dataset.csv"
    )

    test_path = save_dataset(
        test_data,
        "test_dataset.csv"
    )

    print()
    print("--- Saved Files ---")

    print(
        f"Training: {train_path}"
    )

    print(
        f"Validation: {validation_path}"
    )

    print(
        f"Test: {test_path}"
    )

    print()
    print("=" * 60)

    print(
        "DATASET SPLIT COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()