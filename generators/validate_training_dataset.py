import pandas as pd
from pathlib import Path


TARGET_COLUMNS = [
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest"
]


FUTURE_COLUMNS = [
    "next_symptom_severity",
    "next_dizziness",
    "next_fatigue",
    "next_brain_fog",
    "next_hrv",
    "next_resting_hr",
    "next_water_intake_ml",
    "next_activity_level"
]


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


def check_missing_values(data):
    print()
    print("--- Missing Values ---")

    missing = data.isnull().sum()

    print(missing)

    total_missing = missing.sum()

    print()
    print(
        f"Total missing values: {total_missing}"
    )

    return total_missing == 0


def check_duplicates(data):
    print()
    print("--- Duplicate Participant/Day Rows ---")

    duplicates = data.duplicated(
        subset=[
            "participant_id",
            "day"
        ]
    ).sum()

    print(
        f"Duplicate rows: {duplicates}"
    )

    return duplicates == 0


def check_participant_counts(data):
    print()
    print("--- Rows Per Participant ---")

    counts = (
        data
        .groupby("participant_id")
        .size()
    )

    print(
        counts.value_counts()
        .sort_index()
    )

    print()
    print(
        f"Number of participants: {len(counts)}"
    )

    print(
        f"Minimum rows per participant: {counts.min()}"
    )

    print(
        f"Maximum rows per participant: {counts.max()}"
    )


def check_target_columns(data):
    print()
    print("--- Target Columns ---")

    all_present = True

    for column in TARGET_COLUMNS:

        if column in data.columns:

            print(
                f"{column}: PRESENT"
            )

        else:

            print(
                f"{column}: MISSING"
            )

            all_present = False

    return all_present


def check_future_columns(data):
    print()
    print("--- Future-Derived Columns ---")

    found_future_columns = []

    for column in FUTURE_COLUMNS:

        if column in data.columns:

            found_future_columns.append(
                column
            )

    if found_future_columns:

        print(
            "WARNING: Future-derived columns found:"
        )

        for column in found_future_columns:

            print(
                f"  {column}"
            )

        return False

    print(
        "No future-derived columns found."
    )

    return True


def check_target_distributions(data):
    print()
    print("--- Target Distributions ---")

    for column in TARGET_COLUMNS:

        print()
        print(column)

        print(
            data[column]
            .value_counts()
            .sort_index()
        )


def check_day_range(data):
    print()
    print("--- Day Range ---")

    print(
        f"Minimum day: {data['day'].min()}"
    )

    print(
        f"Maximum day: {data['day'].max()}"
    )


def main():

    print("=" * 60)

    print(
        "PaceMate-AI Training Dataset Validation"
    )

    print("=" * 60)

    data = load_training_data()

    print()
    print(
        f"Rows: {len(data)}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print()
    print(
        "Column names:"
    )

    print(
        list(data.columns)
    )

    missing_ok = check_missing_values(
        data
    )

    duplicates_ok = check_duplicates(
        data
    )

    check_participant_counts(
        data
    )

    targets_ok = check_target_columns(
        data
    )

    future_ok = check_future_columns(
        data
    )

    check_target_distributions(
        data
    )

    check_day_range(
        data
    )

    print()
    print("=" * 60)

    if (
        missing_ok
        and duplicates_ok
        and targets_ok
        and future_ok
    ):

        print(
            "VALIDATION PASSED"
        )

    else:

        print(
            "VALIDATION FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()