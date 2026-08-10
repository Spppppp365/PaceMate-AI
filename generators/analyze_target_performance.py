import pandas as pd
from pathlib import Path


TARGET_COLUMNS = [
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


def analyze_target_distribution(data, dataset_name):
    print()
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    for target in TARGET_COLUMNS:
        positive = int(data[target].sum())
        total = len(data)
        rate = positive / total

        print(
            f"{target:20s} "
            f"positive={positive:6d} "
            f"negative={total - positive:6d} "
            f"rate={rate:.4f}"
        )


def analyze_target_relationships(data):
    feature_columns = [
        "sleep_hours",
        "water_intake_ml",
        "resting_hr",
        "hrv",
        "dizziness",
        "fatigue",
        "brain_fog",
        "symptom_severity",
        "activity_level",
        "stress_level",
    ]

    available_features = [
        column
        for column in feature_columns
        if column in data.columns
    ]

    print()
    print("=" * 70)
    print("TARGET / FEATURE MEAN DIFFERENCES")
    print("=" * 70)

    for target in TARGET_COLUMNS:

        print()
        print(f"Target: {target}")

        positive_rows = data[
            data[target] == 1
        ]

        negative_rows = data[
            data[target] == 0
        ]

        comparison = []

        for feature in available_features:
            positive_mean = positive_rows[
                feature
            ].mean()

            negative_mean = negative_rows[
                feature
            ].mean()

            difference = (
                positive_mean
                - negative_mean
            )

            comparison.append({
                "feature": feature,
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "difference": difference,
            })

        comparison_df = pd.DataFrame(
            comparison
        )

        comparison_df["absolute_difference"] = (
            comparison_df["difference"]
            .abs()
        )

        comparison_df = comparison_df.sort_values(
            "absolute_difference",
            ascending=False,
        )

        print(
            comparison_df[
                [
                    "feature",
                    "positive_mean",
                    "negative_mean",
                    "difference",
                ]
            ].round(3).to_string(
                index=False
            )
        )


def main():
    print("=" * 70)
    print("PaceMate-AI Target Performance Analysis")
    print("=" * 70)

    train_data = load_dataset(
        "train_dataset.csv"
    )

    validation_data = load_dataset(
        "validation_dataset.csv"
    )

    test_data = load_dataset(
        "test_dataset.csv"
    )

    analyze_target_distribution(
        train_data,
        "TRAINING DATASET"
    )

    analyze_target_distribution(
        validation_data,
        "VALIDATION DATASET"
    )

    analyze_target_distribution(
        test_data,
        "TEST DATASET"
    )

    analyze_target_relationships(
        train_data
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()