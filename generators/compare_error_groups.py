import pandas as pd
from pathlib import Path


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
    "target",
    "actual",
    "predicted",
    "probability",
    "threshold",
    "error_type",
]


def get_project_root():
    return Path(__file__).resolve().parent.parent


def load_error_analysis():
    path = (
        get_project_root()
        / "results"
        / "model_error_analysis.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Error analysis file not found: {path}"
        )

    return pd.read_csv(path)


def get_numeric_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    numeric_columns = data.select_dtypes(
        include="number"
    ).columns

    return [
        column
        for column in numeric_columns
        if column not in excluded
    ]


def analyze_target(data, target, feature_columns):
    target_data = data[
        data["target"] == target
    ].copy()

    print()
    print("=" * 70)
    print(f"ERROR GROUP ANALYSIS: {target}")
    print("=" * 70)

    groups = [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    available_groups = [
        group
        for group in groups
        if group in target_data["error_type"].values
    ]

    summary = (
        target_data
        .groupby("error_type")[feature_columns]
        .mean()
        .reindex(available_groups)
    )

    print()
    print(
        summary.round(4).to_string()
    )

    return summary


def main():
    print("=" * 70)
    print("PaceMate-AI Error Group Feature Analysis")
    print("=" * 70)

    data = load_error_analysis()

    feature_columns = get_numeric_features(
        data
    )

    print()
    print(
        f"Numeric features analyzed: "
        f"{len(feature_columns)}"
    )

    all_results = []

    for target in TARGETS:
        summary = analyze_target(
            data,
            target,
            feature_columns,
        )

        summary["target"] = target
        summary["error_type"] = summary.index

        all_results.append(
            summary.reset_index(drop=True)
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        get_project_root()
        / "results"
        / "error_group_feature_analysis.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 70)
    print("ERROR GROUP ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()