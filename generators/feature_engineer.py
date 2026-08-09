import pandas as pd
from pathlib import Path


def load_data():
    project_root = Path(__file__).resolve().parent.parent

    daily_path = (
        project_root
        / "data"
        / "daily_observations.csv"
    )

    target_path = (
        project_root
        / "data"
        / "daily_targets.csv"
    )

    if not daily_path.exists():
        raise FileNotFoundError(
            f"Daily data not found at: {daily_path}"
        )

    if not target_path.exists():
        raise FileNotFoundError(
            f"Target data not found at: {target_path}"
        )

    daily_data = pd.read_csv(
        daily_path
    )

    target_data = pd.read_csv(
        target_path
    )

    return daily_data, target_data


def create_historical_features(
    daily_data
):

    data = daily_data.copy()

    data = data.sort_values(
        [
            "participant_id",
            "day"
        ]
    )

    grouped = data.groupby(
        "participant_id"
    )

    # Previous-day features

    data["previous_sleep_hours"] = (
        grouped["sleep_hours"].shift(1)
    )

    data["previous_water_intake_ml"] = (
        grouped["water_intake_ml"].shift(1)
    )

    data["previous_resting_hr"] = (
        grouped["resting_hr"].shift(1)
    )

    data["previous_hrv"] = (
        grouped["hrv"].shift(1)
    )

    data["previous_symptom_severity"] = (
        grouped["symptom_severity"].shift(1)
    )

    # Three-day rolling features

    data["sleep_3day_mean"] = (
        grouped["sleep_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )
    )

    data["water_3day_mean"] = (
        grouped["water_intake_ml"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )
    )

    data["hrv_3day_mean"] = (
        grouped["hrv"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )
    )

    data["symptom_3day_mean"] = (
        grouped["symptom_severity"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3, min_periods=1)
            .mean()
        )
    )

    # Seven-day rolling features

    data["sleep_7day_mean"] = (
        grouped["sleep_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
    )

    data["water_7day_mean"] = (
        grouped["water_intake_ml"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
    )

    data["hrv_7day_mean"] = (
        grouped["hrv"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
    )

    data["symptom_7day_mean"] = (
        grouped["symptom_severity"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
    )

    # Historical trends

    data["hrv_change"] = (
        data["hrv"]
        - data["previous_hrv"]
    )

    data["resting_hr_change"] = (
        data["resting_hr"]
        - data["previous_resting_hr"]
    )

    data["symptom_change"] = (
        data["symptom_severity"]
        - data["previous_symptom_severity"]
    )

    data["sleep_deficit"] = (
        8.0
        - data["sleep_7day_mean"]
    )

    data["hydration_deficit"] = (
        2500
        - data["water_7day_mean"]
    )

    return data


def merge_features_and_targets(
    features,
    targets
):

    target_columns = [
        "participant_id",
        "day",
        "flare_risk",
        "dizziness_risk",
        "fatigue_risk",
        "fainting_risk",
        "need_to_hydrate",
        "need_to_rest"
    ]

    targets = targets[
        target_columns
    ]

    merged = features.merge(
        targets,
        on=[
            "participant_id",
            "day"
        ],
        how="inner"
    )

    return merged


def remove_invalid_rows(
    data
):

    required_columns = [
        "previous_sleep_hours",
        "previous_water_intake_ml",
        "previous_resting_hr",
        "previous_hrv",
        "previous_symptom_severity"
    ]

    data = data.dropna(
        subset=required_columns
    )

    return data


def save_training_data(
    data
):

    project_root = (
        Path(__file__)
        .resolve()
        .parent.parent
    )

    output_path = (
        project_root
        / "data"
        / "training_dataset.csv"
    )

    data.to_csv(
        output_path,
        index=False
    )

    return output_path


def main():

    print(
        "Loading daily observations and targets..."
    )

    daily_data, target_data = load_data()

    print(
        f"Daily observations: {len(daily_data)}"
    )

    print(
        f"Target rows: {len(target_data)}"
    )

    print(
        "Creating historical features..."
    )

    features = create_historical_features(
        daily_data
    )

    print(
        f"Feature rows: {len(features)}"
    )

    print(
        "Merging features with targets..."
    )

    training_data = merge_features_and_targets(
        features,
        target_data
    )

    print(
        f"Merged rows: {len(training_data)}"
    )

    training_data = remove_invalid_rows(
        training_data
    )

    print(
        f"Final training rows: {len(training_data)}"
    )

    output_path = save_training_data(
        training_data
    )

    print(
        f"Saved training dataset to: {output_path}"
    )

    print()
    print(
        f"Training columns: {len(training_data.columns)}"
    )

    print(
        training_data.head()
    )


if __name__ == "__main__":
    main()