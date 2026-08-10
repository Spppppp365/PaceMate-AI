import pandas as pd
from pathlib import Path


def load_daily_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "daily_observations.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Daily dataset not found at: {data_path}"
        )

    return pd.read_csv(data_path)


def create_targets(daily_data):
    data = daily_data.copy()

    grouped = data.groupby("participant_id")

    data["next_symptom_severity"] = (
        grouped["symptom_severity"].shift(-1)
    )

    data["next_dizziness"] = (
        grouped["dizziness"].shift(-1)
    )

    data["next_fatigue"] = (
        grouped["fatigue"].shift(-1)
    )

    data["next_brain_fog"] = (
        grouped["brain_fog"].shift(-1)
    )

    data["next_hrv"] = (
        grouped["hrv"].shift(-1)
    )

    data["next_resting_hr"] = (
        grouped["resting_hr"].shift(-1)
    )

    data["next_water_intake_ml"] = (
        grouped["water_intake_ml"].shift(-1)
    )

    data["next_activity_level"] = (
        grouped["activity_level"].shift(-1)
    )

    data["flare_risk"] = (
        data["next_symptom_severity"] >= 7
    ).astype(int)

    data["dizziness_risk"] = (
        data["next_dizziness"] >= 7
    ).astype(int)

    data["fatigue_risk"] = (
        data["next_fatigue"] >= 7
    ).astype(int)

    data["fainting_risk"] = (
        (data["next_dizziness"] >= 8)
        & (data["next_symptom_severity"] >= 8)
        & (data["next_hrv"] <= 30)
    ).astype(int)

    hydration_score = (
        (data["next_symptom_severity"] >= 6).astype(int)
        + (data["next_dizziness"] >= 6).astype(int)
        + (data["next_hrv"] <= 35).astype(int)
        + (data["water_intake_ml"] < 2500).astype(int)
    )

    data["need_to_hydrate"] = (
        hydration_score >= 3
    ).astype(int)

    rest_score = (
        (data["next_symptom_severity"] >= 6).astype(int)
        + (data["next_fatigue"] >= 6).astype(int)
        + (data["next_brain_fog"] >= 6).astype(int)
        + (data["activity_level"] <= 3).astype(int)
    )

    data["need_to_rest"] = (
        rest_score >= 3
    ).astype(int)

    data = data.dropna()

    return data


def main():
    daily_data = load_daily_data()

    targets = create_targets(daily_data)

    project_root = Path(__file__).resolve().parent.parent

    output_path = (
        project_root
        / "data"
        / "daily_targets.csv"
    )

    targets.to_csv(
        output_path,
        index=False
    )

    print(
        f"Generated {len(targets)} target rows."
    )

    print(
        f"Columns: {len(targets.columns)}"
    )

    print(
        f"Saved targets to: {output_path}"
    )

    print()
    print("Target distributions:")

    target_columns = [
        "flare_risk",
        "dizziness_risk",
        "fatigue_risk",
        "fainting_risk",
        "need_to_hydrate",
        "need_to_rest"
    ]

    for column in target_columns:
        print()
        print(column)
        print(
            targets[column]
            .value_counts()
            .sort_index()
        )


if __name__ == "__main__":
    main()