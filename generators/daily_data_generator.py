import csv
import random
from pathlib import Path

import pandas as pd


def load_participants() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "participants.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Participant dataset not found at: {data_path}"
        )

    return pd.read_csv(
        data_path,
        keep_default_na=False
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float
) -> float:
    return max(
        minimum,
        min(maximum, value)
    )


def generate_daily_data(
    participants: pd.DataFrame,
    days_per_participant: int
) -> list[dict]:

    observations = []

    for _, participant in participants.iterrows():

        baseline_severity = float(
            participant["baseline_severity"]
        )

        baseline_hr = float(
            participant["baseline_resting_hr"]
        )

        baseline_hrv = float(
            participant["baseline_hrv"]
        )

        water_goal = float(
            participant["baseline_water_goal_ml"]
        )

        sleep_goal = float(
            participant["baseline_sleep_goal_hours"]
        )

        previous_symptom = baseline_severity

        for day in range(
            1,
            days_per_participant + 1
        ):

            stress_level = random.randint(
                1,
                10
            )

            activity_level = random.randint(
                1,
                10
            )

            sleep_hours = random.gauss(
                sleep_goal,
                0.8
            )

            sleep_hours = round(
                clamp(
                    sleep_hours,
                    4.0,
                    10.0
                ),
                1
            )

            water_intake_ml = int(
                clamp(
                    random.gauss(
                        water_goal,
                        400
                    ),
                    1000,
                    5000
                )
            )

            sleep_deficit = max(
                0,
                sleep_goal - sleep_hours
            )

            hydration_deficit = max(
                0,
                (water_goal - water_intake_ml)
                / water_goal
            )

            resting_hr = (
                baseline_hr
                + stress_level * 1.8
                + hydration_deficit * 25
                + sleep_deficit * 2.5
                + random.gauss(0, 3)
            )

            resting_hr = int(
                round(
                    clamp(
                        resting_hr,
                        45,
                        130
                    )
                )
            )

            hrv = (
                baseline_hrv
                - stress_level * 2.5
                - hydration_deficit * 30
                - sleep_deficit * 5
                + random.gauss(0, 5)
            )

            hrv = int(
                round(
                    clamp(
                        hrv,
                        5,
                        150
                    )
                )
            )

            dizziness_score = (
                baseline_severity * 0.55
                + hydration_deficit * 5
                + max(
                    0,
                    resting_hr - baseline_hr
                ) * 0.12
                + stress_level * 0.25
                + (10 - hrv) * 0.025
                + random.gauss(0, 1.0)
            )

            dizziness = int(
                round(
                    clamp(
                        dizziness_score,
                        0,
                        10
                    )
                )
            )

            fatigue_score = (
                baseline_severity * 0.50
                + sleep_deficit * 2.0
                + stress_level * 0.30
                + max(
                    0,
                    5 - activity_level
                ) * 0.25
                + (90 - hrv) * 0.015
                + random.gauss(0, 1.0)
            )

            fatigue = int(
                round(
                    clamp(
                        fatigue_score,
                        0,
                        10
                    )
                )
            )

            brain_fog_score = (
                baseline_severity * 0.45
                + sleep_deficit * 1.8
                + stress_level * 0.35
                + hydration_deficit * 3
                + (90 - hrv) * 0.02
                + random.gauss(0, 1.0)
            )

            brain_fog = int(
                round(
                    clamp(
                        brain_fog_score,
                        0,
                        10
                    )
                )
            )

            symptom_severity = (
                0.40 * dizziness
                + 0.35 * fatigue
                + 0.25 * brain_fog
            )

            symptom_severity = (
                0.65 * previous_symptom
                + 0.35 * symptom_severity
                + random.gauss(0, 0.4)
            )

            symptom_severity = round(
                clamp(
                    symptom_severity,
                    0,
                    10
                ),
                1
            )

            previous_symptom = symptom_severity

            observations.append(
                {
                    "participant_id": int(
                        participant["participant_id"]
                    ),
                    "day": day,
                    "sleep_hours": sleep_hours,
                    "water_intake_ml": water_intake_ml,
                    "resting_hr": resting_hr,
                    "hrv": hrv,
                    "dizziness": dizziness,
                    "fatigue": fatigue,
                    "brain_fog": brain_fog,
                    "symptom_severity": symptom_severity,
                    "activity_level": activity_level,
                    "stress_level": stress_level,
                }
            )

    return observations


def save_daily_data(
    observations: list[dict]
) -> Path:

    project_root = Path(__file__).resolve().parent.parent

    data_directory = (
        project_root / "data"
    )

    output_path = (
        data_directory
        / "daily_observations.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        fieldnames = observations[0].keys()

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(observations)

    return output_path


def main() -> None:

    random.seed(42)

    participants = load_participants()

    days_per_participant = 180

    observations = generate_daily_data(
        participants,
        days_per_participant
    )

    output_path = save_daily_data(
        observations
    )

    print(
        f"Generated {len(observations)} daily observations."
    )

    print(
        f"Participants: {len(participants)}"
    )

    print(
        f"Days per participant: {days_per_participant}"
    )

    print(
        f"Saved daily data to: {output_path}"
    )

    print(
        observations[0]
    )


if __name__ == "__main__":
    main()