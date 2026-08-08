from dataclasses import dataclass, asdict
from typing import List
import csv
import random
from pathlib import Path

from .config import (
    NUMBER_OF_PARTICIPANTS,
    RANDOM_SEED,
    MIN_AGE,
    MAX_AGE,
    MIN_HEIGHT_CM,
    MAX_HEIGHT_CM,
    MIN_WEIGHT_KG,
    MAX_WEIGHT_KG
)

random.seed(RANDOM_SEED)


@dataclass
class Participant:
    participant_id: int
    age: int
    biological_sex: str
    height_cm: int
    weight_kg: int
    years_since_diagnosis: int
    baseline_severity: int
    medication_group: str
    compression_garments: bool
    mobility_aid: bool
    athlete: bool
    caffeine_sensitive: bool
    baseline_resting_hr: int
    baseline_hrv: int
    baseline_water_goal_ml: int
    baseline_sleep_goal_hours: float


def generate_participant(participant_id: int) -> Participant:
    biological_sex = random.choice(
        ["Female", "Male"]
    )

    age = random.randint(
        MIN_AGE,
        MAX_AGE
    )

    severity = random.randint(
        2,
        8
    )

    medication_group = random.choice(
        [
            "None",
            "Beta Blocker",
            "Fludrocortisone",
            "Midodrine",
            "Ivabradine",
            "Combination"
        ]
    )

    baseline_resting_hr = random.randint(
        50,
        85
    )

    baseline_hrv = random.randint(
        20,
        90
    )

    return Participant(
        participant_id=participant_id,

        age=age,

        biological_sex=biological_sex,

        height_cm=random.randint(
            MIN_HEIGHT_CM,
            MAX_HEIGHT_CM
        ),

        weight_kg=random.randint(
            MIN_WEIGHT_KG,
            MAX_WEIGHT_KG
        ),

        years_since_diagnosis=random.randint(
            0,
            min(15, max(0, age - 10))
        ),

        baseline_severity=severity,

        medication_group=medication_group,

        compression_garments=random.random() < 0.45,

        mobility_aid=random.random() < 0.10,

        athlete=random.random() < 0.15,

        caffeine_sensitive=random.random() < 0.35,

        baseline_resting_hr=baseline_resting_hr,

        baseline_hrv=baseline_hrv,

        baseline_water_goal_ml=random.choice(
            [
                2000,
                2500,
                3000,
                3500
            ]
        ),

        baseline_sleep_goal_hours=random.choice(
            [
                7.0,
                7.5,
                8.0,
                8.5,
                9.0
            ]
        )
    )


def generate_all_participants() -> List[Participant]:
    participants = []

    for participant_id in range(
        1,
        NUMBER_OF_PARTICIPANTS + 1
    ):
        participant = generate_participant(
            participant_id
        )

        participants.append(
            participant
        )

    return participants


def participants_to_records(
    participants: List[Participant]
) -> List[dict]:
    return [
        asdict(participant)
        for participant in participants
    ]


def save_participants_to_csv(
    participants: List[Participant]
) -> Path:

    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / "data"

    data_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = data_directory / "participants.csv"

    records = participants_to_records(
        participants
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    return output_path


if __name__ == "__main__":
    participants = generate_all_participants()

    output_path = save_participants_to_csv(
        participants
    )

    print(
        f"Generated {len(participants)} participants."
    )

    print(
        f"Saved participants to: {output_path}"
    )

    print(
        participants[0]
    )