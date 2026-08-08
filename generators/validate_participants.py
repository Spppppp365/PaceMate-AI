import pandas as pd
from pathlib import Path


def load_participants() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "participants.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Participant dataset not found at: {data_path}"
        )

    return pd.read_csv(data_path)


def validate_participants(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("PaceMate-AI Participant Dataset Validation")
    print("=" * 60)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Participant IDs ---")
    duplicate_ids = df["participant_id"].duplicated().sum()
    print(f"Duplicate IDs: {duplicate_ids}")

    print("\n--- Age Range ---")
    print(f"Minimum age: {df['age'].min()}")
    print(f"Maximum age: {df['age'].max()}")

    print("\n--- Height Range ---")
    print(f"Minimum height: {df['height_cm'].min()} cm")
    print(f"Maximum height: {df['height_cm'].max()} cm")

    print("\n--- Weight Range ---")
    print(f"Minimum weight: {df['weight_kg'].min()} kg")
    print(f"Maximum weight: {df['weight_kg'].max()} kg")

    print("\n--- Baseline Severity Distribution ---")
    print(df["baseline_severity"].value_counts().sort_index())

    print("\n--- Medication Distribution ---")
    print(df["medication_group"].value_counts())

    print("\n--- Biological Sex Distribution ---")
    print(df["biological_sex"].value_counts())

    print("\n--- Boolean Feature Distribution ---")

    boolean_columns = [
        "compression_garments",
        "mobility_aid",
        "athlete",
        "caffeine_sensitive",
    ]

    for column in boolean_columns:
        print(f"\n{column}:")
        print(df[column].value_counts())

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


def main() -> None:
    df = load_participants()
    validate_participants(df)


if __name__ == "__main__":
    main()