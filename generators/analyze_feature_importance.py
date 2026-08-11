import pandas as pd
from pathlib import Path
import joblib


TARGET_COLUMNS = [
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
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


def get_project_root():
    return Path(__file__).resolve().parent.parent


def load_dataset():
    path = (
        get_project_root()
        / "data"
        / "train_dataset.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def prepare_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    return data.drop(columns=excluded)


def load_model(target):
    path = (
        get_project_root()
        / "models"
        / f"{target}_model.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    return joblib.load(path)


def analyze_target(
    target,
    feature_names,
    model,
):
    importances = model.feature_importances_

    results = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    results = results.sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("=" * 70)
    print(f"FEATURE IMPORTANCE: {target}")
    print("=" * 70)

    print(
        results.head(15)
        .round(5)
        .to_string(index=False)
    )

    return results


def main():
    print("=" * 70)
    print("PaceMate-AI Feature Importance Analysis")
    print("=" * 70)

    data = load_dataset()

    X = prepare_features(data)

    feature_names = list(X.columns)

    print()
    print(f"Number of features: {len(feature_names)}")

    all_results = []

    for target in TARGET_COLUMNS:
        model = load_model(target)

        results = analyze_target(
            target,
            feature_names,
            model,
        )

        results["target"] = target

        all_results.append(results)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        get_project_root()
        / "results"
        / "secondary_feature_importance.csv"
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
    print(f"Results saved to: {output_path}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
