import pandas as pd
from pathlib import Path
import joblib
import numpy as np

TARGETS = [
    "flare_risk",
    "dizziness_risk",
    "fatigue_risk",
    "fainting_risk",
    "need_to_hydrate",
    "need_to_rest",
]

THRESHOLDS = {
    "flare_risk": 0.55,
    "dizziness_risk": 0.25,
    "fatigue_risk": 0.30,
    "fainting_risk": 0.20,
    "need_to_hydrate": 0.35,
    "need_to_rest": 0.35,
}

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


def load_test_dataset():
    project_root = get_project_root()

    possible_paths = [
        project_root / "data" / "test_dataset.csv",
        project_root / "data" / "test_data.csv",
        project_root / "data" / "training_dataset.csv",
    ]

    for path in possible_paths:
        if path.exists():
            data = pd.read_csv(path)

            if "participant_id" in data.columns:
                return data

    raise FileNotFoundError(
        "Could not find a suitable test dataset."
    )


def prepare_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    return data.drop(columns=excluded)


def load_model(target):
    project_root = get_project_root()

    calibrated_path = (
        project_root
        / "models"
        / f"{target}_calibrated_model.joblib"
    )

    standard_path = (
        project_root
        / "models"
        / f"{target}_model.joblib"
    )

    if calibrated_path.exists():
        return joblib.load(calibrated_path)

    if standard_path.exists():
        return joblib.load(standard_path)

    raise FileNotFoundError(
        f"No model found for target: {target}"
    )


def get_predictions(model, X):
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities


def analyze_target(target, data, X):
    model = load_model(target)

    threshold = THRESHOLDS[target]

    probabilities = get_predictions(model, X)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    actual = data[target].astype(int).to_numpy()

    error_type = np.where(
        (actual == 1) & (predictions == 1),
        "true_positive",
        np.where(
            (actual == 0) & (predictions == 0),
            "true_negative",
            np.where(
                (actual == 0) & (predictions == 1),
                "false_positive",
                "false_negative",
            ),
        ),
    )

    results = data[
        [
            column
            for column in data.columns
            if column not in TARGETS
        ]
    ].copy()

    results["target"] = target
    results["actual"] = actual
    results["predicted"] = predictions
    results["probability"] = probabilities
    results["threshold"] = threshold
    results["error_type"] = error_type

    print()
    print("=" * 70)
    print(f"ERROR ANALYSIS: {target}")
    print("=" * 70)

    counts = pd.Series(error_type).value_counts()

    print()
    print("Prediction counts:")

    for category in [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]:
        print(
            f"{category}: "
            f"{counts.get(category, 0)}"
        )

    print()
    print(
        f"Mean predicted probability: "
        f"{probabilities.mean():.4f}"
    )

    print(
        f"Actual positive rate: "
        f"{actual.mean():.4f}"
    )

    print(
        f"Prediction positive rate: "
        f"{predictions.mean():.4f}"
    )

    print()
    print("False-positive probability summary:")

    false_positive_probs = probabilities[
        (actual == 0) & (predictions == 1)
    ]

    if len(false_positive_probs) > 0:
        print(
            pd.Series(false_positive_probs)
            .describe()
            .round(4)
            .to_string()
        )
    else:
        print("No false positives.")

    print()
    print("False-negative probability summary:")

    false_negative_probs = probabilities[
        (actual == 1) & (predictions == 0)
    ]

    if len(false_negative_probs) > 0:
        print(
            pd.Series(false_negative_probs)
            .describe()
            .round(4)
            .to_string()
        )
    else:
        print("No false negatives.")

    return results


def main():
    print("=" * 70)
    print("PaceMate-AI Model Error Analysis")
    print("=" * 70)

    data = load_test_dataset()

    X = prepare_features(data)

    print()
    print(f"Test rows: {len(data)}")
    print(f"Number of features: {len(X.columns)}")

    all_results = []

    for target in TARGETS:
        results = analyze_target(
            target,
            data,
            X,
        )

        all_results.append(results)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = (
        get_project_root()
        / "results"
        / "model_error_analysis.csv"
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
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
