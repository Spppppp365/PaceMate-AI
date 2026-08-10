import pandas as pd
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
)


TARGET_COLUMN = "need_to_hydrate"

RANDOM_SEED = 42

N_ESTIMATORS = 300
MAX_DEPTH = 16
MIN_SAMPLES_LEAF = 5

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


def load_dataset(filename):
    path = get_project_root() / "data" / filename

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


def create_base_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate_model(model, X_test, y_test, name):
    probabilities = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print()
    print(f"Model: {name}")
    print(f"Brier score: {brier:.6f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")

    return {
        "model": name,
        "brier_score": brier,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():
    print("=" * 70)
    print("PaceMate-AI Hydration-Risk Calibration Experiment")
    print("=" * 70)

    train_data = load_dataset("train_dataset.csv")
    test_data = load_dataset("test_dataset.csv")

    X_train = prepare_features(train_data)
    y_train = train_data[TARGET_COLUMN]

    X_test = prepare_features(test_data)
    y_test = test_data[TARGET_COLUMN]

    print()
    print(f"Training rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Test positive cases: {y_test.sum()}")
    print(f"Test positive rate: {y_test.mean():.4f}")

    results = []

    print()
    print("Training base Random Forest...")

    base_model = create_base_model()

    base_model.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            base_model,
            X_test,
            y_test,
            "base_random_forest",
        )
    )

    print()
    print("Training sigmoid calibration...")

    sigmoid_model = CalibratedClassifierCV(
        estimator=create_base_model(),
        method="sigmoid",
        cv=5,
    )

    sigmoid_model.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            sigmoid_model,
            X_test,
            y_test,
            "sigmoid",
        )
    )

    print()
    print("Training isotonic calibration...")

    isotonic_model = CalibratedClassifierCV(
        estimator=create_base_model(),
        method="isotonic",
        cv=5,
    )

    isotonic_model.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            isotonic_model,
            X_test,
            y_test,
            "isotonic",
        )
    )

    results_dataframe = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("HYDRATION CALIBRATION COMPARISON")
    print("=" * 70)

    print(
        results_dataframe.round(6).to_string(
            index=False
        )
    )

    output_path = (
        get_project_root()
        / "results"
        / "hydration_calibration_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    print()
    print("=" * 70)
    print("HYDRATION CALIBRATION EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
