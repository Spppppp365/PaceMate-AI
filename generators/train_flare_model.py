import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


TARGET_COLUMN = "flare_risk"

RANDOM_SEED = 42

N_ESTIMATORS = 300
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 5


EXCLUDED_COLUMNS = [
    "participant_id",
    "day",

    # Current-day outcomes that are too closely tied
    # to the synthetic target construction.
    "dizziness",
    "fatigue",
    "brain_fog",
    "symptom_severity",

    # All target columns except the target being trained.
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


def prepare_features(data):
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in data.columns
    ]

    features = data.drop(
        columns=excluded
    )

    return features


def prepare_target(data):
    return data[TARGET_COLUMN]


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("FLARE RISK MODEL RESULTS")
    print("=" * 60)

    print()
    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")

    print(matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def show_feature_importance(
    model,
    feature_names
):

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("=" * 60)
    print("TOP FEATURE IMPORTANCES")
    print("=" * 60)

    print(
        importance.head(15).to_string(
            index=False
        )
    )


def main():

    print("=" * 60)
    print(
        "PaceMate-AI Flare Risk Model"
    )
    print("=" * 60)

    train_data = load_dataset(
        "train_dataset.csv"
    )

    validation_data = load_dataset(
        "validation_dataset.csv"
    )

    test_data = load_dataset(
        "test_dataset.csv"
    )

    print()
    print(
        f"Training rows: {len(train_data)}"
    )

    print(
        f"Validation rows: {len(validation_data)}"
    )

    print(
        f"Test rows: {len(test_data)}"
    )

    X_train = prepare_features(
        train_data
    )

    X_validation = prepare_features(
        validation_data
    )

    X_test = prepare_features(
        test_data
    )

    y_train = prepare_target(
        train_data
    )

    y_validation = prepare_target(
        validation_data
    )

    y_test = prepare_target(
        test_data
    )

    print()
    print(
        f"Number of features: {X_train.shape[1]}"
    )

    print()
    print(
        "Training model..."
    )

    model = train_model(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    print()
    print(
        "Evaluating on validation set..."
    )

    evaluate_model(
        model,
        X_validation,
        y_validation
    )

    print()
    print(
        "Evaluating on unseen test participants..."
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model,
        X_train.columns
    )


if __name__ == "__main__":
    main()
