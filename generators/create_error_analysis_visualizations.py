import os

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_data():
    error_analysis_path = os.path.join(
        RESULTS_DIR, "model_error_analysis.csv"
    )

    error_groups_path = os.path.join(
        RESULTS_DIR, "error_group_feature_analysis.csv"
    )

    error_analysis = pd.read_csv(error_analysis_path)
    error_groups = pd.read_csv(error_groups_path)

    return error_analysis, error_groups


def create_error_count_plot(error_analysis):
    error_types = [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    counts = error_analysis["error_type"].value_counts()

    values = [counts.get(error_type, 0) for error_type in error_types]

    labels = [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.ylabel("Number of Test Examples")
    plt.title("Prediction Outcomes Across Test Examples")
    plt.xticks(rotation=15)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "prediction_outcomes.png",
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def create_error_probability_plot(error_analysis):
    groups = [
        "false_positive",
        "false_negative",
    ]

    data = []

    for group in groups:
        group_data = error_analysis[
            error_analysis["error_type"] == group
        ]

        data.append(group_data["probability"].values)

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data,
        tick_labels=[
            "False Positives",
            "False Negatives",
        ],
    )

    plt.ylabel("Predicted Probability")
    plt.title("Predicted Probability Distribution for Model Errors")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "error_probability_distribution.png",
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def create_feature_comparison_plot(error_groups):
    numeric_features = [
        column
        for column in error_groups.columns
        if column != "error_type"
    ]

    summary = error_groups.set_index("error_type")

    selected_features = [
        "previous_symptom_severity",
        "symptom_3day_mean",
        "symptom_7day_mean",
        "previous_hrv",
        "hrv_3day_mean",
        "hrv_7day_mean",
    ]

    selected_features = [
        feature
        for feature in selected_features
        if feature in numeric_features
    ]

    plot_data = summary.loc[
        [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
        selected_features,
    ]

    plot_data = plot_data.T

    plt.figure(figsize=(12, 7))
    plot_data.plot(kind="bar", ax=plt.gca())

    plt.ylabel("Mean Feature Value")
    plt.xlabel("Feature")
    plt.title("Feature Differences Across Prediction Outcome Groups")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prediction Outcome")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "error_group_feature_comparison.png",
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def create_longitudinal_feature_plot(error_groups):
    summary = error_groups.set_index("error_type")

    features = [
        "previous_symptom_severity",
        "symptom_3day_mean",
        "symptom_7day_mean",
        "previous_hrv",
        "hrv_3day_mean",
        "hrv_7day_mean",
        "symptom_change",
        "hrv_change",
    ]

    features = [
        feature
        for feature in features
        if feature in summary.columns
    ]

    comparison = summary.loc[
        [
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
        features,
    ]

    comparison = comparison.T

    plt.figure(figsize=(12, 7))
    comparison.plot(kind="bar", ax=plt.gca())

    plt.ylabel("Mean Feature Value")
    plt.xlabel("Longitudinal Feature")
    plt.title("Longitudinal Feature Patterns Across Prediction Outcomes")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Prediction Outcome")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "longitudinal_error_feature_patterns.png",
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    error_analysis, error_groups = load_data()

    create_error_count_plot(error_analysis)
    create_error_probability_plot(error_analysis)
    create_feature_comparison_plot(error_groups)
    create_longitudinal_feature_plot(error_groups)

    print()
    print("PaceMate-AI Error Analysis Visualizations")
    print()
    print("Created figures:")

    print(
        os.path.join(
            FIGURES_DIR,
            "prediction_outcomes.png",
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "error_probability_distribution.png",
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "error_group_feature_comparison.png",
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "longitudinal_error_feature_patterns.png",
        )
    )

    print()
    print("Visualization generation complete.")


if __name__ == "__main__":
    main()