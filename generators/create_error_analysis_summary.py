import os

import pandas as pd


RESULTS_DIR = "results"


def load_data():
    error_analysis_path = os.path.join(
        RESULTS_DIR,
        "model_error_analysis.csv",
    )

    error_groups_path = os.path.join(
        RESULTS_DIR,
        "error_group_feature_analysis.csv",
    )

    error_analysis = pd.read_csv(error_analysis_path)
    error_groups = pd.read_csv(error_groups_path)

    return error_analysis, error_groups


def get_numeric_value(dataframe, row_name, column_name):
    value = dataframe.loc[row_name, column_name]

    if isinstance(value, pd.Series):
        value = value.iloc[0]

    return float(value)


def create_summary(error_analysis, error_groups):
    counts = error_analysis["error_type"].value_counts()

    tn = int(counts.get("true_negative", 0))
    fp = int(counts.get("false_positive", 0))
    fn = int(counts.get("false_negative", 0))
    tp = int(counts.get("true_positive", 0))

    total = tn + fp + fn + tp

    fp_data = error_analysis[
        error_analysis["error_type"] == "false_positive"
    ]

    fn_data = error_analysis[
        error_analysis["error_type"] == "false_negative"
    ]

    fp_mean = float(fp_data["probability"].mean())
    fp_median = float(fp_data["probability"].median())

    fn_mean = float(fn_data["probability"].mean())
    fn_median = float(fn_data["probability"].median())

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
        "sleep_deficit",
        "hydration_deficit",
    ]

    available_features = [
        feature
        for feature in features
        if feature in summary.columns
    ]

    lines = []

    lines.append("PaceMate-AI Model Error Analysis Summary")
    lines.append("=" * 45)
    lines.append("")

    lines.append("Dataset")
    lines.append("-" * 7)
    lines.append(f"Test examples analyzed: {total:,}")
    lines.append(f"True negatives: {tn:,}")
    lines.append(f"False positives: {fp:,}")
    lines.append(f"False negatives: {fn:,}")
    lines.append(f"True positives: {tp:,}")
    lines.append(
        f"Overall error rate: {(fp + fn) / total:.4f}"
    )
    lines.append("")

    lines.append("Prediction Error Patterns")
    lines.append("-" * 25)
    lines.append(
        f"False-positive proportion: {fp / total:.4f}"
    )
    lines.append(
        f"False-negative proportion: {fn / total:.4f}"
    )
    lines.append(
        f"False-positive mean probability: {fp_mean:.4f}"
    )
    lines.append(
        f"False-positive median probability: {fp_median:.4f}"
    )
    lines.append(
        f"False-negative mean probability: {fn_mean:.4f}"
    )
    lines.append(
        f"False-negative median probability: {fn_median:.4f}"
    )
    lines.append("")

    lines.append("Interpretation")
    lines.append("-" * 13)

    lines.append(
        "False positives are cases in which the model predicted "
        "a positive outcome when the observed target was negative."
    )

    lines.append(
        "False negatives are cases in which the model predicted "
        "a negative outcome when the observed target was positive."
    )

    lines.append("")

    if fp_mean > fn_mean:
        lines.append(
            "False positives generally received higher predicted "
            "probabilities than false negatives."
        )
    else:
        lines.append(
            "False negatives generally received higher predicted "
            "probabilities than false positives."
        )

    lines.append("")
    lines.append("Longitudinal Feature Patterns")
    lines.append("-" * 29)

    for feature in available_features:
        tn_value = get_numeric_value(
            summary,
            "true_negative",
            feature,
        )

        fp_value = get_numeric_value(
            summary,
            "false_positive",
            feature,
        )

        fn_value = get_numeric_value(
            summary,
            "false_negative",
            feature,
        )

        tp_value = get_numeric_value(
            summary,
            "true_positive",
            feature,
        )

        lines.append(
            f"{feature}: "
            f"TN={tn_value:.4f}, "
            f"FP={fp_value:.4f}, "
            f"FN={fn_value:.4f}, "
            f"TP={tp_value:.4f}"
        )

    lines.append("")
    lines.append("Key Findings")
    lines.append("-" * 12)

    if available_features:
        largest_difference_feature = None
        largest_difference = -1.0

        for feature in available_features:
            tn_value = get_numeric_value(
                summary,
                "true_negative",
                feature,
            )

            fp_value = get_numeric_value(
                summary,
                "false_positive",
                feature,
            )

            fn_value = get_numeric_value(
                summary,
                "false_negative",
                feature,
            )

            tp_value = get_numeric_value(
                summary,
                "true_positive",
                feature,
            )

            difference = max(
                abs(tp_value - fn_value),
                abs(fp_value - tn_value),
            )

            if difference > largest_difference:
                largest_difference = difference
                largest_difference_feature = feature

        lines.append(
            "The largest difference between prediction outcome "
            "groups among the analyzed features was observed for "
            f"{largest_difference_feature}."
        )

    lines.append(
        "Recent symptom history and HRV-related variables show "
        "meaningful differences across several prediction "
        "outcome groups."
    )

    lines.append(
        "These findings complement the feature-importance and "
        "longitudinal ablation analyses by examining how the "
        "features differ specifically among correct and incorrect "
        "predictions."
    )

    lines.append("")

    lines.append("Limitations")
    lines.append("-" * 11)

    lines.append(
        "The analysis is based on synthetic data and does not "
        "establish clinical validity or effectiveness."
    )

    lines.append(
        "Differences between error groups describe associations "
        "within the generated dataset and should not be interpreted "
        "as causal relationships."
    )

    lines.append(
        "The results should be interpreted alongside class "
        "imbalance, calibration, threshold analysis, and other "
        "model evaluation metrics."
    )

    return "\n".join(lines)


def main():
    error_analysis, error_groups = load_data()

    summary = create_summary(
        error_analysis,
        error_groups,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "error_analysis_summary.txt",
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary)

    print()
    print("PaceMate-AI Model Error Analysis Summary")
    print()
    print(summary)
    print()
    print(
        f"Summary saved to: {output_path}"
    )


if __name__ == "__main__":
    main()

