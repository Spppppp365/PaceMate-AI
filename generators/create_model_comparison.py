import os

import pandas as pd


RESULTS_DIR = "results"


def load_results():
    production_path = os.path.join(
        RESULTS_DIR,
        "final_production_model_results.csv",
    )

    threshold_path = os.path.join(
        RESULTS_DIR,
        "final_threshold_evaluation.csv",
    )

    production = pd.read_csv(production_path)
    threshold = pd.read_csv(threshold_path)

    return production, threshold


def create_comparison_table(production, threshold):
    confusion_columns = [
        "target",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    confusion = threshold[confusion_columns]

    comparison = production.merge(
        confusion,
        on="target",
        how="left",
    )

    comparison = comparison[
        [
            "target",
            "threshold",
            "positive_rate",
            "mean_predicted_probability",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]
    ]

    return comparison


def create_summary(comparison):
    strongest_f1 = comparison.loc[
        comparison["f1"].idxmax()
    ]

    strongest_roc_auc = comparison.loc[
        comparison["roc_auc"].idxmax()
    ]

    strongest_pr_auc = comparison.loc[
        comparison["pr_auc"].idxmax()
    ]

    lowest_brier = comparison.loc[
        comparison["brier_score"].idxmin()
    ]

    highest_recall = comparison.loc[
        comparison["recall"].idxmax()
    ]

    highest_precision = comparison.loc[
        comparison["precision"].idxmax()
    ]

    summary = []

    summary.append("PaceMate-AI Final Production Model Comparison")
    summary.append("")
    summary.append(
        f"Strongest F1: {strongest_f1['target']} "
        f"({strongest_f1['f1']:.4f})"
    )
    summary.append(
        f"Strongest ROC-AUC: {strongest_roc_auc['target']} "
        f"({strongest_roc_auc['roc_auc']:.4f})"
    )
    summary.append(
        f"Strongest PR-AUC: {strongest_pr_auc['target']} "
        f"({strongest_pr_auc['pr_auc']:.4f})"
    )
    summary.append(
        f"Lowest Brier score: {lowest_brier['target']} "
        f"({lowest_brier['brier_score']:.4f})"
    )
    summary.append(
        f"Highest recall: {highest_recall['target']} "
        f"({highest_recall['recall']:.4f})"
    )
    summary.append(
        f"Highest precision: {highest_precision['target']} "
        f"({highest_precision['precision']:.4f})"
    )

    summary.append("")
    summary.append("Target-level observations:")

    for _, row in comparison.iterrows():
        summary.append(
            f"{row['target']}: "
            f"F1={row['f1']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, "
            f"Brier={row['brier_score']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}"
        )

    return "\n".join(summary)


def main():
    production, threshold = load_results()

    comparison = create_comparison_table(
        production,
        threshold,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "final_model_comparison.csv",
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    summary = create_summary(comparison)

    summary_path = os.path.join(
        RESULTS_DIR,
        "final_model_comparison_summary.txt",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(summary)

    print()
    print("PaceMate-AI Final Production Model Comparison")
    print()
    print(comparison.to_string(index=False))
    print()
    print("Summary")
    print()
    print(summary)
    print()
    print(f"Comparison saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print()
    print("Model comparison generation complete.")


if __name__ == "__main__":
    main()