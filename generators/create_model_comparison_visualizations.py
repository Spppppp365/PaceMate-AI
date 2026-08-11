import os

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_comparison():
    path = os.path.join(
        RESULTS_DIR,
        "final_model_comparison.csv",
    )

    return pd.read_csv(path)


def create_discrimination_plot(data):
    targets = data["target"]

    x = range(len(targets))

    plt.figure(figsize=(12, 7))

    plt.bar(
        [value - 0.2 for value in x],
        data["roc_auc"],
        width=0.4,
        label="ROC-AUC",
    )

    plt.bar(
        [value + 0.2 for value in x],
        data["pr_auc"],
        width=0.4,
        label="PR-AUC",
    )

    plt.xticks(
        list(x),
        targets,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Score")
    plt.xlabel("Prediction Target")
    plt.title(
        "Discrimination Performance Across PaceMate-AI Models"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_discrimination_comparison.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def create_classification_plot(data):
    targets = data["target"]

    x = range(len(targets))

    plt.figure(figsize=(12, 7))

    plt.bar(
        [value - 0.3 for value in x],
        data["precision"],
        width=0.2,
        label="Precision",
    )

    plt.bar(
        x,
        data["recall"],
        width=0.2,
        label="Recall",
    )

    plt.bar(
        [value + 0.3 for value in x],
        data["f1"],
        width=0.2,
        label="F1 Score",
    )

    plt.xticks(
        list(x),
        targets,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Score")
    plt.xlabel("Prediction Target")
    plt.title(
        "Precision, Recall, and F1 Across PaceMate-AI Models"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_classification_metrics.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def create_brier_plot(data):
    targets = data["target"]

    plt.figure(figsize=(12, 7))

    plt.bar(
        targets,
        data["brier_score"],
    )

    plt.ylabel("Brier Score")
    plt.xlabel("Prediction Target")
    plt.title(
        "Probability Calibration Error Across PaceMate-AI Models"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_brier_score_comparison.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def create_probability_alignment_plot(data):
    targets = data["target"]

    x = range(len(targets))

    plt.figure(figsize=(12, 7))

    plt.bar(
        [value - 0.2 for value in x],
        data["positive_rate"],
        width=0.4,
        label="Observed Positive Rate",
    )

    plt.bar(
        [value + 0.2 for value in x],
        data["mean_predicted_probability"],
        width=0.4,
        label="Mean Predicted Probability",
    )

    plt.xticks(
        list(x),
        targets,
        rotation=25,
        ha="right",
    )

    plt.ylabel("Rate")
    plt.xlabel("Prediction Target")
    plt.title(
        "Observed Positive Rate vs Mean Predicted Probability"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "model_probability_alignment.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def main():
    os.makedirs(
        FIGURES_DIR,
        exist_ok=True,
    )

    data = load_comparison()

    create_discrimination_plot(data)
    create_classification_plot(data)
    create_brier_plot(data)
    create_probability_alignment_plot(data)

    print()
    print(
        "PaceMate-AI Model Comparison Visualizations"
    )
    print()
    print("Created figures:")

    print(
        os.path.join(
            FIGURES_DIR,
            "model_discrimination_comparison.png",
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "model_classification_metrics.png",
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "model_brier_score_comparison.png",
        )
    )

    print(
        os.path.join(
            FIGURES_DIR,
            "model_probability_alignment.png",
        )
    )

    print()
    print(
        "Visualization generation complete."
    )


if __name__ == "__main__":
    main()