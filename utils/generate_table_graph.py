"""Generate a grouped bar chart from the RAG evaluation table in Untitled-2."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS = [
    "Precision",
    "Recall",
    "Entity Recall",
    "Faithfulness",
    "Sensitivity",
    "Relevancy",
    "Correctness",
]

RESULTS = {
    "Fixed-Length": [0.5972, 0.8056, 0.4306, 1.0, 0.5816, 0.8768, 0.5932],
    "Recursive": [0.6194, 0.75, 0.5139, 0.9924, 0.4717, 0.7974, 0.5205],
    "Semantic": [0.5648, 0.7014, 0.3889, 0.9948, 0.6538, 0.7883, 0.4351],
    "Agentic": [0.4755, 0.4792, 0.4722, 0.9722, 0.2992, 0.8292, 0.5503],
    "Agentic-Enriched": [0.7181, 0.8056, 0.5139, 1.0, 0.5912, 0.869, 0.5883],
    "PageIndex": [1.0, 1.0, 0.5278, 1.0, 0.5899, 0.8871, 0.6186],
    "Hierarchical": [0.7706, 0.9167, 0.5694, 1.0, 0.6377, 0.8722, 0.5258],
}


def generate_graph(output_path: Path | None = None) -> Path:
    """Create and save the evaluation bar chart, returning its output path."""
    output_path = output_path or Path(__file__).resolve().parents[1] / "graphs" / (
        "rag_metrics_chunking_strategies.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = pd.DataFrame.from_dict(RESULTS, orient="index", columns=METRICS)
    plot_data = scores.reset_index(names="Strategy").melt(
        id_vars="Strategy", var_name="Metric", value_name="Score"
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = {
        "Fixed-Length": "#2b5c8f",
        "Recursive": "#d95f02",
        "Semantic": "#7570b3",
        "Agentic": "#1b9e77",
        "Agentic-Enriched": "#e7298a",
        "PageIndex": "#66a61e",
        "Hierarchical": "#a61e1e",
    }
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=plot_data,
        x="Metric",
        y="Score",
        hue="Strategy",
        hue_order=list(RESULTS),
        palette=palette,
        ax=ax,
    )

    ax.set_title(
        "RAG Metrics Evaluation - Chunking Strategy Comparison",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("")
    ax.legend(
        title="Strategy",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        borderaxespad=0.0,
    )

    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=7,
            xytext=(0, 2),
            textcoords="offset points",
        )

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    print(f"Generated graph: {generate_graph()}")