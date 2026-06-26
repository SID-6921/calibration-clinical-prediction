"""
Generate all publication figures for the calibration study.
Run after run_experiments.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

MODEL_COLORS = {
    "Logistic Regression": "#2196F3",
    "Decision Tree": "#4CAF50",
    "Random Forest": "#8BC34A",
    "Gradient Boosting": "#9C27B0",
    "SVM (RBF)": "#FF5722",
    "MLP": "#FF9800",
}

CAL_COLORS = {
    "Uncalibrated": "#EF5350",
    "Platt": "#42A5F5",
    "Isotonic": "#66BB6A",
    "Beta": "#AB47BC",
}


def fig1_ece_heatmap():
    """Figure 1: Heatmap of ECE across datasets x models x calibrators."""
    df = pd.read_csv(RESULTS_DIR / "all_results.csv")

    calibrators = ["Uncalibrated", "Platt", "Isotonic", "Beta"]
    datasets = df["dataset"].unique()
    models = df["model"].unique()

    fig, axes = plt.subplots(1, len(calibrators), figsize=(16, 5), sharey=True)

    for ax_idx, cal in enumerate(calibrators):
        ax = axes[ax_idx]
        subset = df[df["calibrator"] == cal]
        pivot = subset.pivot_table(index="model", columns="dataset",
                                   values="ece", aggfunc="mean")
        pivot = pivot.reindex(index=models, columns=datasets)

        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=0.2)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d.replace(" ", "\n") for d in datasets],
                           fontsize=8, rotation=45, ha="right")
        if ax_idx == 0:
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=9)
        else:
            ax.set_yticks([])
        ax.set_title(cal, fontsize=11, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(datasets)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if val > 0.1 else "black")

    fig.colorbar(im, ax=axes, shrink=0.8, label="ECE (lower is better)")
    fig.suptitle("Figure 1: Expected Calibration Error Across Datasets, Models, and Calibrators",
                 y=1.05, fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_ece_heatmap.png")
    plt.savefig(FIGURES_DIR / "fig1_ece_heatmap.pdf")
    plt.close()
    print("  -> fig1_ece_heatmap")


def fig2_verdict_summary():
    """Figure 2: Stacked bar chart of HELPS/HURTS/NO_EFFECT verdicts."""
    deg = pd.read_csv(RESULTS_DIR / "degradation_analysis.csv")
    if len(deg) == 0:
        print("  -> fig2: no degradation data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # By calibrator
    for ax, group_col, title in [
        (ax1, "calibrator", "By Calibration Method"),
        (ax2, "model", "By Model Family"),
    ]:
        verdicts = ["HELPS", "NO_EFFECT", "HURTS"]
        colors_v = {"HELPS": "#4CAF50", "NO_EFFECT": "#9E9E9E", "HURTS": "#F44336"}
        groups = deg[group_col].unique()
        x = np.arange(len(groups))
        bottoms = np.zeros(len(groups))

        for v in verdicts:
            counts = []
            for g in groups:
                c = len(deg[(deg[group_col] == g) & (deg["verdict"] == v)])
                counts.append(c)
            ax.bar(x, counts, bottom=bottoms, label=v,
                   color=colors_v[v], alpha=0.85, width=0.6)
            bottoms += counts

        ax.set_xticks(x)
        ax.set_xticklabels([g.replace(" ", "\n") for g in groups], fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Figure 2: When Does Post-hoc Calibration Help vs. Hurt? (p<0.05, paired t-test)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_verdict_summary.png")
    plt.savefig(FIGURES_DIR / "fig2_verdict_summary.pdf")
    plt.close()
    print("  -> fig2_verdict_summary")


def fig3_ece_change_scatter():
    """Figure 3: Scatter plot of ECE change vs dataset size and prevalence."""
    deg = pd.read_csv(RESULTS_DIR / "degradation_analysis.csv")
    df = pd.read_csv(RESULTS_DIR / "all_results.csv")
    if len(deg) == 0:
        print("  -> fig3: no data")
        return

    # Merge dataset metadata
    meta = df.groupby("dataset").agg(
        n_samples=("n_samples", "first"),
        prevalence=("prevalence", "first"),
    ).reset_index()
    deg = deg.merge(meta, on="dataset")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    verdict_colors = {"HELPS": "#4CAF50", "NO_EFFECT": "#9E9E9E", "HURTS": "#F44336"}

    for ax, x_col, xlabel in [
        (ax1, "n_samples", "Dataset Size (n)"),
        (ax2, "prevalence", "Class Prevalence"),
    ]:
        for v, color in verdict_colors.items():
            subset = deg[deg["verdict"] == v]
            ax.scatter(subset[x_col], subset["ece_pct_change"],
                       c=color, label=v, alpha=0.7, s=40, edgecolors="white", linewidth=0.5)

        ax.axhline(0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECE Change (%)")
        ax.legend(fontsize=9)

    fig.suptitle("Figure 3: Calibration Effect vs. Dataset Characteristics", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_ece_scatter.png")
    plt.savefig(FIGURES_DIR / "fig3_ece_scatter.pdf")
    plt.close()
    print("  -> fig3_ece_scatter")


def fig4_auc_vs_ece():
    """Figure 4: AUC vs ECE scatter -- are they correlated?"""
    df = pd.read_csv(RESULTS_DIR / "all_results.csv")
    summary = df.groupby(["dataset", "model", "calibrator"]).agg(
        auc=("auc", "mean"), ece=("ece", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))

    for cal, color in CAL_COLORS.items():
        sub = summary[summary["calibrator"] == cal]
        ax.scatter(sub["auc"], sub["ece"], c=color, label=cal,
                   alpha=0.7, s=50, edgecolors="white", linewidth=0.5)

    from scipy.stats import pearsonr
    r, p = pearsonr(summary["auc"], summary["ece"])
    ax.set_xlabel("AUC-ROC (Discrimination)")
    ax.set_ylabel("ECE (Calibration Error)")
    ax.set_title(f"Figure 4: Discrimination vs. Calibration (r={r:.3f}, p={p:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_auc_vs_ece.png")
    plt.savefig(FIGURES_DIR / "fig4_auc_vs_ece.pdf")
    plt.close()
    print("  -> fig4_auc_vs_ece")


def fig5_decision_flowchart():
    """Figure 5: Decision flowchart for calibration method selection."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#E3F2FD", edge="#1565C0"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=8, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my, text, fontsize=7, color="#666")

    box(3.5, 7, 3, 0.7, "START:\nTrained model with\npredicted probabilities", "#BBDEFB", "#1565C0")
    box(3.5, 5.5, 3, 0.7, "Is n_calibration > 1000?", "#FFF9C4", "#F9A825")
    arrow(5, 7, 5, 6.2)

    box(0.5, 4, 2.5, 0.7, "Use Platt Scaling\n(low-variance)", "#C8E6C9", "#2E7D32")
    box(7, 4, 2.5, 0.7, "Is the model\ntree-based/ensemble?", "#FFF9C4", "#F9A825")
    arrow(3.5, 5.5, 1.75, 4.7, "No")
    arrow(6.5, 5.5, 8.25, 4.7, "Yes")

    box(5.5, 2.5, 2.5, 0.7, "Use Platt or Beta\n(avoid Isotonic)", "#FFCCBC", "#D84315")
    box(8.5, 2.5, 1.5, 0.7, "Use Isotonic", "#C8E6C9", "#2E7D32")
    arrow(7, 4, 6.75, 3.2, "Yes")
    arrow(9.5, 4, 9.25, 3.2, "No")

    box(2.5, 1, 5, 0.7, "ALWAYS: Validate calibration on held-out data\n"
        "Report ECE + reliability diagram alongside AUC",
        "#E1BEE7", "#6A1B9A")
    arrow(1.75, 4, 5, 1.7)
    arrow(6.75, 2.5, 5, 1.7)
    arrow(9.25, 2.5, 7.5, 1.7)

    ax.set_title("Figure 5: Decision Flowchart for Calibration Method Selection",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.savefig(FIGURES_DIR / "fig5_decision_flowchart.png")
    plt.savefig(FIGURES_DIR / "fig5_decision_flowchart.pdf")
    plt.close()
    print("  -> fig5_decision_flowchart")


def table_main():
    """Generate main results table as CSV."""
    df = pd.read_csv(RESULTS_DIR / "all_results.csv")
    summary = df.groupby(["dataset", "model", "calibrator"]).agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
    ).round(4).reset_index()
    summary.to_csv(FIGURES_DIR / "table_main_results.csv", index=False)
    print("  -> table_main_results.csv")


if __name__ == "__main__":
    print("Generating figures and tables...\n")
    fig1_ece_heatmap()
    fig2_verdict_summary()
    fig3_ece_change_scatter()
    fig4_auc_vs_ece()
    fig5_decision_flowchart()
    table_main()
    print("\n[OK] All figures saved to figures/")
