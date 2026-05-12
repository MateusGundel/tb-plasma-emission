"""Funções de plotagem LaTeX-ready para publicação."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


# Config global para figuras de publicação
rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def save_fig(fig, path, formats=("pdf", "png")):
    """Salva figura em múltiplos formatos."""
    base = os.path.splitext(path)[0]
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}")
    plt.close(fig)


def plot_baseline_comparison(results_path, figures_dir):
    """Compara modelos baseline (bar chart com erro)."""
    with open(results_path) as f:
        results = json.load(f)

    models = list(results.keys())
    metrics_names = ["accuracy", "sensitivity", "specificity", "f1", "auc"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_names))
    width = 0.25
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, model_name in enumerate(models):
        means = [results[model_name]["avg_metrics"].get(m, {}).get("mean", 0)
                 for m in metrics_names]
        stds = [results[model_name]["avg_metrics"].get(m, {}).get("std", 0)
                for m in metrics_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=model_name,
               color=colors[i], edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xlabel("Métrica")
    ax.set_ylabel("Valor")
    ax.set_title("Comparação de classificadores baseline (5-fold CV)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics_names])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, os.path.join(figures_dir, "fig_baseline_comparison"))
    print("  ✓ fig_baseline_comparison")


def plot_cnn_comparison(results_path, figures_dir):
    """Compara arquiteturas CNN (bar chart com erro)."""
    with open(results_path) as f:
        results = json.load(f)

    models = list(results.keys())
    metrics_names = ["accuracy", "sensitivity", "specificity", "f1", "auc"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics_names))
    width = 0.25
    colors = ["#9b59b6", "#f39c12", "#1abc9c"]

    for i, model_name in enumerate(models):
        means = [results[model_name]["avg_metrics"].get(m, {}).get("mean", 0)
                 for m in metrics_names]
        stds = [results[model_name]["avg_metrics"].get(m, {}).get("std", 0)
                for m in metrics_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=model_name,
               color=colors[i % len(colors)], edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xlabel("Métrica")
    ax.set_ylabel("Valor")
    ax.set_title("Comparação de arquiteturas CNN (5-fold CV)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics_names])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, os.path.join(figures_dir, "fig_cnn_comparison"))
    print("  ✓ fig_cnn_comparison")


def plot_confusion_matrix_best(y_true, y_pred, model_name, figures_dir):
    """Matriz de confusão do melhor modelo."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negativo", "Positivo"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matriz de confusão — {model_name}")

    save_fig(fig, os.path.join(figures_dir, "fig_confusion_matrix_best"))
    print("  ✓ fig_confusion_matrix_best")


def plot_roc_curves(results_dict, figures_dir, title="Curvas ROC"):
    """Curvas ROC de múltiplos modelos."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for i, (name, data) in enumerate(results_dict.items()):
        if "test_probs" in data and "test_labels" in data:
            fpr, tpr, _ = roc_curve(data["test_labels"], data["test_probs"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    save_fig(fig, os.path.join(figures_dir, "fig_roc_curves"))
    print("  ✓ fig_roc_curves")


def plot_learning_curves(histories, arch_name, figures_dir):
    """Curvas de aprendizado (loss e AUC por época)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for fold_i, history in enumerate(histories):
        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], alpha=0.5, label=f"Fold {fold_i} train")
        ax1.plot(epochs, history["val_loss"], "--", alpha=0.5, label=f"Fold {fold_i} val")

        ax2.plot(epochs, history["val_auc"], alpha=0.7, label=f"Fold {fold_i}")

    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Learning curves — {arch_name}")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Época")
    ax2.set_ylabel("AUC (validação)")
    ax2.set_title(f"AUC por fold — {arch_name}")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    save_fig(fig, os.path.join(figures_dir, f"fig_learning_curves_{arch_name}"))
    print(f"  ✓ fig_learning_curves_{arch_name}")


def plot_fold_stability(results, figures_dir, title="Estabilidade entre folds"):
    """Box plot da variação entre folds para cada modelo."""
    fig, ax = plt.subplots(figsize=(8, 4))

    data = []
    labels = []
    for model_name, result in results.items():
        aucs = [m.get("auc", 0) for m in result["fold_metrics"]]
        data.append(aucs)
        labels.append(model_name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("AUC")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    save_fig(fig, os.path.join(figures_dir, "fig_fold_stability"))
    print("  ✓ fig_fold_stability")
