#!/usr/bin/env python3
"""Script 08: Gera todas as figuras finais para publicação."""

import sys
import os
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.reproducibility import load_config
from src.evaluation.plots import (
    plot_baseline_comparison, plot_cnn_comparison,
    plot_fold_stability, plot_learning_curves,
)


def main():
    config = load_config()
    figures_dir = config["paths"]["figures_dir"]
    logs_dir = config["paths"]["logs_dir"]
    article_figures_dir = config["paths"]["article_figures_dir"]

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(article_figures_dir, exist_ok=True)

    print("=== Gerando figuras de publicação ===\n")

    # Baseline comparison
    baseline_path = os.path.join(logs_dir, "baseline_results.json")
    if os.path.exists(baseline_path):
        plot_baseline_comparison(baseline_path, figures_dir)

        # Fold stability - baseline
        with open(baseline_path) as f:
            baseline_results = json.load(f)
        plot_fold_stability(baseline_results, figures_dir,
                           title="Estabilidade entre folds — Baseline")
    else:
        print("  ⚠ baseline_results.json não encontrado")

    # CNN comparison
    cnn_path = os.path.join(logs_dir, "cnn_results.json")
    if os.path.exists(cnn_path):
        plot_cnn_comparison(cnn_path, figures_dir)

        with open(cnn_path) as f:
            cnn_results = json.load(f)

        # Learning curves por arquitetura
        for arch_name, result in cnn_results.items():
            if "fold_histories" in result:
                plot_learning_curves(result["fold_histories"], arch_name, figures_dir)

        # Fold stability - CNN
        plot_fold_stability(cnn_results, figures_dir,
                           title="Estabilidade entre folds — CNN")
    else:
        print("  ⚠ cnn_results.json não encontrado")

    # Copiar figuras para pasta do artigo
    print(f"\nCopiando figuras para: {article_figures_dir}")
    copied = 0
    for fname in os.listdir(figures_dir):
        if fname.endswith(".pdf"):
            src = os.path.join(figures_dir, fname)
            dst = os.path.join(article_figures_dir, fname)
            shutil.copy2(src, dst)
            copied += 1
    print(f"  {copied} figuras PDF copiadas")

    print("\nFiguras geradas com sucesso!")


if __name__ == "__main__":
    main()
