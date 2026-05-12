#!/usr/bin/env python3
"""Script 06: Análise Grad-CAM — gera heatmaps de TP, TN, FP, FN."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.reproducibility import setup_experiment, set_seed
from src.data.patient_splitter import patient_split
from src.data.torch_dataset import PlasmaDataset
from src.models.cnn_models import create_cnn_model
from src.evaluation.gradcam import GradCAM, overlay_heatmap


def main():
    config, device = setup_experiment()
    results_dir = config["paths"]["results_dir"]
    figures_dir = config["paths"]["figures_dir"]

    # Carregar registry e fazer split
    registry_path = os.path.join(results_dir, "patient_registry_filtered.csv")
    df = pd.read_csv(registry_path)
    df_trainval, df_test, folds = patient_split(
        df, n_folds=config["splitting"]["n_folds"],
        test_size=config["splitting"]["test_size"],
        seed=config["splitting"]["random_seed"],
        dataset="PEVA",
    )

    # Carregar melhor modelo (MobileNetV2 teve melhor AUC no hold-out)
    best_arch = "mobilenetv2"
    ckpt_path = os.path.join(results_dir, "checkpoints", f"{best_arch}_final.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint não encontrado: {ckpt_path}")
        print("Execute scripts/05_run_cnn.py primeiro.")
        return

    model = create_cnn_model(best_arch, num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model)

    # Criar dataset de teste
    test_ds = PlasmaDataset(df_test, target_size=config["preprocessing"]["target_size"])

    # Classificar todas as imagens do teste
    categories = {"TP": [], "TN": [], "FP": [], "FN": []}

    for idx in range(len(test_ds)):
        img_tensor, label = test_ds[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = probs[0, pred].item()

        true_label = label.item()
        if pred == 1 and true_label == 1:
            cat = "TP"
        elif pred == 0 and true_label == 0:
            cat = "TN"
        elif pred == 1 and true_label == 0:
            cat = "FP"
        else:
            cat = "FN"

        categories[cat].append({
            "idx": idx,
            "confidence": conf,
            "image_path": df_test.iloc[idx]["image_path"],
        })

    print("\n=== Distribuição de predições ===")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)}")

    # Gerar galeria Grad-CAM: 2 exemplos de cada categoria
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    row_labels = ["TP", "TN", "FP", "FN"]

    for row, cat in enumerate(row_labels):
        items = categories[cat]
        if not items:
            for col in range(4):
                axes[row, col].text(0.5, 0.5, f"Sem {cat}", ha="center", va="center")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        # Pegar até 2 exemplos
        samples = items[:2]
        for j, sample in enumerate(samples):
            idx = sample["idx"]
            img_tensor, _ = test_ds[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Imagem original
            img_raw = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
            img_raw = cv2.resize(img_raw, (224, 224))

            # Grad-CAM
            heatmap, pred_class, confidence = gradcam.generate(img_tensor)
            overlay = overlay_heatmap(img_raw, heatmap)

            # Plot original
            axes[row, j * 2].imshow(img_raw, cmap="gray")
            axes[row, j * 2].set_title(f"{cat} (conf={confidence:.2f})", fontsize=9)
            axes[row, j * 2].set_xticks([])
            axes[row, j * 2].set_yticks([])

            # Plot Grad-CAM
            axes[row, j * 2 + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[row, j * 2 + 1].set_title("Grad-CAM", fontsize=9)
            axes[row, j * 2 + 1].set_xticks([])
            axes[row, j * 2 + 1].set_yticks([])

        # Preencher colunas vazias se menos de 2 exemplos
        if len(samples) < 2:
            for col in range(len(samples) * 2, 4):
                axes[row, col].axis("off")

        axes[row, 0].set_ylabel(cat, fontsize=12, fontweight="bold", rotation=0,
                                 labelpad=30, va="center")

    plt.suptitle(f"Grad-CAM Analysis — {best_arch}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_gradcam_gallery.pdf"), dpi=300)
    plt.savefig(os.path.join(figures_dir, "fig_gradcam_gallery.png"), dpi=300)
    plt.close()
    print(f"\n✓ Galeria Grad-CAM salva em: {figures_dir}")


if __name__ == "__main__":
    main()
