#!/usr/bin/env python3
"""Script 02: Análise exploratória do dataset — gera figuras para seção 4.1."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

from src.utils.reproducibility import load_config

# Configuração de figuras LaTeX-ready
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


def load_registry(config):
    path = os.path.join(config["paths"]["results_dir"], "patient_registry.csv")
    return pd.read_csv(path)


def fig_class_distribution(df, figures_dir):
    """Distribuição de classes por dataset (bar chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, ds in zip(axes, ["PEVA", "PESC"]):
        sub = df[df["dataset"] == ds]
        counts = sub["label_name"].value_counts()
        colors = ["#2ecc71" if l == "POSITIVE" else "#e74c3c" for l in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{ds}")
        ax.set_ylabel("Número de imagens")
        ratio = counts.min() / counts.max()
        ax.text(0.95, 0.95, f"Razão: 1:{counts.max()//counts.min()}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.suptitle("Distribuição de classes", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_class_distribution.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig_class_distribution.png"))
    plt.close()
    print("  ✓ fig_class_distribution")


def fig_images_per_patient(df, figures_dir):
    """Histograma de imagens por paciente."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, ds in zip(axes, ["PEVA", "PESC"]):
        sub = df[df["dataset"] == ds]
        imgs_per_patient = sub.groupby(["patient_id", "label_name"]).size().reset_index(name="count")

        for label, color in [("POSITIVE", "#2ecc71"), ("NEGATIVE", "#e74c3c")]:
            data = imgs_per_patient[imgs_per_patient["label_name"] == label]["count"]
            if len(data) > 0:
                ax.hist(data, bins=20, alpha=0.7, color=color, edgecolor="black",
                        linewidth=0.5, label=f"{label} (n={len(data)})")

        ax.set_title(f"{ds}")
        ax.set_xlabel("Imagens por paciente")
        ax.set_ylabel("Frequência")
        ax.legend()

    plt.suptitle("Distribuição de imagens por paciente", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_images_per_patient.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig_images_per_patient.png"))
    plt.close()
    print("  ✓ fig_images_per_patient")


def fig_sample_images(df, figures_dir, n=3):
    """Grid de exemplos: n×2 (POS vs NEG) para PEVA."""
    peva = df[df["dataset"] == "PEVA"]
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    axes[0, 0].set_title("POSITIVE", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("NEGATIVE", fontsize=12, fontweight="bold")

    for col, label in enumerate(["POSITIVE", "NEGATIVE"]):
        sub = peva[peva["label_name"] == label]
        samples = sub.sample(n=n, random_state=rng)
        for row, (_, record) in enumerate(samples.iterrows()):
            img = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_ylabel(f"Pac. {record['patient_id']}", fontsize=8)

    plt.suptitle("Exemplos de imagens (PEVA)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_sample_images.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig_sample_images.png"))
    plt.close()
    print("  ✓ fig_sample_images")


def fig_intensity_histogram(df, figures_dir, max_images=500):
    """Histograma de intensidade POS vs NEG (PEVA)."""
    peva = df[df["dataset"] == "PEVA"]
    rng = np.random.RandomState(42)

    fig, ax = plt.subplots(figsize=(6, 4))

    for label, color in [("POSITIVE", "#2ecc71"), ("NEGATIVE", "#e74c3c")]:
        sub = peva[peva["label_name"] == label]
        sample = sub.sample(n=min(max_images, len(sub)), random_state=rng)
        all_intensities = []
        for _, record in tqdm(sample.iterrows(), total=len(sample),
                              desc=f"Intensidade {label}", leave=False):
            img = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_intensities.extend(img.flatten()[::10])  # subsample pixels

        ax.hist(all_intensities, bins=256, range=(0, 255), alpha=0.5,
                color=color, density=True, label=label)

    ax.set_xlabel("Intensidade de pixel")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição de intensidade (PEVA)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_intensity_histogram.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig_intensity_histogram.png"))
    plt.close()
    print("  ✓ fig_intensity_histogram")


def fig_mean_image(df, figures_dir, target_size=224, max_images=300):
    """Imagem média por classe (PEVA)."""
    peva = df[df["dataset"] == "PEVA"]
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    for ax, (label, title) in zip(axes, [("POSITIVE", "Média POSITIVA"),
                                          ("NEGATIVE", "Média NEGATIVA")]):
        sub = peva[peva["label_name"] == label]
        sample = sub.sample(n=min(max_images, len(sub)), random_state=rng)
        accumulator = np.zeros((target_size, target_size), dtype=np.float64)
        count = 0

        for _, record in tqdm(sample.iterrows(), total=len(sample),
                              desc=f"Média {label}", leave=False):
            img = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (target_size, target_size))
                accumulator += img_resized.astype(np.float64)
                count += 1

        if count > 0:
            mean_img = (accumulator / count).astype(np.uint8)
            ax.imshow(mean_img, cmap="gray")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Imagem média por classe (PEVA)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig_mean_image.pdf"))
    plt.savefig(os.path.join(figures_dir, "fig_mean_image.png"))
    plt.close()
    print("  ✓ fig_mean_image")


def generate_eda_summary(df, results_dir):
    """Gera resumo estatístico em texto para usar no artigo."""
    summary = []
    summary.append("=== RESUMO EDA ===\n")

    for ds in ["PEVA", "PESC"]:
        sub = df[df["dataset"] == ds]
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0]
        summary.append(f"\n--- {ds} ---")
        summary.append(f"Total imagens: {len(sub)}")
        summary.append(f"Positivas: {len(pos)} ({len(pos)/len(sub)*100:.1f}%)")
        summary.append(f"Negativas: {len(neg)} ({len(neg)/len(sub)*100:.1f}%)")
        summary.append(f"Razão NEG:POS = {len(neg)/max(len(pos),1):.1f}:1")
        summary.append(f"Pacientes únicos: {sub['patient_id'].nunique()}")
        summary.append(f"  POS: {pos['patient_id'].nunique()}")
        summary.append(f"  NEG: {neg['patient_id'].nunique()}")

        imgs_pp = sub.groupby("patient_id").size()
        summary.append(f"Imagens/paciente: média={imgs_pp.mean():.1f}, "
                       f"mediana={imgs_pp.median():.0f}, "
                       f"min={imgs_pp.min()}, max={imgs_pp.max()}")

    text = "\n".join(summary)
    out_path = os.path.join(results_dir, "eda_summary.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\n{text}")
    print(f"\nResumo salvo em: {out_path}")


def main():
    config = load_config()
    figures_dir = config["paths"]["figures_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    df = load_registry(config)
    print(f"Registry carregado: {len(df)} imagens\n")

    print("Gerando figuras EDA...")
    fig_class_distribution(df, figures_dir)
    fig_images_per_patient(df, figures_dir)
    fig_sample_images(df, figures_dir)
    fig_intensity_histogram(df, figures_dir)
    fig_mean_image(df, figures_dir)

    generate_eda_summary(df, results_dir)

    print(f"\nFiguras salvas em: {figures_dir}")


if __name__ == "__main__":
    main()
