#!/usr/bin/env python3
"""Script 03: Pré-processa imagens e gera relatório de qualidade."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.reproducibility import load_config
from src.data.preprocessing import preprocess_image


def main():
    config = load_config()
    results_dir = config["paths"]["results_dir"]
    registry_path = os.path.join(results_dir, "patient_registry.csv")

    df = pd.read_csv(registry_path)
    print(f"Registry carregado: {len(df)} imagens")

    prep = config["preprocessing"]
    target_size = prep["target_size"]
    min_mean = prep["min_mean_intensity"]
    max_mean = prep["max_mean_intensity"]

    # Processar e verificar qualidade de todas as imagens
    quality_report = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verificando qualidade"):
        img_path = row["image_path"]
        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_raw is None:
            quality_report.append({
                **row.to_dict(),
                "status": "ERRO_LEITURA",
                "mean_intensity": None,
                "shape": None,
            })
            continue

        mean_intensity = float(np.mean(img_raw))
        passed = min_mean <= mean_intensity <= max_mean

        quality_report.append({
            **row.to_dict(),
            "status": "OK" if passed else "REJEITADA",
            "mean_intensity": round(mean_intensity, 2),
            "shape": f"{img_raw.shape[0]}x{img_raw.shape[1]}",
        })

    report_df = pd.DataFrame(quality_report)

    # Estatísticas
    ok = report_df[report_df["status"] == "OK"]
    rejected = report_df[report_df["status"] == "REJEITADA"]
    errors = report_df[report_df["status"] == "ERRO_LEITURA"]

    print(f"\n=== Relatório de Qualidade ===")
    print(f"Total: {len(report_df)}")
    print(f"OK: {len(ok)} ({len(ok)/len(report_df)*100:.1f}%)")
    print(f"Rejeitadas: {len(rejected)} ({len(rejected)/len(report_df)*100:.1f}%)")
    print(f"Erros de leitura: {len(errors)}")

    if len(rejected) > 0:
        print(f"\nImagens rejeitadas por dataset:")
        for ds in rejected["dataset"].unique():
            sub = rejected[rejected["dataset"] == ds]
            print(f"  {ds}: {len(sub)} ({len(sub[sub['label']==1])} POS, {len(sub[sub['label']==0])} NEG)")
        print(f"\nIntensidade média das rejeitadas:")
        print(f"  min={rejected['mean_intensity'].min():.1f}, "
              f"max={rejected['mean_intensity'].max():.1f}, "
              f"média={rejected['mean_intensity'].mean():.1f}")

    # Salvar relatório
    report_path = os.path.join(results_dir, "quality_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nRelatório salvo em: {report_path}")

    # Salvar registry filtrado (apenas imagens OK)
    ok_paths = set(ok["image_path"])
    df_filtered = df[df["image_path"].isin(ok_paths)].copy()
    filtered_path = os.path.join(results_dir, "patient_registry_filtered.csv")
    df_filtered.to_csv(filtered_path, index=False)
    print(f"Registry filtrado salvo em: {filtered_path} ({len(df_filtered)} imagens)")

    # Verificar uma amostra de pré-processamento
    print("\n=== Amostra de pré-processamento ===")
    sample = df_filtered.sample(n=5, random_state=42)
    for _, row in sample.iterrows():
        processed = preprocess_image(row["image_path"], target_size, apply_roi=True,
                                     min_mean=min_mean, max_mean=max_mean)
        if processed is not None:
            print(f"  {os.path.basename(row['image_path'])}: "
                  f"original → {processed.shape[0]}×{processed.shape[1]}")


if __name__ == "__main__":
    main()
