#!/usr/bin/env python3
"""Script 04: Treina e avalia baseline clássico (RF, SVM) com 5-fold por paciente."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.reproducibility import load_config, set_seed
from src.data.patient_splitter import patient_split
from src.data.preprocessing import preprocess_image
from src.features.handcrafted import extract_all_features
from src.models.baseline_classifiers import get_baseline_models
from src.evaluation.metrics import compute_metrics, bootstrap_ci, format_metric_with_ci


def extract_features_from_registry(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extrai features handcrafted de todas as imagens do registry."""
    target_size = config["preprocessing"]["target_size"]
    X, y = [], []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extraindo features"):
        img = preprocess_image(
            row["image_path"], target_size=target_size, apply_roi=False,
            min_mean=config["preprocessing"]["min_mean_intensity"],
            max_mean=config["preprocessing"]["max_mean_intensity"],
        )
        if img is None:
            failed += 1
            continue
        features = extract_all_features(img)
        X.append(features)
        y.append(row["label"])

    if failed > 0:
        print(f"  {failed} imagens falharam no pré-processamento")

    return np.array(X), np.array(y)


def run_baseline(config: dict):
    """Executa pipeline baseline completo."""
    set_seed(config["reproducibility"]["seed"])
    results_dir = config["paths"]["results_dir"]
    logs_dir = config["paths"]["logs_dir"]

    # Carregar registry filtrado
    registry_path = os.path.join(results_dir, "patient_registry_filtered.csv")
    df = pd.read_csv(registry_path)
    print(f"Registry filtrado: {len(df)} imagens")

    # Split por paciente (apenas PEVA por enquanto)
    print("\n=== Divisão por paciente (PEVA) ===")
    df_trainval, df_test, folds = patient_split(
        df, n_folds=config["splitting"]["n_folds"],
        test_size=config["splitting"]["test_size"],
        seed=config["splitting"]["random_seed"],
        dataset="PEVA",
    )

    # Extrair features
    print("\n=== Extração de features (train+val) ===")
    X_trainval, y_trainval = extract_features_from_registry(df_trainval, config)
    print(f"Features shape: {X_trainval.shape}")

    print("\n=== Extração de features (test) ===")
    X_test, y_test = extract_features_from_registry(df_test, config)
    print(f"Test features shape: {X_test.shape}")

    # Modelos
    models = get_baseline_models(config)
    all_results = {}

    for model_name, pipeline in models.items():
        print(f"\n{'='*60}")
        print(f"Modelo: {model_name}")
        print(f"{'='*60}")

        fold_metrics = []
        start_time = time.time()

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_train = X_trainval[train_idx]
            y_train = y_trainval[train_idx]
            X_val = X_trainval[val_idx]
            y_val = y_trainval[val_idx]

            # Treinar
            pipeline.fit(X_train, y_train)

            # Predizer
            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            metrics = compute_metrics(y_val, y_pred, y_prob)
            fold_metrics.append(metrics)

            print(f"  Fold {fold_i}: AUC={metrics.get('auc', 'N/A'):.4f}, "
                  f"Sens={metrics['sensitivity']:.4f}, "
                  f"Spec={metrics['specificity']:.4f}, "
                  f"F1={metrics['f1']:.4f}")

        elapsed = time.time() - start_time

        # Métricas médias dos folds
        avg_metrics = {}
        for key in fold_metrics[0]:
            if key in ("tp", "tn", "fp", "fn"):
                continue
            values = [m[key] for m in fold_metrics if key in m]
            avg_metrics[key] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
            }

        print(f"\n  Média 5-fold:")
        for key, val in avg_metrics.items():
            print(f"    {key}: {val['mean']:.4f} ± {val['std']:.4f}")

        # Avaliar no test set (treinar em todo trainval)
        print(f"\n  Avaliação no hold-out test:")
        pipeline.fit(X_trainval, y_trainval)
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_prob)
        test_ci = bootstrap_ci(y_test, y_test_pred, y_test_prob)

        for key in ["accuracy", "sensitivity", "specificity", "f1", "auc"]:
            if key in test_metrics:
                print(f"    {format_metric_with_ci(key, test_metrics[key], test_ci)}")

        all_results[model_name] = {
            "fold_metrics": fold_metrics,
            "avg_metrics": avg_metrics,
            "test_metrics": test_metrics,
            "test_ci": test_ci,
            "training_time_seconds": round(elapsed, 2),
        }

    # Salvar resultados
    log_path = os.path.join(logs_dir, "baseline_results.json")
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados salvos em: {log_path}")

    return all_results


def main():
    config = load_config()
    run_baseline(config)


if __name__ == "__main__":
    main()
