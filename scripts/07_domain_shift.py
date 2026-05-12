#!/usr/bin/env python3
"""Script 07: Análise de domain shift — treinar em PEVA/testar em PESC e vice-versa."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.reproducibility import setup_experiment, set_seed
from src.data.torch_dataset import PlasmaDataset
from src.data.augmentation import get_train_transforms
from src.models.cnn_models import create_cnn_model
from src.models.train_cnn import train_model, evaluate
from src.evaluation.metrics import compute_metrics, bootstrap_ci, format_metric_with_ci
from src.features.handcrafted import extract_all_features
from src.data.preprocessing import preprocess_image
from src.models.baseline_classifiers import get_baseline_models
from tqdm import tqdm


def extract_features_from_df(df, config):
    """Extrai features handcrafted de um DataFrame."""
    target_size = config["preprocessing"]["target_size"]
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extraindo features"):
        img = preprocess_image(
            row["image_path"], target_size=target_size, apply_roi=False,
            min_mean=config["preprocessing"]["min_mean_intensity"],
            max_mean=config["preprocessing"]["max_mean_intensity"],
        )
        if img is None:
            continue
        X.append(extract_all_features(img))
        y.append(row["label"])
    return np.array(X), np.array(y)


def run_domain_shift_baseline(df_train, df_test, config, direction):
    """Avalia domain shift com modelos baseline."""
    print(f"\n--- Baseline: {direction} ---")

    X_train, y_train = extract_features_from_df(df_train, config)
    X_test, y_test = extract_features_from_df(df_test, config)

    print(f"  Train: {X_train.shape[0]} imgs (POS={sum(y_train==1)}, NEG={sum(y_train==0)})")
    print(f"  Test:  {X_test.shape[0]} imgs (POS={sum(y_test==1)}, NEG={sum(y_test==0)})")

    models = get_baseline_models(config)
    results = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        ci = bootstrap_ci(y_test, y_pred, y_prob)
        results[name] = {"metrics": metrics, "ci": ci}

        print(f"  {name}:")
        for key in ["accuracy", "sensitivity", "specificity", "f1", "auc"]:
            if key in metrics:
                print(f"    {format_metric_with_ci(key, metrics[key], ci)}")

    return results


def run_domain_shift_cnn(df_train, df_test, config, device, direction):
    """Avalia domain shift com MobileNetV2 (melhor modelo)."""
    print(f"\n--- CNN (MobileNetV2): {direction} ---")
    arch = "mobilenetv2"
    cnn_config = config["cnn"]

    train_transform = get_train_transforms(config)
    train_ds = PlasmaDataset(df_train, target_size=config["preprocessing"]["target_size"],
                              transform=train_transform, is_train=True)
    test_ds = PlasmaDataset(df_test, target_size=config["preprocessing"]["target_size"],
                             transform=None, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cnn_config["batch_size"],
                               shuffle=True, num_workers=cnn_config["num_workers"],
                               pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cnn_config["batch_size"],
                              shuffle=False, num_workers=cnn_config["num_workers"],
                              pin_memory=True)

    print(f"  Train: {len(train_ds)} imgs")
    print(f"  Test:  {len(test_ds)} imgs")

    set_seed(config["reproducibility"]["seed"])
    model = create_cnn_model(arch, num_classes=2, pretrained=True).to(device)

    start = time.time()
    train_model(model, train_loader, test_loader, config, device, arch_name=arch)
    elapsed = time.time() - start

    criterion = torch.nn.CrossEntropyLoss()
    _, _, test_auc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    test_preds = (test_probs >= 0.5).astype(int)
    metrics = compute_metrics(test_labels, test_preds, test_probs)
    ci = bootstrap_ci(test_labels, test_preds, test_probs)

    print(f"  Resultados ({direction}):")
    for key in ["accuracy", "sensitivity", "specificity", "f1", "auc"]:
        if key in metrics:
            print(f"    {format_metric_with_ci(key, metrics[key], ci)}")

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return {"metrics": metrics, "ci": ci, "time": round(elapsed, 2)}


def main():
    config, device = setup_experiment()
    results_dir = config["paths"]["results_dir"]
    logs_dir = config["paths"]["logs_dir"]

    registry_path = os.path.join(results_dir, "patient_registry_filtered.csv")
    df = pd.read_csv(registry_path)

    df_peva = df[df["dataset"] == "PEVA"].reset_index(drop=True)
    df_pesc = df[df["dataset"] == "PESC"].reset_index(drop=True)

    print(f"PEVA: {len(df_peva)} imgs ({sum(df_peva['label']==1)} POS, {sum(df_peva['label']==0)} NEG)")
    print(f"PESC: {len(df_pesc)} imgs ({sum(df_pesc['label']==1)} POS, {sum(df_pesc['label']==0)} NEG)")

    all_results = {}

    # === Direção 1: Treinar em PEVA → Testar em PESC ===
    print("\n" + "="*60)
    print("DIREÇÃO 1: Treinar em PEVA → Testar em PESC")
    print("="*60)

    baseline_peva_pesc = run_domain_shift_baseline(df_peva, df_pesc, config, "PEVA→PESC")
    cnn_peva_pesc = run_domain_shift_cnn(df_peva, df_pesc, config, device, "PEVA→PESC")

    all_results["PEVA_to_PESC"] = {
        "baseline": {k: v["metrics"] for k, v in baseline_peva_pesc.items()},
        "baseline_ci": {k: v["ci"] for k, v in baseline_peva_pesc.items()},
        "cnn": cnn_peva_pesc["metrics"],
        "cnn_ci": cnn_peva_pesc["ci"],
    }

    # === Direção 2: Treinar em PESC → Testar em PEVA ===
    print("\n" + "="*60)
    print("DIREÇÃO 2: Treinar em PESC → Testar em PEVA")
    print("="*60)

    baseline_pesc_peva = run_domain_shift_baseline(df_pesc, df_peva, config, "PESC→PEVA")
    cnn_pesc_peva = run_domain_shift_cnn(df_pesc, df_peva, config, device, "PESC→PEVA")

    all_results["PESC_to_PEVA"] = {
        "baseline": {k: v["metrics"] for k, v in baseline_pesc_peva.items()},
        "baseline_ci": {k: v["ci"] for k, v in baseline_pesc_peva.items()},
        "cnn": cnn_pesc_peva["metrics"],
        "cnn_ci": cnn_pesc_peva["ci"],
    }

    # === Referência: desempenho intra-dataset (PEVA hold-out) ===
    # Já disponível em baseline_results.json e cnn_results.json

    # Salvar resultados
    log_path = os.path.join(logs_dir, "domain_shift_results.json")
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados salvos em: {log_path}")

    # Resumo comparativo
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO — Domain Shift")
    print("="*60)
    print(f"\n{'Direção':<25} {'Modelo':<15} {'AUC':<10} {'Sens':<10} {'Spec':<10}")
    print("-" * 70)

    for direction, label in [("PEVA_to_PESC", "PEVA→PESC"), ("PESC_to_PEVA", "PESC→PEVA")]:
        res = all_results[direction]
        # Melhor baseline
        best_bl = max(res["baseline"].items(), key=lambda x: x[1].get("auc", 0))
        print(f"{label:<25} {best_bl[0]:<15} {best_bl[1].get('auc', 0):<10.4f} "
              f"{best_bl[1]['sensitivity']:<10.4f} {best_bl[1]['specificity']:<10.4f}")
        # CNN
        print(f"{'':25} {'MobileNetV2':<15} {res['cnn'].get('auc', 0):<10.4f} "
              f"{res['cnn']['sensitivity']:<10.4f} {res['cnn']['specificity']:<10.4f}")


if __name__ == "__main__":
    main()
