#!/usr/bin/env python3
"""Script 05: Treina CNNs com transfer learning — 3 arquiteturas × 5 folds."""

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
from src.data.patient_splitter import patient_split
from src.data.torch_dataset import PlasmaDataset
from src.data.augmentation import get_train_transforms
from src.models.cnn_models import create_cnn_model
from src.models.train_cnn import train_model, evaluate
from src.evaluation.metrics import compute_metrics, bootstrap_ci, format_metric_with_ci


def run_cnn_experiments(config, device):
    """Executa todos os experimentos CNN."""
    results_dir = config["paths"]["results_dir"]
    logs_dir = config["paths"]["logs_dir"]
    cnn_config = config["cnn"]

    # Carregar registry filtrado
    registry_path = os.path.join(results_dir, "patient_registry_filtered.csv")
    df = pd.read_csv(registry_path)
    print(f"Registry: {len(df)} imagens")

    # Split por paciente
    print("\n=== Divisão por paciente (PEVA) ===")
    df_trainval, df_test, folds = patient_split(
        df, n_folds=config["splitting"]["n_folds"],
        test_size=config["splitting"]["test_size"],
        seed=config["splitting"]["random_seed"],
        dataset="PEVA",
    )

    train_transform = get_train_transforms(config)
    all_results = {}

    for arch_name in cnn_config["architectures"]:
        print(f"\n{'='*60}")
        print(f"Arquitetura: {arch_name}")
        print(f"{'='*60}")

        fold_metrics = []
        fold_histories = []
        start_time = time.time()

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_i} ---")
            set_seed(config["reproducibility"]["seed"] + fold_i)

            df_train = df_trainval.iloc[train_idx]
            df_val = df_trainval.iloc[val_idx]

            train_ds = PlasmaDataset(df_train, target_size=config["preprocessing"]["target_size"],
                                     transform=train_transform, is_train=True)
            val_ds = PlasmaDataset(df_val, target_size=config["preprocessing"]["target_size"],
                                   transform=None, is_train=False)

            train_loader = DataLoader(train_ds, batch_size=cnn_config["batch_size"],
                                      shuffle=True, num_workers=cnn_config["num_workers"],
                                      pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=cnn_config["batch_size"],
                                    shuffle=False, num_workers=cnn_config["num_workers"],
                                    pin_memory=True)

            # Criar modelo
            model = create_cnn_model(arch_name, num_classes=2, pretrained=True)
            model = model.to(device)

            # Treinar
            result = train_model(model, train_loader, val_loader, config, device,
                                fold_idx=fold_i, arch_name=arch_name)
            fold_histories.append(result["history"])

            # Avaliar no val set
            criterion = torch.nn.CrossEntropyLoss()
            _, _, val_auc, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
            val_preds = (val_probs >= 0.5).astype(int)
            metrics = compute_metrics(val_labels, val_preds, val_probs)
            fold_metrics.append(metrics)

            print(f"  Fold {fold_i} final: AUC={metrics.get('auc', 0):.4f}, "
                  f"Sens={metrics['sensitivity']:.4f}, "
                  f"Spec={metrics['specificity']:.4f}, "
                  f"F1={metrics['f1']:.4f}")

            # Salvar checkpoint do melhor fold
            ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"{arch_name}_fold{fold_i}.pth"))

            # Limpar memória GPU
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

        elapsed = time.time() - start_time

        # Métricas médias
        avg_metrics = {}
        for key in fold_metrics[0]:
            if key in ("tp", "tn", "fp", "fn"):
                continue
            values = [m[key] for m in fold_metrics if key in m]
            avg_metrics[key] = {"mean": round(np.mean(values), 4),
                                "std": round(np.std(values), 4)}

        print(f"\n  Média 5-fold ({arch_name}):")
        for key, val in avg_metrics.items():
            print(f"    {key}: {val['mean']:.4f} ± {val['std']:.4f}")

        # Avaliar no test set (treinar no trainval completo)
        print(f"\n  Avaliação hold-out test...")
        set_seed(config["reproducibility"]["seed"])
        model = create_cnn_model(arch_name, num_classes=2, pretrained=True).to(device)

        full_train_ds = PlasmaDataset(df_trainval,
                                       target_size=config["preprocessing"]["target_size"],
                                       transform=train_transform, is_train=True)
        test_ds = PlasmaDataset(df_test, target_size=config["preprocessing"]["target_size"],
                                transform=None, is_train=False)
        full_train_loader = DataLoader(full_train_ds, batch_size=cnn_config["batch_size"],
                                        shuffle=True, num_workers=cnn_config["num_workers"],
                                        pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=cnn_config["batch_size"],
                                  shuffle=False, num_workers=cnn_config["num_workers"],
                                  pin_memory=True)

        train_model(model, full_train_loader, test_loader, config, device,
                    arch_name=arch_name)

        criterion = torch.nn.CrossEntropyLoss()
        _, _, test_auc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
        test_preds = (test_probs >= 0.5).astype(int)
        test_metrics = compute_metrics(test_labels, test_preds, test_probs)
        test_ci = bootstrap_ci(test_labels, test_preds, test_probs)

        for key in ["accuracy", "sensitivity", "specificity", "f1", "auc"]:
            if key in test_metrics:
                print(f"    {format_metric_with_ci(key, test_metrics[key], test_ci)}")

        # Salvar melhor modelo
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{arch_name}_final.pth"))
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        all_results[arch_name] = {
            "fold_metrics": fold_metrics,
            "avg_metrics": avg_metrics,
            "test_metrics": test_metrics,
            "test_ci": test_ci,
            "training_time_seconds": round(elapsed, 2),
            "fold_histories": fold_histories,
        }

    # Salvar resultados
    # Converter histories para serializável
    serializable = {}
    for arch, res in all_results.items():
        s = {k: v for k, v in res.items() if k != "fold_histories"}
        s["fold_histories"] = [
            {k: ([float(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], (int, float, np.floating))
                 else v)
             for k, v in h.items()}
            for h in res["fold_histories"]
        ]
        serializable[arch] = s

    log_path = os.path.join(logs_dir, "cnn_results.json")
    with open(log_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResultados salvos em: {log_path}")

    return all_results


def main():
    config, device = setup_experiment()
    run_cnn_experiments(config, device)


if __name__ == "__main__":
    main()
