#!/usr/bin/env python3
"""Script 09: Experimento de blob cropping — MobileNetV2 com e sem crop.

Etapas:
  1. Detecta blob de emissão de plasma via threshold + contours
  2. Salva imagens cropadas em diretório separado (sem alterar originais)
  3. Treina MobileNetV2 com imagens cropadas (5-fold CV + hold-out)
  4. Gera tabela comparativa com baseline (sem crop)
  5. Gera Grad-CAM comparativo (com vs sem crop)
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.reproducibility import setup_experiment, set_seed
from src.data.patient_splitter import patient_split
from src.data.torch_dataset import PlasmaDataset
from src.data.augmentation import get_train_transforms
from src.models.cnn_models import create_cnn_model
from src.models.train_cnn import train_model, evaluate
from src.evaluation.metrics import compute_metrics, bootstrap_ci
from src.evaluation.gradcam import GradCAM, overlay_heatmap


# ---------------------------------------------------------------------------
# 1. Blob detection e cropping
# ---------------------------------------------------------------------------

def detect_blob(img_gray: np.ndarray, threshold: int = 30,
                min_area: int = 500, padding_ratio: float = 0.25):
    """
    Detecta a região de emissão de plasma via threshold + maior contorno.

    Returns:
        (x1, y1, x2, y2) do crop quadrado centrado no blob, ou None se falhar.
    """
    _, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2

    # Crop quadrado com padding
    side = int(max(w, h) * (1 + padding_ratio))
    half = side // 2

    # Clamp ao tamanho da imagem
    img_h, img_w = img_gray.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img_w, cx + half)
    y2 = min(img_h, cy + half)

    # Garantir que o crop seja quadrado (ajustar pelo lado menor)
    crop_w = x2 - x1
    crop_h = y2 - y1
    actual_side = min(crop_w, crop_h)
    # Re-centralizar
    x1 = max(0, cx - actual_side // 2)
    y1 = max(0, cy - actual_side // 2)
    x2 = x1 + actual_side
    y2 = y1 + actual_side

    # Clamp final
    if x2 > img_w:
        x2 = img_w
        x1 = max(0, x2 - actual_side)
    if y2 > img_h:
        y2 = img_h
        y1 = max(0, y2 - actual_side)

    return (x1, y1, x2, y2)


def crop_all_images(df: pd.DataFrame, output_dir: str, threshold: int = 30):
    """
    Gera versões cropadas de todas as imagens.

    Returns:
        DataFrame com coluna 'cropped_path' adicionada.
        Imagens sem blob detectado usam crop central como fallback.
    """
    os.makedirs(output_dir, exist_ok=True)
    cropped_paths = []
    stats = {"blob_found": 0, "fallback_center": 0, "failed": 0}

    for i, row in df.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            cropped_paths.append(None)
            stats["failed"] += 1
            continue

        bbox = detect_blob(img, threshold=threshold)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped = img[y1:y2, x1:x2]
            stats["blob_found"] += 1
        else:
            # Fallback: crop central quadrado (menor dimensão)
            h, w = img.shape[:2]
            side = min(h, w)
            cx, cy = w // 2, h // 2
            x1 = cx - side // 2
            y1 = cy - side // 2
            cropped = img[y1:y1+side, x1:x1+side]
            stats["fallback_center"] += 1

        # Manter estrutura de pastas relativa
        rel_path = os.path.basename(row["image_path"])
        folder = row.get("folder_name", "unknown")
        save_dir = os.path.join(output_dir, row["dataset"], row["label_name"], folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, rel_path.replace(".bmp", "_crop.png"))

        cv2.imwrite(save_path, cropped)
        cropped_paths.append(save_path)

    df_out = df.copy()
    df_out["cropped_path"] = cropped_paths

    print(f"\nCropping stats:")
    print(f"  Blob detected: {stats['blob_found']}")
    print(f"  Fallback center crop: {stats['fallback_center']}")
    print(f"  Failed (no image): {stats['failed']}")

    return df_out


def analyze_blob_statistics(df: pd.DataFrame, threshold: int = 30,
                            output_dir: str = None):
    """
    Analisa estatísticas do blob detection para documentação no artigo.
    Salva relatório e gráficos de distribuição.
    """
    records = []
    for _, row in df.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, thresh_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            records.append({"detected": False, "label": row["label"],
                            "patient_id": row["patient_id"]})
            continue

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area = cv2.contourArea(largest)
        cx, cy = x + w // 2, y + h // 2

        records.append({
            "detected": True, "label": row["label"],
            "patient_id": row["patient_id"],
            "cx": cx, "cy": cy, "w": w, "h": h,
            "area": area, "img_area": img.shape[0] * img.shape[1],
            "area_ratio": area / (img.shape[0] * img.shape[1]),
            "mean_intensity": float(img.mean()),
        })

    stats_df = pd.DataFrame(records)
    detected = stats_df[stats_df["detected"]]

    report_lines = [
        "=" * 60,
        "RELATÓRIO DE BLOB DETECTION — Documentação para artigo",
        "=" * 60,
        f"Total de imagens analisadas: {len(stats_df)}",
        f"Blobs detectados: {len(detected)} ({100*len(detected)/len(stats_df):.1f}%)",
        f"Sem detecção (fallback center crop): {len(stats_df) - len(detected)}",
        "",
        "--- Estatísticas do centro do blob (pixels, imagem 640x480) ---",
        f"Centro X: média={detected['cx'].mean():.1f}, std={detected['cx'].std():.1f}, "
        f"range=[{detected['cx'].min()}, {detected['cx'].max()}]",
        f"Centro Y: média={detected['cy'].mean():.1f}, std={detected['cy'].std():.1f}, "
        f"range=[{detected['cy'].min()}, {detected['cy'].max()}]",
        "",
        "--- Dimensões do bounding box ---",
        f"Largura: média={detected['w'].mean():.1f}, std={detected['w'].std():.1f}, "
        f"range=[{detected['w'].min()}, {detected['w'].max()}]",
        f"Altura:  média={detected['h'].mean():.1f}, std={detected['h'].std():.1f}, "
        f"range=[{detected['h'].min()}, {detected['h'].max()}]",
        f"Área:    média={detected['area'].mean():.0f}, std={detected['area'].std():.0f}",
        f"Razão área blob/imagem: média={detected['area_ratio'].mean():.4f} "
        f"({detected['area_ratio'].mean()*100:.2f}%)",
        "",
        "--- Por classe ---",
    ]

    for label_val, label_name in [(1, "Positivo (TB+)"), (0, "Negativo (TB-)")]:
        sub = detected[detected["label"] == label_val]
        if len(sub) == 0:
            continue
        report_lines.extend([
            f"  {label_name} (n={len(sub)}):",
            f"    Centro X: {sub['cx'].mean():.1f} ± {sub['cx'].std():.1f}",
            f"    Centro Y: {sub['cy'].mean():.1f} ± {sub['cy'].std():.1f}",
            f"    Largura:  {sub['w'].mean():.1f} ± {sub['w'].std():.1f}",
            f"    Altura:   {sub['h'].mean():.1f} ± {sub['h'].std():.1f}",
            f"    Área:     {sub['area'].mean():.0f} ± {sub['area'].std():.0f}",
            f"    Intensidade média: {sub['mean_intensity'].mean():.2f} ± {sub['mean_intensity'].std():.2f}",
        ])

    report_lines.extend([
        "",
        "--- Parâmetros do blob cropping ---",
        f"Threshold de binarização: {threshold}",
        f"Área mínima para detecção: 500 pixels",
        f"Padding ratio: 25% (crop quadrado = max(w,h) * 1.25)",
        f"Fallback: crop central quadrado (min(H,W) pixels)",
        "Método: threshold fixo + maior contorno + bounding box quadrado com padding",
    ])

    report = "\n".join(report_lines)
    print(report)

    if output_dir:
        report_path = os.path.join(output_dir, "blob_detection_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nRelatório salvo em: {report_path}")

        stats_csv = os.path.join(output_dir, "blob_detection_stats.csv")
        detected.to_csv(stats_csv, index=False)
        print(f"Estatísticas CSV salvas em: {stats_csv}")

    return stats_df


# ---------------------------------------------------------------------------
# 2. Dataset que lê do path cropado
# ---------------------------------------------------------------------------

class CroppedPlasmaDataset(PlasmaDataset):
    """PlasmaDataset que lê de cropped_path ao invés de image_path."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row.get("cropped_path", row["image_path"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        img = cv2.resize(img, (self.target_size, self.target_size),
                         interpolation=cv2.INTER_AREA)

        if self.transform is not None and self.is_train:
            result = self.transform(image=img)
            img = result["image"]

        from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD
        img_3ch = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        img_3ch = (img_3ch - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1)).float()
        label = torch.tensor(row["label"], dtype=torch.long)

        return img_tensor, label


# ---------------------------------------------------------------------------
# 3. Treino MobileNetV2 com imagens cropadas
# ---------------------------------------------------------------------------

def run_cropped_experiment(config, device, df_trainval, df_test, folds):
    """Treina MobileNetV2 nas imagens cropadas com o mesmo pipeline."""
    cnn_config = config["cnn"]
    results_dir = config["paths"]["results_dir"]
    arch_name = "mobilenetv2"

    train_transform = get_train_transforms(config)
    fold_metrics = []
    start_time = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_i} ---")
        set_seed(config["reproducibility"]["seed"] + fold_i)

        df_train = df_trainval.iloc[train_idx]
        df_val = df_trainval.iloc[val_idx]

        train_ds = CroppedPlasmaDataset(
            df_train, target_size=config["preprocessing"]["target_size"],
            transform=train_transform, is_train=True)
        val_ds = CroppedPlasmaDataset(
            df_val, target_size=config["preprocessing"]["target_size"],
            transform=None, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=cnn_config["batch_size"],
                                  shuffle=True, num_workers=cnn_config["num_workers"],
                                  pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cnn_config["batch_size"],
                                shuffle=False, num_workers=cnn_config["num_workers"],
                                pin_memory=True)

        model = create_cnn_model(arch_name, num_classes=2, pretrained=True).to(device)
        result = train_model(model, train_loader, val_loader, config, device,
                             fold_idx=fold_i, arch_name=f"{arch_name}_crop")

        criterion = torch.nn.CrossEntropyLoss()
        _, _, val_auc, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        val_preds = (val_probs >= 0.5).astype(int)
        metrics = compute_metrics(val_labels, val_preds, val_probs)
        fold_metrics.append(metrics)

        print(f"  Fold {fold_i}: AUC={metrics.get('auc', 0):.4f}, "
              f"Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")

        # Salvar checkpoint
        ckpt_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"{arch_name}_crop_fold{fold_i}.pth"))
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

    print(f"\nMédia 5-fold (cropped):")
    for key, val in avg_metrics.items():
        print(f"  {key}: {val['mean']:.4f} ± {val['std']:.4f}")

    # Hold-out test (treinar no trainval completo)
    print(f"\nAvaliação hold-out test (cropped)...")
    set_seed(config["reproducibility"]["seed"])
    model = create_cnn_model(arch_name, num_classes=2, pretrained=True).to(device)

    full_train_ds = CroppedPlasmaDataset(
        df_trainval, target_size=config["preprocessing"]["target_size"],
        transform=train_transform, is_train=True)
    test_ds = CroppedPlasmaDataset(
        df_test, target_size=config["preprocessing"]["target_size"],
        transform=None, is_train=False)

    full_train_loader = DataLoader(full_train_ds, batch_size=cnn_config["batch_size"],
                                   shuffle=True, num_workers=cnn_config["num_workers"],
                                   pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cnn_config["batch_size"],
                             shuffle=False, num_workers=cnn_config["num_workers"],
                             pin_memory=True)

    train_model(model, full_train_loader, test_loader, config, device,
                arch_name=f"{arch_name}_crop")

    criterion = torch.nn.CrossEntropyLoss()
    _, _, test_auc, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    test_preds = (test_probs >= 0.5).astype(int)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    test_ci = bootstrap_ci(test_labels, test_preds, test_probs)

    print(f"\nHold-out test (cropped):")
    for k, v in test_metrics.items():
        if k not in ("tp", "tn", "fp", "fn"):
            ci_str = ""
            if k in test_ci:
                ci_str = f" [{test_ci[k]['lower']:.4f}, {test_ci[k]['upper']:.4f}]"
            print(f"  {k}: {v:.4f}{ci_str}")

    # Salvar modelo final
    torch.save(model.state_dict(),
               os.path.join(ckpt_dir, f"{arch_name}_crop_final.pth"))

    return {
        "fold_metrics": fold_metrics,
        "avg_metrics": avg_metrics,
        "test_metrics": test_metrics,
        "test_ci": test_ci,
        "training_time_seconds": round(elapsed, 2),
        "model": model,
        "test_ds": test_ds,
        "df_test": df_test,
    }


# ---------------------------------------------------------------------------
# 4. Tabela comparativa
# ---------------------------------------------------------------------------

def generate_comparison_table(baseline_results: dict, crop_results: dict,
                              output_dir: str):
    """Gera tabela comparativa LaTeX e CSV."""

    metrics_keys = ["auc", "sensitivity", "specificity", "accuracy", "f1"]

    rows = []
    for key in metrics_keys:
        # Baseline
        b_cv = baseline_results["avg_metrics"].get(key, {})
        b_test = baseline_results["test_metrics"].get(key, 0)
        b_ci = baseline_results.get("test_ci", {}).get(key, {})

        # Cropped
        c_cv = crop_results["avg_metrics"].get(key, {})
        c_test = crop_results["test_metrics"].get(key, 0)
        c_ci = crop_results.get("test_ci", {}).get(key, {})

        rows.append({
            "Metric": key.upper() if key == "auc" else key.capitalize(),
            "Baseline_CV": f"{b_cv.get('mean', 0):.3f} ± {b_cv.get('std', 0):.3f}",
            "Baseline_Test": f"{b_test:.3f}",
            "Baseline_CI": f"[{b_ci.get('lower', 0):.3f}, {b_ci.get('upper', 0):.3f}]" if b_ci else "-",
            "Crop_CV": f"{c_cv.get('mean', 0):.3f} ± {c_cv.get('std', 0):.3f}",
            "Crop_Test": f"{c_test:.3f}",
            "Crop_CI": f"[{c_ci.get('lower', 0):.3f}, {c_ci.get('upper', 0):.3f}]" if c_ci else "-",
            "Delta_Test": f"{c_test - b_test:+.3f}",
        })

    df_table = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "blob_crop_comparison.csv")
    df_table.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")

    # LaTeX
    latex_path = os.path.join(output_dir, "blob_crop_comparison.tex")
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparação de desempenho do MobileNetV2 com e sem \\textit{blob cropping} no conjunto de teste \\textit{hold-out}.}\n")
        f.write("\\label{tab:blob-crop-comparison}\n")
        f.write("\\begin{tabular}{l c c c c c}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Métrica} & \\textbf{Sem crop} & \\textbf{IC 95\\%} & \\textbf{Com crop} & \\textbf{IC 95\\%} & \\textbf{$\\Delta$} \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(f"{row['Metric']} & {row['Baseline_Test']} & {row['Baseline_CI']} & "
                    f"{row['Crop_Test']} & {row['Crop_CI']} & {row['Delta_Test']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX salva em: {latex_path}")

    # Print table
    print(f"\n{'Metric':<14} {'Baseline':<10} {'Cropped':<10} {'Delta':<8}")
    print("-" * 42)
    for row in rows:
        print(f"{row['Metric']:<14} {row['Baseline_Test']:<10} {row['Crop_Test']:<10} {row['Delta_Test']:<8}")


# ---------------------------------------------------------------------------
# 5. Grad-CAM comparativo
# ---------------------------------------------------------------------------

def generate_gradcam_comparison(config, device, baseline_model_path: str,
                                crop_model, df_test_orig, df_test_crop,
                                output_dir: str):
    """Gera galeria Grad-CAM lado a lado: sem crop vs com crop."""

    arch_name = "mobilenetv2"
    target_size = config["preprocessing"]["target_size"]

    # Carregar modelo baseline
    baseline_model = create_cnn_model(arch_name, num_classes=2, pretrained=False)
    baseline_model.load_state_dict(
        torch.load(baseline_model_path, map_location=device, weights_only=True))
    baseline_model = baseline_model.to(device)
    baseline_model.eval()

    crop_model.eval()

    gradcam_baseline = GradCAM(baseline_model)
    gradcam_crop = GradCAM(crop_model)

    # Criar datasets
    test_ds_orig = PlasmaDataset(df_test_orig, target_size=target_size)
    test_ds_crop = CroppedPlasmaDataset(df_test_crop, target_size=target_size)

    # Selecionar exemplos: 3 positivos e 3 negativos do test set
    pos_indices = df_test_orig[df_test_orig["label"] == 1].index.tolist()[:3]
    neg_indices = df_test_orig[df_test_orig["label"] == 0].index.tolist()[:3]
    sample_indices = pos_indices + neg_indices

    if not sample_indices:
        print("Sem amostras suficientes para Grad-CAM.")
        return

    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "Grad-CAM (sem crop)", "Cropada", "Grad-CAM (com crop)"]

    for row_i, idx in enumerate(sample_indices):
        label = df_test_orig.iloc[idx]["label"]
        label_str = "TB+" if label == 1 else "TB-"

        # --- Sem crop ---
        img_tensor_orig, _ = test_ds_orig[idx]
        img_tensor_orig = img_tensor_orig.unsqueeze(0).to(device)
        img_raw_orig = cv2.imread(df_test_orig.iloc[idx]["image_path"], cv2.IMREAD_GRAYSCALE)
        img_raw_orig = cv2.resize(img_raw_orig, (target_size, target_size))

        heatmap_orig, pred_orig, conf_orig = gradcam_baseline.generate(img_tensor_orig)
        overlay_orig = overlay_heatmap(img_raw_orig, heatmap_orig)

        # --- Com crop ---
        img_tensor_crop, _ = test_ds_crop[idx]
        img_tensor_crop = img_tensor_crop.unsqueeze(0).to(device)
        crop_path = df_test_crop.iloc[idx].get("cropped_path", df_test_crop.iloc[idx]["image_path"])
        img_raw_crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img_raw_crop is not None:
            img_raw_crop = cv2.resize(img_raw_crop, (target_size, target_size))
        else:
            img_raw_crop = np.zeros((target_size, target_size), dtype=np.uint8)

        heatmap_crop, pred_crop, conf_crop = gradcam_crop.generate(img_tensor_crop)
        overlay_crop = overlay_heatmap(img_raw_crop, heatmap_crop)

        # Plot
        axes[row_i, 0].imshow(img_raw_orig, cmap="gray")
        axes[row_i, 0].set_ylabel(f"{label_str}", fontsize=12, fontweight="bold",
                                   rotation=0, labelpad=30, va="center")

        axes[row_i, 1].imshow(cv2.cvtColor(overlay_orig, cv2.COLOR_BGR2RGB))
        pred_str = "TB+" if pred_orig == 1 else "TB-"
        axes[row_i, 1].set_title(f"Pred: {pred_str} ({conf_orig:.2f})", fontsize=9)

        axes[row_i, 2].imshow(img_raw_crop, cmap="gray")

        axes[row_i, 3].imshow(cv2.cvtColor(overlay_crop, cv2.COLOR_BGR2RGB))
        pred_str = "TB+" if pred_crop == 1 else "TB-"
        axes[row_i, 3].set_title(f"Pred: {pred_str} ({conf_crop:.2f})", fontsize=9)

        for col in range(4):
            axes[row_i, col].set_xticks([])
            axes[row_i, col].set_yticks([])

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=10)

    plt.suptitle("Grad-CAM: MobileNetV2 sem crop vs com blob cropping", fontsize=14, y=1.01)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"fig_gradcam_blob_crop.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGrad-CAM comparativo salvo em: {output_dir}")

    del baseline_model
    torch.cuda.empty_cache() if device.type == "cuda" else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config, device = setup_experiment()
    results_dir = config["paths"]["results_dir"]
    figures_dir = config["paths"]["figures_dir"]
    logs_dir = config["paths"]["logs_dir"]

    # Carregar registry
    registry_path = os.path.join(results_dir, "patient_registry_filtered.csv")
    df = pd.read_csv(registry_path)
    peva = df[df["dataset"] == "PEVA"].copy()
    print(f"PEVA: {len(peva)} imagens")

    # ===================================================================
    # Etapa 1: Blob cropping
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 1: Blob cropping")
    print("=" * 60)

    # Análise de estatísticas do blob detection (para documentação)
    print("\nAnalisando estatísticas de blob detection...")
    analyze_blob_statistics(peva, threshold=30, output_dir=logs_dir)

    # Crop das imagens
    crop_dir = os.path.join(config["paths"]["project_root"], "PEVA_cropped")
    df_cropped = crop_all_images(peva, crop_dir, threshold=30)

    # Salvar registry das imagens cropadas
    crop_registry_path = os.path.join(results_dir, "patient_registry_cropped.csv")
    df_cropped.to_csv(crop_registry_path, index=False)
    print(f"Registry cropado salvo em: {crop_registry_path}")

    # ===================================================================
    # Etapa 2: Split (mesmo que baseline — mesma seed, mesmo split)
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 2: Split por paciente")
    print("=" * 60)

    df_trainval, df_test, folds = patient_split(
        df_cropped, n_folds=config["splitting"]["n_folds"],
        test_size=config["splitting"]["test_size"],
        seed=config["splitting"]["random_seed"],
        dataset="PEVA",
    )

    # ===================================================================
    # Etapa 3: Treino MobileNetV2 com imagens cropadas
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 3: Treino MobileNetV2 (blob cropped)")
    print("=" * 60)

    crop_results = run_cropped_experiment(config, device, df_trainval, df_test, folds)

    # ===================================================================
    # Etapa 4: Tabela comparativa
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 4: Comparação com baseline")
    print("=" * 60)

    # Carregar resultados baseline
    baseline_path = os.path.join(logs_dir, "cnn_results.json")
    with open(baseline_path, "r") as f:
        all_baseline = json.load(f)
    baseline_results = all_baseline["mobilenetv2"]

    generate_comparison_table(baseline_results, crop_results, logs_dir)

    # Salvar resultados do crop experiment
    crop_log = {
        "fold_metrics": crop_results["fold_metrics"],
        "avg_metrics": crop_results["avg_metrics"],
        "test_metrics": crop_results["test_metrics"],
        "test_ci": crop_results["test_ci"],
        "training_time_seconds": crop_results["training_time_seconds"],
    }
    crop_log_path = os.path.join(logs_dir, "blob_crop_results.json")
    with open(crop_log_path, "w") as f:
        json.dump(crop_log, f, indent=2)
    print(f"\nResultados do crop salvos em: {crop_log_path}")

    # ===================================================================
    # Etapa 5: Grad-CAM comparativo
    # ===================================================================
    print("\n" + "=" * 60)
    print("ETAPA 5: Grad-CAM comparativo")
    print("=" * 60)

    # Para o Grad-CAM, precisamos do df_test original (sem crop) e com crop
    df_orig = pd.read_csv(registry_path)
    _, df_test_orig, _ = patient_split(
        df_orig, n_folds=config["splitting"]["n_folds"],
        test_size=config["splitting"]["test_size"],
        seed=config["splitting"]["random_seed"],
        dataset="PEVA",
    )

    baseline_ckpt = os.path.join(results_dir, "checkpoints", "mobilenetv2_final.pth")
    generate_gradcam_comparison(
        config, device,
        baseline_model_path=baseline_ckpt,
        crop_model=crop_results["model"],
        df_test_orig=df_test_orig,
        df_test_crop=df_test,
        output_dir=figures_dir,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTO CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()
