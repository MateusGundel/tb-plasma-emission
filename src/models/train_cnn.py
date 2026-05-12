"""Loop de treino CNN: 2 fases (frozen + fine-tune), early stopping, mixed precision."""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from src.models.cnn_models import freeze_backbone, unfreeze_all, get_trainable_params


class EarlyStopping:
    """Early stopping baseado em AUC de validação."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Treina uma época."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Avalia no conjunto de validação."""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    auc = 0.0
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, acc, auc, all_probs, all_labels


def train_model(
    model, train_loader, val_loader, config, device,
    fold_idx: int = 0, arch_name: str = "",
) -> dict:
    """
    Treino em 2 fases: backbone congelado + fine-tuning completo.

    Returns:
        dict com histórico de treino e métricas finais
    """
    cnn_config = config["cnn"]
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "phase": []}

    # Pesos balanceados para a loss
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)
    n_neg = (train_labels == 0).sum()
    n_pos = (train_labels == 1).sum()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Mixed precision
    scaler = GradScaler("cuda") if cnn_config["mixed_precision"] and device.type == "cuda" else None

    early_stop = EarlyStopping(patience=cnn_config["early_stopping_patience"])

    # === FASE 1: Backbone congelado ===
    print(f"  [Fase 1] Backbone congelado — {cnn_config['frozen_epochs']} épocas")
    freeze_backbone(model)
    print(f"    Parâmetros treináveis: {get_trainable_params(model):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cnn_config["learning_rate"],
        weight_decay=cnn_config["weight_decay"],
    )

    for epoch in range(cnn_config["frozen_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device, scaler)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["phase"].append("frozen")
        print(f"    Época {epoch+1}/{cnn_config['frozen_epochs']}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_auc={val_auc:.4f}")

    # === FASE 2: Fine-tuning completo ===
    print(f"  [Fase 2] Fine-tuning — até {cnn_config['finetune_epochs']} épocas")
    unfreeze_all(model)
    print(f"    Parâmetros treináveis: {get_trainable_params(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cnn_config["finetune_lr"],
        weight_decay=cnn_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    for epoch in range(cnn_config["finetune_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device, scaler)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["phase"].append("finetune")

        print(f"    Época {epoch+1}/{cnn_config['finetune_epochs']}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_auc={val_auc:.4f}")

        if early_stop(val_auc, model):
            print(f"    Early stopping na época {epoch+1}")
            break

    early_stop.load_best(model)

    # Avaliação final
    val_loss, val_acc, val_auc, val_probs, val_labels = evaluate(
        model, val_loader, criterion, device
    )

    return {
        "history": history,
        "best_val_auc": float(early_stop.best_score or val_auc),
        "final_val_auc": float(val_auc),
        "final_val_acc": float(val_acc),
    }
