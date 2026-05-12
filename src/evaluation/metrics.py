"""Métricas de avaliação: sensibilidade, especificidade, AUC, F1, CI bootstrap."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    recall_score, precision_score, accuracy_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """Calcula todas as métricas relevantes."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: np.ndarray = None,
                 n_bootstrap: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """Calcula intervalos de confiança bootstrap para as métricas."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    all_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_prob[idx] if y_prob is not None else None

        if len(np.unique(y_t)) < 2:
            continue

        all_metrics.append(compute_metrics(y_t, y_p, y_pr))

    if not all_metrics:
        return {}

    ci = {}
    for key in all_metrics[0]:
        if key in ("tp", "tn", "fp", "fn"):
            continue
        values = [m[key] for m in all_metrics if key in m]
        if values:
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            mean = np.mean(values)
            ci[key] = {"mean": round(mean, 4), "lower": round(lower, 4),
                       "upper": round(upper, 4)}

    return ci


def format_metric_with_ci(name: str, value: float, ci: dict = None) -> str:
    """Formata métrica com IC para exibição."""
    if ci and name in ci:
        return f"{name}: {value:.4f} [{ci[name]['lower']:.4f}, {ci[name]['upper']:.4f}]"
    return f"{name}: {value:.4f}"
