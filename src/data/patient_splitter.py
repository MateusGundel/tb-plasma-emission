"""Divisão de dados por paciente: StratifiedGroupKFold + hold-out test."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def patient_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_size: float = 0.15,
    seed: int = 42,
    dataset: str = "PEVA",
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Divide o dataset por paciente para evitar data leakage.

    1. Separa um hold-out test set (~test_size dos pacientes)
    2. No restante, aplica StratifiedGroupKFold

    Returns:
        df_trainval: DataFrame de treino+validação
        df_test: DataFrame de teste (hold-out)
        folds: lista de (train_idx, val_idx) referentes a df_trainval
    """
    sub = df[df["dataset"] == dataset].copy().reset_index(drop=True)
    rng = np.random.RandomState(seed)

    # Separar pacientes por classe
    patient_labels = sub.groupby("patient_id")["label"].first()
    pos_patients = patient_labels[patient_labels == 1].index.tolist()
    neg_patients = patient_labels[patient_labels == 0].index.tolist()

    # Hold-out: selecionar ~test_size dos pacientes de cada classe
    n_pos_test = max(1, round(len(pos_patients) * test_size))
    n_neg_test = max(1, round(len(neg_patients) * test_size))

    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    test_patients = set(pos_patients[:n_pos_test]) | set(neg_patients[:n_neg_test])
    trainval_patients = set(pos_patients[n_pos_test:]) | set(neg_patients[n_neg_test:])

    df_test = sub[sub["patient_id"].isin(test_patients)].reset_index(drop=True)
    df_trainval = sub[sub["patient_id"].isin(trainval_patients)].reset_index(drop=True)

    print(f"Hold-out test: {len(test_patients)} pacientes ({len(df_test)} imagens)")
    print(f"  POS: {len(df_test[df_test['label']==1])} imgs, "
          f"NEG: {len(df_test[df_test['label']==0])} imgs")
    print(f"Train+Val: {len(trainval_patients)} pacientes ({len(df_trainval)} imagens)")

    # StratifiedGroupKFold no train+val
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    groups = df_trainval["patient_id"].values
    labels = df_trainval["label"].values

    folds = list(sgkf.split(df_trainval, labels, groups))

    # Validar que nenhum paciente está em train e val do mesmo fold
    for i, (train_idx, val_idx) in enumerate(folds):
        train_pats = set(df_trainval.iloc[train_idx]["patient_id"])
        val_pats = set(df_trainval.iloc[val_idx]["patient_id"])
        overlap = train_pats & val_pats
        if overlap:
            print(f"  ⚠ FOLD {i}: pacientes em train E val: {overlap}")
        else:
            n_val_pos = sum(df_trainval.iloc[val_idx]["label"] == 1)
            n_val_neg = sum(df_trainval.iloc[val_idx]["label"] == 0)
            print(f"  Fold {i}: train={len(train_idx)}, val={len(val_idx)} "
                  f"(POS={n_val_pos}, NEG={n_val_neg}) ✓")

    return df_trainval, df_test, folds
