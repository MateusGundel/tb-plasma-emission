"""Registry do dataset: mapeia imagens → paciente, sessão, label, dataset."""

import os
import re
import glob
import pandas as pd


# Regex para extrair patient_id da pasta: "20250713-2030_453" → "453"
# Sessões: "20250713-2030_453-1" → patient_id="453", session="1"
FOLDER_REGEX = re.compile(r"^(\d{8}-\d{4})_(\d+)(?:-(\d+))?$")


def parse_folder_name(folder_name: str) -> dict | None:
    """Extrai timestamp, patient_id e session_id do nome da pasta."""
    match = FOLDER_REGEX.match(folder_name)
    if not match:
        return None
    return {
        "timestamp": match.group(1),
        "patient_id": int(match.group(2)),
        "session_id": int(match.group(3)) if match.group(3) else 0,
    }


def build_registry(dataset_dir: str, dataset_name: str, image_ext: str = "*.bmp") -> pd.DataFrame:
    """
    Constrói registro de todas as imagens de um dataset.

    Args:
        dataset_dir: Caminho raiz do dataset (ex: /path/PEVA)
        dataset_name: Nome do dataset (ex: "PEVA")
        image_ext: Extensão das imagens

    Returns:
        DataFrame com colunas: image_path, dataset, patient_id, session_id, label, folder_name
    """
    records = []

    for label_name in ["POSITIVE", "NEGATIVE"]:
        label_dir = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"AVISO: diretório não encontrado: {label_dir}")
            continue

        label = 1 if label_name == "POSITIVE" else 0

        for folder_name in sorted(os.listdir(label_dir)):
            folder_path = os.path.join(label_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            parsed = parse_folder_name(folder_name)
            if parsed is None:
                print(f"AVISO: pasta com formato inesperado: {folder_name}")
                continue

            images = sorted(glob.glob(os.path.join(folder_path, image_ext)))
            for img_path in images:
                records.append({
                    "image_path": img_path,
                    "dataset": dataset_name,
                    "patient_id": parsed["patient_id"],
                    "session_id": parsed["session_id"],
                    "label": label,
                    "label_name": label_name,
                    "folder_name": folder_name,
                })

    df = pd.DataFrame(records)
    return df


def deduplicate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sessões duplicadas, mantendo apenas a última versão.

    O padrão é: timestamp_ID-versão. Se versão=0 (sem sufixo), é a sessão original.
    Se há versões >0, são recoletas/duplicatas — usamos apenas a de maior session_id
    para cada combinação (dataset, patient_id, timestamp, label).
    """
    if df.empty:
        return df

    # Extrair timestamp da folder_name para agrupar
    df = df.copy()
    df["timestamp"] = df["folder_name"].str.extract(r"^(\d{8}-\d{4})_")

    before = len(df)
    before_folders = df["folder_name"].nunique()

    # Para cada grupo (dataset, label, patient_id, timestamp), manter só a maior session_id
    idx = df.groupby(["dataset", "label", "patient_id", "timestamp"])["session_id"].idxmax()
    # Pegar as folder_names que devem ser mantidas
    keep_folders = df.loc[idx, "folder_name"].unique()
    df_filtered = df[df["folder_name"].isin(keep_folders)].copy()

    after = len(df_filtered)
    after_folders = df_filtered["folder_name"].nunique()
    removed = before - after
    if removed > 0:
        print(f"  Deduplicação: {before_folders} pastas → {after_folders} pastas "
              f"({before} → {after} imagens, {removed} removidas)")
    else:
        print(f"  Sem duplicatas encontradas")

    df_filtered.drop(columns=["timestamp"], inplace=True)
    return df_filtered.reset_index(drop=True)


def build_full_registry(config: dict) -> pd.DataFrame:
    """Constrói registry combinando PEVA e PESC."""
    dfs = []
    for name, path_key in [("PEVA", "peva_dir"), ("PESC", "pesc_dir")]:
        dataset_dir = config["paths"][path_key]
        if os.path.isdir(dataset_dir):
            df = build_registry(dataset_dir, name, config["data"]["image_extension"])
            print(f"{name} (bruto): {len(df)} imagens, "
                  f"{df['patient_id'].nunique()} pacientes únicos, "
                  f"POS={len(df[df['label']==1])}, NEG={len(df[df['label']==0])}")
            df = deduplicate_sessions(df)
            print(f"{name} (final): {len(df)} imagens, "
                  f"POS={len(df[df['label']==1])}, NEG={len(df[df['label']==0])}")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def validate_registry(df: pd.DataFrame):
    """Validações do registry."""
    print("\n=== Validação do Registry ===")

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        pos_patients = sub[sub["label"] == 1]["patient_id"].unique()
        neg_patients = sub[sub["label"] == 0]["patient_id"].unique()
        overlap = set(pos_patients) & set(neg_patients)

        print(f"\n{ds}:")
        print(f"  Total imagens: {len(sub)}")
        print(f"  Pacientes POS: {len(pos_patients)} ({sorted(pos_patients)})")
        print(f"  Pacientes NEG: {len(neg_patients)}")
        print(f"  Imagens POS: {len(sub[sub['label']==1])}")
        print(f"  Imagens NEG: {len(sub[sub['label']==0])}")
        if overlap:
            print(f"  ⚠ ALERTA: pacientes em ambas as classes: {overlap}")
        else:
            print(f"  ✓ Sem IDs ambíguos entre classes")
