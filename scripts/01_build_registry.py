#!/usr/bin/env python3
"""Script 01: Constrói o registro de pacientes e imagens."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.reproducibility import load_config
from src.data.dataset_registry import build_full_registry, validate_registry


def main():
    config = load_config()
    output_path = os.path.join(config["paths"]["results_dir"], "patient_registry.csv")

    print("Construindo registry de imagens...\n")
    df = build_full_registry(config)

    if df.empty:
        print("ERRO: nenhuma imagem encontrada!")
        sys.exit(1)

    validate_registry(df)

    df.to_csv(output_path, index=False)
    print(f"\nRegistry salvo em: {output_path}")
    print(f"Total: {len(df)} linhas")


if __name__ == "__main__":
    main()
