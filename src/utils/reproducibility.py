"""Utilitários de reprodutibilidade: seeds, device, configuração."""

import os
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Fixa todas as seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detecta GPU disponível."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Usando CPU")
    return device


def load_config(config_path: str = None) -> dict:
    """Carrega configuração YAML do experimento."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "configs", "experiment_config.yaml"
        )
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_experiment(config: dict = None) -> tuple[dict, torch.device]:
    """Setup completo: carrega config, fixa seed, detecta device."""
    if config is None:
        config = load_config()
    seed = config["reproducibility"]["seed"]
    set_seed(seed)
    device = get_device()
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    os.makedirs(config["paths"]["figures_dir"], exist_ok=True)
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
    return config, device
