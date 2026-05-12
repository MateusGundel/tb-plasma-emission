"""Data augmentation com albumentations — apenas para treino."""

import albumentations as A
import numpy as np


def get_train_transforms(config: dict) -> A.Compose:
    """Transforms de augmentation para treino."""
    aug = config["augmentation"]
    return A.Compose([
        A.Rotate(limit=aug["rotation_limit"], p=0.8, border_mode=0),
        A.HorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0.0),
        A.VerticalFlip(p=0.5 if aug["vertical_flip"] else 0.0),
        A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_limit"],
            contrast_limit=0.0,
            p=0.5,
        ),
        A.GaussNoise(p=0.3),
    ])


def get_val_transforms() -> A.Compose:
    """Transforms para validação/teste — apenas redimensionamento, sem augmentation."""
    return A.Compose([])


def apply_augmentation(img: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Aplica transform a uma imagem."""
    result = transform(image=img)
    return result["image"]
