"""PyTorch Dataset para imagens de plasma."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_image, IMAGENET_MEAN, IMAGENET_STD


class PlasmaDataset(Dataset):
    """Dataset PyTorch para imagens de plasma."""

    def __init__(self, df, target_size=224, transform=None, is_train=False):
        """
        Args:
            df: DataFrame com colunas image_path, label
            target_size: tamanho de resize
            transform: albumentations transform
            is_train: se True, aplica augmentation
        """
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback: imagem preta
            img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        # Resize
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        # Augmentation (apenas treino)
        if self.transform is not None and self.is_train:
            result = self.transform(image=img)
            img = result["image"]

        # Grayscale → 3 canais
        img_3ch = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0

        # Normalização ImageNet
        img_3ch = (img_3ch - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW
        img_tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1)).float()
        label = torch.tensor(row["label"], dtype=torch.long)

        return img_tensor, label
