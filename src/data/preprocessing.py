"""Pré-processamento: filtro de qualidade, ROI, resize, normalização."""

import cv2
import numpy as np


# Estatísticas ImageNet para normalização (usadas em transfer learning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def quality_filter(img: np.ndarray, min_mean: float = 10, max_mean: float = 245) -> bool:
    """
    Verifica se a imagem passa no filtro de qualidade.
    Rejeita imagens muito escuras (pretas) ou muito claras (saturadas).
    """
    if img is None:
        return False
    mean_intensity = np.mean(img)
    return min_mean <= mean_intensity <= max_mean


def extract_roi(img: np.ndarray) -> np.ndarray:
    """
    Extrai região de interesse da imagem.
    Para imagens de plasma, tenta encontrar a região circular do poço.
    Se não encontrar, retorna a imagem original.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Tentar detectar o poço circular
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0] // 2,
        param1=50, param2=30,
        minRadius=gray.shape[0] // 4, maxRadius=gray.shape[0] // 2
    )

    if circles is not None:
        circle = np.round(circles[0, 0]).astype(int)
        cx, cy, r = circle
        # Crop quadrado ao redor do círculo
        y1 = max(0, cy - r)
        y2 = min(img.shape[0], cy + r)
        x1 = max(0, cx - r)
        x2 = min(img.shape[1], cx + r)
        return img[y1:y2, x1:x2]

    return img


def resize_image(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Redimensiona imagem para target_size × target_size."""
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)


def normalize_for_model(img: np.ndarray) -> np.ndarray:
    """
    Normaliza imagem para uso com modelos pré-treinados no ImageNet.
    Input: uint8 [0, 255] → Output: float32 normalizado.
    """
    img_float = img.astype(np.float32) / 255.0
    if len(img_float.shape) == 2:
        # Grayscale → 3 canais
        img_float = np.stack([img_float] * 3, axis=-1)
    img_float = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    return img_float


def preprocess_image(
    image_path: str,
    target_size: int = 224,
    apply_roi: bool = True,
    min_mean: float = 10,
    max_mean: float = 245,
) -> np.ndarray | None:
    """
    Pipeline completo de pré-processamento de uma imagem.

    Returns:
        Imagem pré-processada (uint8, grayscale, target_size×target_size)
        ou None se rejeitada pelo filtro de qualidade.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if not quality_filter(img, min_mean, max_mean):
        return None

    if apply_roi:
        img = extract_roi(img)

    img = resize_image(img, target_size)
    return img
