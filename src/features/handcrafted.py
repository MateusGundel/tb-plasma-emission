"""Features handcrafted: HOG, LBP, histograma, GLCM, Hu moments."""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.feature import hog as skimage_hog


def extract_hog(img: np.ndarray) -> np.ndarray:
    """HOG features."""
    features = skimage_hog(
        img, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), feature_vector=True,
    )
    return features


def extract_lbp(img: np.ndarray, n_points: int = 24, radius: int = 3) -> np.ndarray:
    """LBP uniform histogram."""
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    n_bins = n_points + 2  # uniform LBP
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_histogram(img: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Histograma de intensidade normalizado."""
    hist = cv2.calcHist([img], [0], None, [n_bins], [0, 256])
    hist = hist.flatten() / (img.shape[0] * img.shape[1])
    return hist


def extract_glcm(img: np.ndarray) -> np.ndarray:
    """GLCM texture features (contrast, dissimilarity, homogeneity, energy, correlation)."""
    # Quantizar para menos níveis para GLCM
    img_q = (img // 4).astype(np.uint8)  # 64 níveis
    glcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)
    props = []
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).flatten()
        props.extend([vals.mean(), vals.std()])
    return np.array(props)


def extract_hu_moments(img: np.ndarray) -> np.ndarray:
    """Hu moments (7 valores, log-transformados)."""
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    # Log transform para escala mais tratável
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu_log


def extract_all_features(img: np.ndarray) -> np.ndarray:
    """Extrai todas as features handcrafted de uma imagem grayscale 224×224."""
    features = np.concatenate([
        extract_hog(img),
        extract_lbp(img),
        extract_histogram(img),
        extract_glcm(img),
        extract_hu_moments(img),
    ])
    return features
