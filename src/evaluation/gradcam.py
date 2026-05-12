"""Grad-CAM: visualização de atenção na última camada convolucional."""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    """Grad-CAM para modelos timm."""

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: modelo CNN
            target_layer: camada alvo (se None, tenta detectar automaticamente)
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        if target_layer is None:
            target_layer = self._find_last_conv(model)

        self.target_layer = target_layer
        self._register_hooks()

    def _find_last_conv(self, model):
        """Encontra a última camada convolucional."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Gera mapa Grad-CAM.

        Args:
            input_tensor: tensor [1, C, H, W]
            target_class: classe alvo (se None, usa a predita)

        Returns:
            heatmap: numpy array [H, W] normalizado [0, 1]
            prediction: classe predita
            confidence: probabilidade da classe predita
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalizar
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize para tamanho da imagem de entrada
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (w, h))

        return heatmap, target_class, confidence


def overlay_heatmap(image, heatmap, alpha=0.4):
    """Sobrepõe heatmap Grad-CAM na imagem original."""
    # Imagem em grayscale → colorida
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Heatmap → colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(image_color, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
