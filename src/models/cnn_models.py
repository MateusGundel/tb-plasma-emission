"""CNN models com transfer learning: ResNet-50, EfficientNet-B0, MobileNetV2."""

import torch.nn as nn
import timm


def create_cnn_model(arch_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Cria modelo CNN com head customizado para classificação binária.

    Args:
        arch_name: "resnet50", "efficientnet_b0", "mobilenetv2"
        num_classes: número de classes
        pretrained: usar pesos pré-treinados do ImageNet
    """
    model_map = {
        "resnet50": "resnet50",
        "efficientnet_b0": "efficientnet_b0",
        "mobilenetv2": "mobilenetv2_100",
    }

    timm_name = model_map.get(arch_name, arch_name)
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module):
    """Congela todos os parâmetros exceto a cabeça de classificação."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name and "head" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Descongela todos os parâmetros para fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module) -> int:
    """Conta parâmetros treináveis."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
