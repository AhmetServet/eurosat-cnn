"""Model factory for transfer learning on EuroSAT.

Provides functions that return a pretrained backbone with a frozen feature
extractor and a custom classification head.

Supported architectures (one per family):
  - vgg16_bn
  - resnet50
  - densenet121
  - efficientnet_b0
  - mobilenet_v2
"""

import torch.nn as nn
from torchvision import models


def _build_classifier(in_features: int, num_classes: int, dropout: float, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def create_vgg16_bn(num_classes: int = 10, dropout: float = 0.3, hidden_dim: int = 512, freeze_backbone: bool = True):
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze_backbone(model)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def create_resnet50(num_classes: int = 10, dropout: float = 0.3, hidden_dim: int = 512, freeze_backbone: bool = True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        _freeze_backbone(model)
    in_features = model.fc.in_features
    model.fc = _build_classifier(in_features, num_classes, dropout, hidden_dim)
    return model


def create_densenet121(num_classes: int = 10, dropout: float = 0.3, hidden_dim: int = 512, freeze_backbone: bool = True):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze_backbone(model)
    in_features = model.classifier.in_features
    model.classifier = _build_classifier(in_features, num_classes, dropout, hidden_dim)
    return model


def create_efficientnet_b0(num_classes: int = 10, dropout: float = 0.3, hidden_dim: int = 512, freeze_backbone: bool = True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze_backbone(model)
    in_features = model.classifier[1].in_features
    model.classifier = _build_classifier(in_features, num_classes, dropout, hidden_dim)
    return model


def create_mobilenet_v2(num_classes: int = 10, dropout: float = 0.3, hidden_dim: int = 512, freeze_backbone: bool = True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        _freeze_backbone(model)
    in_features = model.classifier[1].in_features
    model.classifier = _build_classifier(in_features, num_classes, dropout, hidden_dim)
    return model


MODEL_REGISTRY = {
    "vgg16_bn": create_vgg16_bn,
    "resnet50": create_resnet50,
    "densenet121": create_densenet121,
    "efficientnet_b0": create_efficientnet_b0,
    "mobilenet_v2": create_mobilenet_v2,
}


def create_model(model_name: str, **kwargs):
    """Create a model by name. Extra kwargs go to the constructor."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models():
    return list(MODEL_REGISTRY)
