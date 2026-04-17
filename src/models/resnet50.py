from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Always train the classification head.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
