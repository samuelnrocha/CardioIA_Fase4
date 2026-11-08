"""
Arquiteturas Convolucionais — CardioIA Fase 4 (Parte 2)
Grupo: Data Health Solutions (FIAP)

Implementa:
1. `CardioCNN_Scratch`: Rede Neural Convolucional construída do zero (4 blocos Conv2d +
   BatchNorm2d + ReLU + MaxPool/AdaptiveAvgPool + cabeçote denso com Dropout).
2. `get_cardio_resnet`: Transfer Learning baseado na arquitetura ResNet18 pré-treinada
   no ImageNet (`ResNet18_Weights.DEFAULT`), com congelamento das camadas iniciais e
   fine-tuning seletivo do bloco `layer4` + classificador customizado.
3. `export_model_onnx`: Exportação leve e interoperável do modelo para formato ONNX.
"""

import os
from typing import Dict
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CardioCNN_Scratch(nn.Module):
    """
    Arquitetura CNN proprietária construída do zero para detecção de Cardiomegalia
    em radiografias torácicas (224x224x3).
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.40):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 4: 28x28 -> 4x4 (AdaptiveAvgPool)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_cardio_resnet(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    unfreeze_layer4: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """
    Constrói o modelo de Transfer Learning baseado na ResNet18 pré-treinada no ImageNet.
    Permite congelar os extratores de baixo nível e liberar `layer4` + cabeça customizada
    para ajuste fino (fine-tuning) no domínio radiológico.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception:
        # Fallback seguro caso o ambiente esteja 100% offline sem cache de pesos ImageNet
        model = resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        if unfreeze_layer4:
            for param in model.layer4.parameters():
                param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.35),
        nn.Linear(256, num_classes),
    )
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calcula total de parâmetros e parâmetros treináveis do modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def export_model_onnx(
    model: nn.Module,
    onnx_path: str,
    device: torch.device = torch.device("cpu"),
    img_size: int = 224,
) -> str:
    """Exporta um modelo PyTorch treinado para formato interoperável ONNX."""
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    model_eval = model.to(device).eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model_eval,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_xray"],
            output_names=["logits"],
            dynamic_axes={"input_xray": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False,
        )
    return onnx_path
