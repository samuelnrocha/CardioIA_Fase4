"""
Motor de Treinamento Supervisionado e Fine-Tuning — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)
"""

import os
import copy
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """
    Executa o loop de treinamento e validação com AdamW, CosineAnnealingLR e
    persistência automática do melhor checkpoint (baseado em acurácia/loss de validação).
    """
    model = model.to(device)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_s": [],
    }

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validação
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["epoch_time_s"].append(float(elapsed))

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(best_weights, save_path)

    model.load_state_dict(best_weights)
    if save_path and not os.path.exists(save_path):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(best_weights, save_path)

    return history
