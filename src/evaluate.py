"""
Avaliação Clínica e Métricas Diagnósticas — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)

Calcula:
- Acurácia Global
- Precisão (Valor Preditivo Positivo)
- Sensibilidade / Recall (Taxa de Verdadeiros Positivos)
- Especificidade (Taxa de Verdadeiros Negativos)
- F1-Score
- ROC-AUC e coordenadas da Curva ROC
- Matriz de Confusão 2x2
- Vetores de metadados demográficos para auditoria de Fairness
"""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Avalia o modelo em um DataLoader e retorna dicionário completo com métricas clínicas,
    curvas ROC, matriz de confusão e metadados de subgrupos para análise ética.
    """
    model = model.to(device)
    model.eval()

    y_true: List[int] = []
    y_probs: List[float] = []
    genders: List[str] = []
    age_groups: List[str] = []
    view_positions: List[str] = []
    patient_ids: List[str] = []

    with torch.no_grad():
        for images, labels, meta in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            y_true.extend(labels.numpy().tolist())
            y_probs.extend(probs.tolist())

            if isinstance(meta, dict):
                genders.extend(list(meta.get("gender", [])))
                age_groups.extend(list(meta.get("age_group", [])))
                view_positions.extend(list(meta.get("view_position", [])))
                patient_ids.extend(list(meta.get("patient_id", [])))

    y_true_arr = np.array(y_true, dtype=int)
    y_probs_arr = np.array(y_probs, dtype=float)
    y_pred_arr = (y_probs_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    if len(np.unique(y_true_arr)) > 1:
        auc = float(roc_auc_score(y_true_arr, y_probs_arr))
        fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_probs_arr)
    else:
        auc = 0.5
        fpr, tpr, roc_thresholds = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / max(1, (tn + fp)))

    return {
        "acuracia": acc,
        "precisao": prec,
        "recall_sensibilidade": rec,
        "especificidade": specificity,
        "f1_score": f1,
        "roc_auc": auc,
        "matriz_confusao": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "y_probs": y_probs_arr,
        "gender": np.array(genders),
        "age_group": np.array(age_groups),
        "view_position": np.array(view_positions),
        "patient_id": np.array(patient_ids),
    }
