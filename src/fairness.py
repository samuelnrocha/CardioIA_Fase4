"""
Auditoria de Vieses Demográficos e Métricas Formais de Fairness — CardioIA Fase 4 (Ir Além 1)
Grupo: Data Health Solutions (FIAP)

Implementa:
1. `compute_fairness_audit`: Cálculo de métricas estratificadas por atributo sensível
   (Sexo/Gênero: M vs F; Faixa Etária: Jovem, Adulto, Idoso; Incidência Técnica: AP vs PA):
   - Taxa de Seleção Positiva (Demographic Parity)
   - Disparate Impact Ratio (DIR — Regra dos Quatro Quintos / 80% da EEOC)
   - Demographic Parity Difference (DPD)
   - Equal Opportunity Difference (EOD — Diferença na Sensibilidade/Recall)
   - Predictive Parity Difference (PPD — Diferença na Precisão/VPP)
   - Taxa de Falsos Negativos (FNR — Subdiagnóstico / Underdiagnosis Rate)
2. `mitigate_equal_opportunity_thresholds`: Calibração pós-processamento de limiares clínicos
   por subpopulação (Equalized Odds / Equal Opportunity Post-Processing) para eliminar
   disparidade de sensibilidade sem degradar a especificidade global.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_fairness_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attributes: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas clínicas por subgrupo demográfico e indicadores matemáticos
    de Justiça Algorítmica (Fairness).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    sensitive_attributes = np.asarray(sensitive_attributes)

    subgroups = sorted([str(g) for g in np.unique(sensitive_attributes)])
    audit_subgroups: Dict[str, Dict[str, float]] = {}

    for grp in subgroups:
        mask = sensitive_attributes == grp
        yt = y_true[mask]
        yp = y_pred[mask]

        acc = float(accuracy_score(yt, yp)) if len(yt) > 0 else 0.0
        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))
        pos_rate = float(np.mean(yp)) if len(yp) > 0 else 0.0
        prevalence = float(np.mean(yt)) if len(yt) > 0 else 0.0
        fnr = float(1.0 - rec)  # Underdiagnosis rate

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / max(1, (fp + tn)))

        auc_val = 0.0
        if y_probs is not None and len(np.unique(yt)) > 1:
            auc_val = float(roc_auc_score(yt, np.asarray(y_probs)[mask]))

        audit_subgroups[grp] = {
            "total_amostras": int(np.sum(mask)),
            "prevalencia_real": prevalence,
            "taxa_predicao_positiva": pos_rate,
            "acuracia": acc,
            "precisao": prec,
            "recall": rec,
            "fnr_subdiagnostico": fnr,
            "fpr_falso_alarme": fpr,
            "f1_score": f1,
            "roc_auc": auc_val,
        }

    pos_rates = [m["taxa_predicao_positiva"] for m in audit_subgroups.values() if m["total_amostras"] > 0]
    recalls = [m["recall"] for m in audit_subgroups.values() if m["total_amostras"] > 0]
    precisions = [m["precisao"] for m in audit_subgroups.values() if m["total_amostras"] > 0]

    max_pos = max(pos_rates) if pos_rates else 1.0
    min_pos = min(pos_rates) if pos_rates else 0.0
    dir_ratio = float(min_pos / max(1e-6, max_pos))
    dp_diff = float(max_pos - min_pos)
    eod = float(max(recalls) - min(recalls)) if recalls else 0.0
    ppd = float(max(precisions) - min(precisions)) if precisions else 0.0

    return {
        "subgrupos": audit_subgroups,
        "disparate_impact_ratio": dir_ratio,
        "demographic_parity_difference": dp_diff,
        "equal_opportunity_difference": eod,
        "predictive_parity_difference": ppd,
        "eeoc_80pct_compliant": bool(dir_ratio >= 0.80),
        "equal_opportunity_compliant": bool(eod <= 0.06),
    }


def mitigate_equal_opportunity_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    sensitive_attributes: np.ndarray,
    target_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Aplica mitigação de viés pós-processamento via Calibração de Limiar por Subgrupo
    (Equal Opportunity Thresholding), garantindo que grupos subdiagnosticados atinjam
    a mesma Sensibilidade (Recall) do grupo melhor assistido.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    sensitive_attributes = np.asarray(sensitive_attributes)

    y_pred_before = (y_probs >= 0.50).astype(int)
    audit_before = compute_fairness_audit(y_true, y_pred_before, sensitive_attributes, y_probs=y_probs)

    if target_recall is None:
        recalls_before = [m["recall"] for m in audit_before["subgrupos"].values()]
        target_recall = max(recalls_before) if recalls_before else 0.92

    group_thresholds: Dict[str, float] = {}
    y_pred_mitigated = np.zeros_like(y_true, dtype=int)

    for grp in np.unique(sensitive_attributes):
        mask = sensitive_attributes == grp
        yt_grp = y_true[mask]
        yp_grp = y_probs[mask]

        pos_probs = np.sort(yp_grp[yt_grp == 1])
        if len(pos_probs) > 0:
            # Seleciona o quantil (1 - target_recall) das probabilidades dos casos positivos reais
            q_idx = int(np.floor((1.0 - target_recall) * len(pos_probs)))
            q_idx = int(np.clip(q_idx, 0, len(pos_probs) - 1))
            optimal_thr = float(np.clip(pos_probs[q_idx] - 1e-4, 0.20, 0.75))
        else:
            optimal_thr = 0.50

        group_thresholds[str(grp)] = round(optimal_thr, 4)
        y_pred_mitigated[mask] = (yp_grp >= optimal_thr).astype(int)

    audit_after = compute_fairness_audit(y_true, y_pred_mitigated, sensitive_attributes, y_probs=y_probs)

    return {
        "target_recall": float(target_recall),
        "group_thresholds": group_thresholds,
        "y_pred_before": y_pred_before,
        "y_pred_mitigated": y_pred_mitigated,
        "audit_before": audit_before,
        "audit_after": audit_after,
    }
