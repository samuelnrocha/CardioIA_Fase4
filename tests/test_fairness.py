import numpy as np
import pytest

from src.fairness import compute_fairness_audit, mitigate_equal_opportunity_thresholds


def test_compute_fairness_audit_metrics():
    # Grupo M (favorecido) vs Grupo F (com subdiagnóstico antes da mitigação)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.3, 0.2, 0.85, 0.75, 0.42, 0.44, 0.1, 0.2, 0.15, 0.05])
    y_pred = (y_probs >= 0.50).astype(int)
    sensitive = np.array(["M"] * 8 + ["F"] * 8)

    audit = compute_fairness_audit(y_true, y_pred, sensitive, y_probs=y_probs)
    assert "subgrupos" in audit
    assert "M" in audit["subgrupos"] and "F" in audit["subgrupos"]
    assert "disparate_impact_ratio" in audit
    assert "demographic_parity_difference" in audit
    assert "equal_opportunity_difference" in audit
    assert "predictive_parity_difference" in audit

    # No grupo M recall = 4/4 = 1.0; no grupo F recall = 2/4 = 0.50 (com threshold 0.5)
    assert np.isclose(audit["subgrupos"]["M"]["recall"], 1.0)
    assert np.isclose(audit["subgrupos"]["F"]["recall"], 0.5)
    assert np.isclose(audit["equal_opportunity_difference"], 0.5)
    assert np.isclose(audit["disparate_impact_ratio"], 0.5)


def test_mitigate_equal_opportunity_thresholds_restores_fairness():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    y_probs = np.array([0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.3, 0.2, 0.85, 0.75, 0.42, 0.44, 0.1, 0.2, 0.15, 0.05])
    sensitive = np.array(["M"] * 8 + ["F"] * 8)

    mitigated = mitigate_equal_opportunity_thresholds(
        y_true=y_true,
        y_probs=y_probs,
        sensitive_attributes=sensitive,
        target_recall=0.95,
    )
    assert "group_thresholds" in mitigated
    assert "y_pred_mitigated" in mitigated
    assert "audit_after" in mitigated

    audit_after = mitigated["audit_after"]
    # Após calibração de limiar por grupo, EOD cai para ~0 e DIR sobe para >= 0.80
    assert audit_after["equal_opportunity_difference"] <= 0.05
    assert audit_after["disparate_impact_ratio"] >= 0.80
