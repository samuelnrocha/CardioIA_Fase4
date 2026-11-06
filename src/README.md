# Módulos Python — CardioIA Fase 4

Contém os módulos Python reutilizáveis e testáveis:

- `dataset.py`: Classes PyTorch Dataset e funções de carregamento de DataLoader.
- `models.py`: Arquiteturas da CNN construída do zero (`CardioCNN_Scratch`) e Transfer Learning (`CardioTransfer_ResNet`).
- `train.py`: Rotinas de treinamento, otimização (AdamW), scheduler e salvamento de checkpoints.
- `evaluate.py`: Cálculo de métricas clínicas (Acurácia, Sensibilidade/Recall, Precisão, F1-Score, Curva ROC-AUC).
- `fairness.py`: Auditoria de disparidade de impacto (DIR) e paridade de oportunidade.

Consulte o guia [`knowledge_base/08_SCRIPTS_E_SNIPPETS_DE_REFERENCIA.md`](../knowledge_base/08_SCRIPTS_E_SNIPPETS_DE_REFERENCIA.md) para código de referência.
