# 04 — Parte 2: Modelos CNN e Transfer Learning
## Arquiteturas Convolucionais, Treinamento via Colab-MCP e Benchmark Clínico

---

## 1. Visão Geral dos Dois Modelos

O edital da Fase 4 exige a implementação e comparação empírica de **duas abordagens de aprendizado profundo**:
1. **CNN Construída do Zero (*from scratch*):** Uma arquitetura proprietária e enxuta, permitindo entender o aprendizado de representações espaciais a partir dos dados do projeto.
2. **Transfer Learning com Modelo Pré-treinado:** Aproveitamento do conhecimento prévio de modelos de grande porte treinados no ImageNet (como **ResNet18/50** ou **VGG16**), adaptando as camadas finais para o domínio médico.

```
┌────────────────────────────────────────┬────────────────────────────────────────┐
│     CNN DO ZERO (CardioCNN)            │   TRANSFER LEARNING (CardioResNet)     │
├────────────────────────────────────────┼────────────────────────────────────────┤
│ • 3 a 4 Blocos Conv2D + BatchNorm      │ • Backbone ResNet pré-treinado         │
│ • Pooling Progressivo (MaxPooling2D)   │ • Camadas convolucionais congeladas    │
│ • Dropout (0.3 a 0.5) para evitar over │ • Nova cabeça densa treinada no subset │
│ • Parâmetros: ~500k a 1.5M             │ • Fine-tuning seletivo nas últimas lrs │
│ • Treinamento rápido, explicabilidade  │ • Parâmetros: ~11M a 25M               │
│ • Maior sensibilidade a desbalanceio   │ • Excelente generalização e ROC-AUC    │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 2. Especificação Arquitetural Sugerida

### 2.1 Arquitetura da CNN do Zero (`CardioCNN_Scratch`)
- **Entrada:** Tensor $[B, 3, 224, 224]$
- **Bloco 1:** `Conv2D(3, 32, kernel=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
- **Bloco 2:** `Conv2D(32, 64, kernel=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
- **Bloco 3:** `Conv2D(64, 128, kernel=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)`
- **Bloco 4:** `Conv2D(128, 256, kernel=3, padding=1)` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d((4, 4))`
- **Classificador:**
  - `Flatten` $\rightarrow$ `Linear(256 * 4 * 4, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(128, 2)` (ou saída 1 com Sigmoid para binário).

### 2.2 Estratégia de Transfer Learning (`CardioTransfer_ResNet`)
- **Backbone:** `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` ou `resnet50`.
- **Congelamento:** Travar parâmetros iniciais:
  ```python
  for param in model.parameters():
      param.requires_grad = False
  ```
- **Cabeça Substituta (Custom Head):**
  ```python
  num_ftrs = model.fc.in_features
  model.fc = nn.Sequential(
      nn.Linear(num_ftrs, 256),
      nn.ReLU(),
      nn.Dropout(0.35),
      nn.Linear(256, 2) # [Normal, Cardiomegalia]
  )
  ```
- **Fine-Tuning Opcional:** Descongelar o último bloco residual (`layer4`) nas últimas épocas com uma taxa de aprendizado reduzida ($lr = 10^{-5}$).

---

## 3. Protocolo de Treinamento com `colab-mcp`

Para realizar o treinamento com aceleração de hardware:
1. Abra a conexão remota com `colab-mcp`:
   ```json
   {
     "ServerName": "colab-mcp",
     "ToolName": "open_colab_browser_connection",
     "Arguments": {}
   }
   ```
2. Execute o notebook `notebooks/02_classificacao_cnn_e_transfer_learning.ipynb` utilizando GPU (`cuda`).
3. Otimizador recomendado: `AdamW` ($lr = 10^{-3}$ para CNN do zero, $lr = 10^{-4}$ para Transfer Learning) com *scheduler* `CosineAnnealingLR` ou `ReduceLROnPlateau`.
4. Função de Perda (*Loss*): `CrossEntropyLoss` ponderada caso haja desbalanceamento de classes (`weight=tensor([1.0, peso_classe_minoritaria])`).
5. **Salvamento do Checkpoint:** O melhor modelo deve ser exportado:
   - PyTorch: `backend/models/best_cardio_resnet.pt`
   - Formato interoperável ONNX: `backend/models/cardio_resnet.onnx` (para garantir inferência ultrarrápida sem depender de PyTorch no mobile/backend leve).

---

## 4. Métricas de Avaliação Mandatórias

Para cumprir todos os critérios de avaliação:
- **Acurácia Global (%)**
- **Precisão (Precision):** Proporção de verdadeiros positivos sobre todas as predições de cardiomegalia.
- **Sensibilidade / Revocação (Recall):** Crucial na medicina — quantos casos reais de cardiomegalia o modelo não deixou escapar.
- **F1-Score:** Média harmônica entre Precisão e Recall.
- **Curva ROC e ROC-AUC (%):** Capacidade discriminativa do modelo em diferentes limiares.
- **Matriz de Confusão Normalizada:** Exibindo claramente Verdadeiros Positivos, Verdadeiros Negativos, Falsos Positivos e Falsos Negativos.

### Tabela Comparativa de Resultados Esperada no Relatório

| Métrica | CNN Customizada (Do Zero) | Transfer Learning (ResNet18/50) | Diferença (%) |
| :--- | :---: | :---: | :---: |
| **Acurácia no Teste** | ~82% - 88% | ~91% - 96% | +5% a +10% |
| **Recall / Sensibilidade** | ~80% - 85% | ~92% - 97% | Vantagem clínica |
| **Precisão** | ~81% - 87% | ~90% - 95% | Menos falsos alarmes |
| **F1-Score** | ~81% - 86% | ~91% - 96% | Estabilidade |
| **ROC-AUC** | ~87% - 92% | ~96% - 99% | Alta separabilidade |
| **Tamanho do Modelo** | ~8 MB | ~45 MB | Trade-off de recursos |

---

## 5. Protótipo de Apresentação dos Resultados

O edital solicita um protótipo de apresentação. O agente pode contemplar:
1. **Interface Interativa no Próprio Notebook:** Utilizando widgets (ex.: `ipywidgets`) para upload de imagem e exibição de barras de probabilidade instantâneas.
2. **Interface Web Leve (Flask):** Interface estática responsiva no `backend/` onde o usuário seleciona um exame e vê o diagnóstico gerado.
