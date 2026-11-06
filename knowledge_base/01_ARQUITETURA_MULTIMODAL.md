# 01 — Arquitetura de Software e Multimodalidade
## Da Telemetria IoT (Fase 3) à Visão Computacional Diagnóstica (Fase 4)

---

## 1. Visão Arquitetural Sistêmica

A grande força técnica do ecossistema **CardioIA** perante a banca avaliadora da FIAP é a sua **natureza multimodal e evolutiva**. Enquanto muitas soluções acadêmicas entregam modelos de visão computacional isolados da prática médica, o CardioIA conecta o ciclo completo de cuidados:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ARQUITETURA CARDIOIA                             │
│                                                                             │
│  [CAMADA 1: Borda / IoT (Fase 3)]                                           │
│  ESP32 Vestível ──► Sensores (DHT22, Pulso BPM, MPU6050)                    │
│      │                                                                      │
│      ├── Queda de Wi-Fi? ──► Memória Flash SPIFFS (10k FIFO, 13.8h)         │
│      └── Wi-Fi Ativo?    ──► MQTT (broker.hivemq.com)                       │
│                                      │                                      │
│  [CAMADA 2: Fog & Nuvem (Fase 3)]    ▼                                      │
│  Dashboard Node-RED ◄────── Ingestão Telemetria Contínua                    │
│  API REST (FastAPI) ◄────── Ingestão Sinais Vitais + RPA E-mail             │
│                                      │                                      │
│  [CAMADA 3: Diagnóstico por Imagem (Fase 4 - Nova)]                         │
│  Raio-X de Tórax / ECG Imagem ──► Pipeline CNN (ResNet / VGG16)             │
│                                      │                                      │
│  [CAMADA 4: Fusão Multimodal & Mobile (Fase 4 - Nova)]                      │
│  Backend Flask/FastAPI ◄──── Recebe Imagem + Sinais Vitais                  │
│      │                                                                      │
│      ▼                                                                      │
│  Score Multimodal = f(Probabilidade_CNN, Escore_NEWS2_IoT)                  │
│      │                                                                      │
│      ▼                                                                      │
│  Aplicativo Mobile React Native ◄── Exibe Diagnóstico, Alerta e Conduta     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. A Fusão Multimodal: Como e Por Que Conectar

### 2.1 O Problema Clínico
Em cardiologia, um exame de imagem raramente é interpretado no vácuo:
- Um paciente com silhueta cardíaca aumentada no Raio-X (**Cardiomegalia**) mas com sinais vitais estáveis (BPM 70, sem taquicardia) pode estar em acompanhamento crônico compensado.
- Por outro lado, um paciente com **Cardiomegalia** no Raio-X que, simultaneamente, apresenta **Taquicardia (> 120 BPM)** e **Imobilidade / Queda brusca no acelerômetro MPU6050** está em **Insuficiência Cardíaca Aguda Descompensada (Edema Agudo de Pulmão ou Choque Cardiogênico)**.

### 2.2 Algoritmo de Pontuação de Risco Multimodal (Opcional & Recomendado)
O backend da Fase 4 pode incorporar uma rota opcional inteligente `POST /api/v1/multimodal` que implementa a fusão tardia (*late fusion*):

$$\text{Risco Global (0--100)} = \alpha \cdot \text{Score}_{\text{NEWS2}}(\text{IoT}) + \beta \cdot (100 \times P_{\text{CNN}}(\text{Cardiomegalia}))$$

Onde:
- $\alpha = 0.40$ (Peso dos sinais vitais dinâmicos em tempo real da Fase 3).
- $\beta = 0.60$ (Peso da confirmação anatômica por Visão Computacional da Fase 4).

Essa integração dá ao projeto uma coesão sem precedentes, garantindo nota máxima em arquitetura e integração interdisciplinar.

---

## 3. Componentes de Software da Fase 4

### 3.1 Módulo de Visão Computacional (`src/models.py`, `notebooks/`)
- **Pipeline de Treinamento e Inferência:**
  - `CardioCNN_Scratch`: Rede convolucional leve desenvolvida do zero (3 a 4 blocos convolucionais com Dropout e Batch Normalization).
  - `CardioTransfer_ResNet`: Modelo pré-treinado no ImageNet (ResNet18 ou ResNet50) com camadas finais congeladas (*feature extractor*) ou descongeladas para ajuste fino (*fine-tuning*).
- **Métricas:** Geração automatizada de gráficos de treino (Loss/Acurácia por época), Curva ROC com valor de AUC, Curva Precision-Recall e Matriz de Confusão categorizada.

### 3.2 Módulo de Governança e Fairness (`src/fairness.py`)
- Módulo dedicado a auditar a base de testes dividindo os resultados por:
  - Sexo / Gênero (Masculino vs. Feminino).
  - Faixa Etária (Jovens < 45 anos, Adultos 45-65, Idosos > 65 anos).
- Cálculo das métricas:
  - **Disparate Impact Ratio (DIR)**: Taxa de predições positivas do grupo desfavorecido dividido pelo grupo favorecido.
  - **Equal Opportunity Difference (EOD)**: Diferença na taxa de verdadeiros positivos (Recall) entre os grupos.

### 3.3 Backend REST de Inferência (`backend/app.py`)
- Implementado em **Flask** (conforme sugerido no edital) ou **FastAPI** (reutilizando a maturidade da Fase 3).
- Carrega os pesos otimizados da CNN na inicialização (`warm-up`).
- Exponibiliza endpoints JSON com validação de entrada, tratamento de imagem (PIL / OpenCV) e conversão em tensor.

### 3.4 Aplicativo Mobile (`mobile/`)
- Construído em **React Native com Expo**.
- Interface nos padrões oficiais de design da FIAP (`#ED145B`, `#FF2E73`, Dark Theme).
- Integração de câmera nativa e seletor de arquivos de imagem.
- Tratamento de estados: *Carregando / Analisando Imagem*, *Sucesso com Diagnóstico*, *Erro / Conexão Offline*.
- Modo Mock embutido para testes autônomos sem necessidade de servidor ativo.
