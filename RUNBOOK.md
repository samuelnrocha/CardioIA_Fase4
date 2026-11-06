# RUNBOOK MESTRE — CONSTRUÇÃO DO CARDIOIA FASE 4
## Guia Operacional Executivo para Agentes Autônomos de IA

> **Documento:** `RUNBOOK.md`  
> **Nível de Autoridade:** Diretriz Mestra de Engenharia e Arquitetura  
> **Escopo:** Fase 4 — Assistente Cardiológico Virtual com Visão Computacional (FIAP)  
> **Repositório Alvo:** `samuelnrocha/CardioIA_Fase4`  
> **Equipe:** Data Health Solutions  

---

## 1. VISÃO GERAL E PAPEL DO AGENTE IMPLEMENTADOR

Você, agente implementador, foi convocado para construir do zero a **Fase 4 do ecossistema CardioIA**. 
Nas fases anteriores, a equipe construiu:
- **Fase 2:** Análise de dados clínicos e estruturação hospitalar.
- **Fase 3:** Dispositivo IoT vestível (ESP32 no Wokwi) com telemetria MQTT em tempo real, buffer offline SPIFFS (13,8h de resiliência), API REST em FastAPI com RPA de e-mail e Inteligência Artificial Neuromórfica (LIF e FitzHugh-Nagumo) em séries temporais de ECG com redução de 87% de energia na borda.

Na **Fase 4**, a CardioIA atinge a maturidade da **Visão Computacional e Multimodalidade Médica**. O objetivo é diagnosticar patologias cardiotorácicas (como **Cardiomegalia**, **Pneumonia/Congestão Pulmonar** associada a insuficiência cardíaca ou **Anomalias em Imagens de ECG**) a partir de exames de imagem médica, conectando esse modelo a uma interface de fácil uso e a um aplicativo mobile.

### 🌟 Seu Grau de Liberdade
- **Você tem TOTAL LIBERDADE** para decidir a arquitetura de código, escolha entre frameworks de deep learning (`PyTorch` ou `TensorFlow/Keras`), organização de componentes no React Native (`Expo` recomendado pela flexibilidade ou CLI bare), hiperparâmetros, número de camadas convolucionais e estratégias de data augmentation.
- **Seu compromisso inegociável** é com a **qualidade científica**, a **satisfação integral dos critérios do edital FIAP** (10 pontos normais + 1 ponto extra de grupo), a **compatibilidade 100% autônoma com Google Colab** e o uso do **`colab-mcp`** para as sessões de treinamento e experimentação.

---

## 2. A REGRA DE OURO DO GOOGLE COLAB E COLAB-MCP

### 2.1 Uso Mandatório do `colab-mcp`
O usuário estabeleceu uma diretriz explícita: **o agente deve utilizar o servidor `/colab-mcp` para as necessidades de treinamento e experimentação interativa em nuvem**.
- O MCP Server registrado é `colab-mcp`.
- A ferramenta principal é `open_colab_browser_connection` (lazy tool).
- **Como invocar:**
  ```json
  ToolName: call_mcp_tool
  Arguments: {
    "ServerName": "colab-mcp",
    "ToolName": "open_colab_browser_connection",
    "Arguments": {}
  }
  ```
- **Finalidade:** Abrir uma sessão ativa do Google Colab no navegador para desbloquear ferramentas de execução e monitoramento de notebooks remotos providos de GPU/TPU pelo Google Colab.
- Consulte o guia dedicado: [`02_COLAB_MCP_E_NOTEBOOK_STANDARDS.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/02_COLAB_MCP_E_NOTEBOOK_STANDARDS.md).

### 2.2 Requisito de Execução Autônoma no Colab
Todo e qualquer notebook `.ipynb` criado nesta fase DEVE ser executável no Google Colab sem que o usuário ou avaliador precise subir pastas manualmente:
1. Deve possuir a **célula de inicialização inteligente** que verifica se está rodando no Google Colab (`'google.colab' in sys.modules`).
2. Se estiver no Colab, deve clonar ou baixar os assets/scripts automaticamente do GitHub público `https://github.com/samuelnrocha/CardioIA_Fase4`.
3. Deve incluir um **gerador/downloader automático de amostras reais do dataset** (ex.: NIH Chest X-ray ou subset curado de cardiomegalia) e um mecanismo de geração sintética fisiológica caso o ambiente esteja sem internet ou atinja rate limit.
4. O topo do notebook deve exibir o badge oficial:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samuelnrocha/CardioIA_Fase4/blob/main/notebooks/...)
   ```

---

## 3. CHECKLIST EXECUTIVO POR FASES DE CONSTRUÇÃO

```
┌────────────────────────────────────────────────────────────────────────┐
│                      FLUXOGRAMA DE EXECUÇÃO                            │
│                                                                        │
│  [FASE 0: Setup & Estrutura de Diretórios]                             │
│       │                                                                │
│       ▼                                                                │
│  [FASE 1: Parte 1 — Pipeline de Dados e Pré-processamento]             │
│       │                                                                │
│       ▼                                                                │
│  [FASE 2: Parte 2 — Treinamento CNN Scratch vs Transfer Learning]      │
│       │   (Orquestrado via colab-mcp no Google Colab)                  │
│       ▼                                                                │
│  [FASE 3: Ir Além 1 — Métricas de Fairness & Governança Ética]         │
│       │                                                                │
│       ▼                                                                │
│  [FASE 4: Ir Além 2 — Backend Flask/FastAPI REST de Inferência]        │
│       │                                                                │
│       ▼                                                                │
│  [FASE 5: Ir Além 2 — Aplicativo Mobile em React Native]               │
│       │                                                                │
│       ▼                                                                │
│  [FASE 6: Relatórios Técnicos (PDF/Markdown) & Documentação Master]    │
│       │                                                                │
│       ▼                                                                │
│  [FASE 7: Verificação Fim-a-Fim, Testes de Sanidade & Push GitHub]     │
└────────────────────────────────────────────────────────────────────────┘
```

---

### FASE 0: SETUP DO REPOSITÓRIO E ESTRUTURAÇÃO

1. **Organização da árvore de arquivos:**
   O agente deve estruturar o repositório `/home/samuelrocha/Documents/fiap/CardioIA_Fase4` de forma modular e limpa:
   ```text
   CardioIA_Fase4/
   ├── README.md
   ├── RUNBOOK.md
   ├── requirements.txt
   ├── knowledge_base/               <-- Base de conhecimento
   ├── data/                         <-- Amostras locais e scripts de download
   │   ├── raw/
   │   └── processed/
   ├── notebooks/                    <-- Notebooks autônomos para o Google Colab
   │   ├── 01_preprocessamento_e_organizacao_dados.ipynb
   │   ├── 02_classificacao_cnn_e_transfer_learning.ipynb
   │   └── 03_etica_e_fairness_visao_computacional.ipynb
   ├── src/                          <-- Código modular reutilizável
   │   ├── __init__.py
   │   ├── dataset.py
   │   ├── models.py
   │   ├── train.py
   │   ├── evaluate.py
   │   └── fairness.py
   ├── backend/                      <-- API REST Flask/FastAPI
   │   ├── app.py
   │   ├── config.py
   │   ├── routes.py
   │   ├── models/                   <-- Pesos exportados (.pt ou .h5/onnx)
   │   └── tests/
   ├── mobile/                       <-- Aplicativo React Native (Expo)
   │   ├── App.js / App.tsx
   │   ├── app.json
   │   ├── package.json
   │   ├── src/
   │   │   ├── screens/
   │   │   ├── components/
   │   │   ├── services/api.js
   │   │   └── theme/colors.js       <-- Cores oficiais FIAP (#ED145B)
   │   └── assets/
   └── relatorios/                   <-- Relatórios exigidos pelo edital
       ├── relatorio_parte1_preprocessamento.pdf/.md
       ├── relatorio_parte2_modelos_cnn.pdf/.md
       └── relatorio_ir_alem1_governanca_etica.pdf/.md
   ```

---

### FASE 1: PARTE 1 — PRÉ-PROCESSAMENTO E ORGANIZAÇÃO (3 PONTOS)

- **Objetivo do Edital:** Selecionar dataset público de imagens médicas (ex.: **NIH Chest X-rays** com foco em Cardiomegalia/Normal ou dataset de ECG em imagem), aplicar redimensionamento, normalização, divisão treino/validação/teste e documentar o pipeline.
- **Diretrizes para o Agente:**
  1. Consulte o documento [`03_PARTE1_DATASET_E_PREPROCESSAMENTO.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/03_PARTE1_DATASET_E_PREPROCESSAMENTO.md).
  2. Implemente o notebook [`notebooks/01_preprocessamento_e_organizacao_dados.ipynb`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/notebooks/01_preprocessamento_e_organizacao_dados.ipynb) com células ricas em markdown explicativo.
  3. Crie rotina determinística de particionamento estratificado (ex.: 70% treino, 15% validação, 15% teste) evitando vazamento de dados de pacientes entre splits (*patient leakage*).
  4. Implemente Data Augmentation focado em radiologia (pequenas rotações $\pm 7^\circ$, zoom suave, corte adaptativo — sem flips verticais irreais).
  5. Gere o relatório de 1 a 2 páginas justificando as escolhas de normalização (ex.: ImageNet stats vs normalização z-score local).

---

### FASE 2: PARTE 2 — CLASSIFICAÇÃO COM CNN E TRANSFER LEARNING (4 PONTOS)

- **Objetivo do Edital:** Testar duas abordagens:
  1. CNN simples construída do zero.
  2. Transfer Learning com modelo pré-treinado consagrado (ResNet18/50, VGG16 ou EfficientNet).
  Avaliar com Acurácia, Matriz de Confusão, Precisão, Recall, F1-Score e Curva ROC. Apresentar em protótipo interativo.
- **Diretrizes para o Agente:**
  1. Consulte o documento [`04_PARTE2_CNN_SCRATCH_VS_TRANSFER.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/04_PARTE2_CNN_SCRATCH_VS_TRANSFER.md).
  2. **Orquestração via `colab-mcp`:** Conecte ao Colab e execute os treinos em GPU para gerar pesos otimizados. Salve o melhor checkpoint (ex.: `best_cardio_resnet.pt` ou `.onnx`).
  3. No notebook [`notebooks/02_classificacao_cnn_e_transfer_learning.ipynb`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/notebooks/02_classificacao_cnn_e_transfer_learning.ipynb), gere os gráficos científicos comparativos no padrão escuro FIAP (espelhando a qualidade da Fase 3).
  4. Salve todas as figuras em alta resolução para que os relatórios e o protótipo possam usá-las.
  5. O modelo treinado deve ser exportado de forma leve (< 100MB) para que possa rodar inferência rápida no backend Flask e localmente.

---

### FASE 3: IR ALÉM 1 — ÉTICA, FAIRNESS E GOVERNANÇA (ITEM DIFERENCIAL)

- **Objetivo do Edital:** Identificar limitações e vieses no dataset (desbalanceamento por idade, gênero, representatividade demográfica), aplicar métricas de fairness e propor plano de mitigação em relatório técnico de até 2 páginas.
- **Diretrizes para o Agente:**
  1. Consulte o documento [`05_IR_ALEM_1_ETICA_E_FAIRNESS.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/05_IR_ALEM_1_ETICA_E_FAIRNESS.md).
  2. Calcule métricas reconhecidas da literatura:
     - **Disparate Impact Ratio (DIR):** Regra dos 80% (EEOC).
     - **Demographic Parity Difference:** Igualdade de taxas de seleção positiva entre subgrupos (ex.: Homens vs Mulheres ou faixas etárias).
     - **Equalized Odds / Equal Opportunity:** Paridade de Recall/Sensibilidade (crucial para que nenhum grupo sofra mais com falsos negativos de cardiomegalia).
  3. Crie o notebook experimental [`notebooks/03_etica_e_fairness_visao_computacional.ipynb`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/notebooks/03_etica_e_fairness_visao_computacional.ipynb) com gráficos comparativos de métricas por subpopulação.
  4. Redija o relatório técnico de Governança e Ética Médica detalhando diretrizes da OMS, FDA e LGPD sobre IA na saúde.

---

### FASE 4: IR ALÉM 2 — BACKEND FLASK/FASTAPI PARA INFERÊNCIA

- **Objetivo do Edital:** Disponibilizar os resultados da CNN através de uma interface web / backend funcional em Flask ou Node.js que alimentará o aplicativo mobile.
- **Diretrizes para o Agente:**
  1. Consulte o documento [`06_IR_ALEM_2_BACKEND_FLASK_API.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/06_IR_ALEM_2_BACKEND_FLASK_API.md).
  2. Crie uma API RESTful leve em Flask ou FastAPI dentro de `backend/app.py`.
  3. Endpoints obrigatórios:
     - `GET /api/v1/health`: Status do serviço e modelo carregado.
     - `POST /api/v1/predict`: Recebe imagem via `multipart/form-data`, executa pré-processamento, roda inferência e devolve:
       ```json
       {
         "classe": "Cardiomegalia",
         "confianca_pct": 94.8,
         "probabilidade": 0.948,
         "limiar_clinico": 0.50,
         "tempo_inferencia_ms": 38.2,
         "status_alerta": "CRITICO",
         "recomendacao_clinica": "Encaminhamento prioritário para ecocardiograma e avaliação de fração de ejeção."
       }
       ```
     - *(Bônus Multimodal)*: `POST /api/v1/multimodal`: Permite enviar tanto a imagem do Raio-X/ECG quanto os sinais vitais IoT da Fase 3 (BPM, Temperatura, Queda) para uma pontuação combinada de risco!
  4. Inclua uma interface web estática simples (`backend/templates/index.html`) para permitir testes imediatos via browser caso o avaliador queira testar sem o celular.

---

### FASE 5: IR ALÉM 2 — APLICATIVO MOBILE EM REACT NATIVE (VÍDEO DE ATÉ 3 MIN)

- **Objetivo do Edital:** Levar os resultados da classificação para um protótipo mobile em React Native, com tela de upload de imagem e exibição das categorias detectadas pela CNN.
- **Diretrizes para o Agente:**
  1. Consulte o documento [`07_IR_ALEM_2_APP_MOBILE_REACT_NATIVE.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/07_IR_ALEM_2_APP_MOBILE_REACT_NATIVE.md).
  2. Estruture o app dentro de `mobile/` usando React Native / Expo:
     - Tela 1: **Upload de Exame Médico** (captura via câmera ou seleção da galeria com preview imediato).
     - Tela 2: **Card de Diagnóstico Inteligente** (exibe a probabilidade da CNN, barra de confiança colorida, selo de risco clínico e recomendações médicas no padrão visual escuro da FIAP).
     - Tela 3: **Painel Integrado Multimodal** (exibição combinada do exame de imagem com a telemetria do paciente).
  3. O app deve ser moderno, responsivo e possuir mock de conexão automático para funcionar mesmo sem servidor ativo (modo demonstração *offline-resilient*).
  4. Documente roteiro claro e sucinto para gravação de vídeo demonstrativo de até 3 minutos.

---

### FASE 6: DOCUMENTAÇÃO, RELATÓRIOS E MATRIZ DE NOTA MÁXIMA

1. **Relatórios em PDF e Markdown:**
   Gere os relatórios formais solicitados pelo edital com diagramação profissional, tabelas comparativas e citações bibliográficas.
2. **README Master:**
   O `README.md` raiz do repositório deve ser impecável, com badges, introdução executiva, links diretos de 1 clique para rodar os notebooks no Google Colab, instruções de inicialização do backend Flask e do app mobile, e tabela de integrantes do grupo **Data Health Solutions**.
3. **Verificação de Sanidade:**
   Antes de dar a missão como concluída, consulte [`09_CHECKLIST_E_CRITERIOS_NOTA_MAXIMA.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/09_CHECKLIST_E_CRITERIOS_NOTA_MAXIMA.md) e certifique-se de que cada um dos critérios de avaliação foi preenchido com evidências verificáveis.

---

## 4. DIRETRIZES DE ESTILO E DESIGN SYSTEM FIAP

Para manter coerência estrita com a identidade visual construída na Fase 3:
- **Cor Primária de Destaque:** `#ED145B` (FIAP Pink / Magenta oficial)
- **Acentos Luminosos:** `#FF2E73` (Neon Pink), `#38BDF8` (Sky Blue / Tech), `#10B981` (Emerald Green / Normalidade clínica), `#F59E0B` (Amber / Atenção)
- **Fundo Base:** `#030305` (Preto puro) e `#0F172A` / `#111117` (Dark Slate para cards e containers)
- **Tipografia:** Títulos limpos em caixa mista, combinando sans-serif geométrica com toques itálicos editoriais nos destaques.
