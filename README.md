# CardioIA Fase 4 — Assistente Cardiológico Virtual com Visão Computacional
## Base de Conhecimento e Runbook Mestre para Agentes Autônomos

> **Status:** Ativo e Orientativo  
> **Instituição:** FIAP — Pós-Tech / Graduação em Inteligência Artificial e Saúde Digital  
> **Equipe:** Data Health Solutions  
> **Público-Alvo:** Agentes de IA Autônomos e Engenheiros de Software / Cientistas de Dados  

---

## 🎯 Propósito deste Diretório

Este diretório contém a **Base de Conhecimento Completa** e o **Runbook Executivo** necessários para guiar qualquer agente autônomo na concepção, treinamento, empacotamento e entrega da **Fase 4 do projeto CardioIA**.

O princípio orientador deste material é conceder **total liberdade criativa e de engenharia ao agente implementador**, fornecendo os limites de negócio, os requisitos acadêmicos da FIAP, os critérios de avaliação (10 + 1 ponto extra), as armadilhas técnicas e o protocolo de uso da ferramenta **`/colab-mcp`** para treinamento em nuvem no Google Colab.

---

## 🗺️ Mapa de Navegação da Base de Conhecimento

| Documento | Foco Principal | Público / Quando Consultar |
| :--- | :--- | :--- |
| **[`RUNBOOK.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/RUNBOOK.md)** | **Guia Mestre de Execução:** Ordem de fases, regras de ouro, checklist e decisões | **Ponto de entrada obrigatório** para o agente implementador |
| **[`00_EXECUTIVE_SUMMARY_E_MAPA_MENTAL.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/00_EXECUTIVE_SUMMARY_E_MAPA_MENTAL.md)** | Visão do produto, contextualização com Fases 1 a 3, identidade do grupo | Entendimento holístico do problema clínico |
| **[`01_ARQUITETURA_MULTIMODAL.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/01_ARQUITETURA_MULTIMODAL.md)** | Conexão entre IoT (Fase 3: BPM/Temperatura/LIF SNN) e Visão (Fase 4: CNN) | Decisões de arquitetura de dados e backend |
| **[`02_COLAB_MCP_E_NOTEBOOK_STANDARDS.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/02_COLAB_MCP_E_NOTEBOOK_STANDARDS.md)** | **Protocolo Colab-MCP:** Como invocar `open_colab_browser_connection` e criar notebooks 100% autônomos no Google Colab | Antes e durante o treinamento dos modelos de IA |
| **[`03_PARTE1_DATASET_E_PREPROCESSAMENTO.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/03_PARTE1_DATASET_E_PREPROCESSAMENTO.md)** | Datasets médicos (NIH Chest X-ray, ECGs), pipelines de resize, normalização e split | Implementação da **Parte 1** |
| **[`04_PARTE2_CNN_SCRATCH_VS_TRANSFER.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/04_PARTE2_CNN_SCRATCH_VS_TRANSFER.md)** | CNN Customizada vs Transfer Learning (ResNet/VGG16), métricas, export de modelo | Implementação da **Parte 2** |
| **[`05_IR_ALEM_1_ETICA_E_FAIRNESS.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/05_IR_ALEM_1_ETICA_E_FAIRNESS.md)** | Análise de vieses demográficos, métricas de justiça algorítmica e relatório | Implementação do **Ir Além 1** |
| **[`06_IR_ALEM_2_BACKEND_FLASK_API.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/06_IR_ALEM_2_BACKEND_FLASK_API.md)** | API REST de inferência em Python (Flask/FastAPI), upload multipart, schemas JSON | Implementação do backend do **Ir Além 2** |
| **[`07_IR_ALEM_2_APP_MOBILE_REACT_NATIVE.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/07_IR_ALEM_2_APP_MOBILE_REACT_NATIVE.md)** | App mobile em React Native, telas de upload, exibição de diagnóstico, identidade FIAP | Implementação do frontend do **Ir Além 2** |
| **[`08_SCRIPTS_E_SNIPPETS_DE_REFERENCIA.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/08_SCRIPTS_E_SNIPPETS_DE_REFERENCIA.md)** | Trechos de código prontos para reúso (PyTorch/TensorFlow, Flask, DataLoaders) | Aceleração de codificação |
| **[`09_CHECKLIST_E_CRITERIOS_NOTA_MAXIMA.md`](file:///home/samuelrocha/Documents/fiap/CardioIA_Fase4/knowledge_base/09_CHECKLIST_E_CRITERIOS_NOTA_MAXIMA.md)** | Matriz detalhada de pontuação FIAP (10 pontos normais + 1 ponto extra de grupo) | Validação final pré-entrega |

---

## ⚡ Princípios Fundamentais para o Agente Implementador

1. **Liberdade Técnica com Foco no Objetivo:** O agente tem plena autonomia para escolher bibliotecas (ex.: PyTorch ou TensorFlow/Keras, React Native puro ou Expo, Flask ou FastAPI), contanto que respeite os entregáveis explícitos do edital FIAP.
2. **Compatibilidade Incondicional com Google Colab:** Qualquer notebook `.ipynb` gerado DEVE rodar em um clique no Google Colab, contendo mecanismos automáticos de obtenção de dataset e fallback sintético caso conexões externas falhem.
3. **Uso Mandatório do `colab-mcp`:** Para treinamentos pesados e validação remota, o agente deve orquestrar a conexão através da ferramenta MCP `open_colab_browser_connection`.
4. **Continuidade de Marca e Excelência:** Preservar a identidade do grupo **Data Health Solutions**, paleta oficial da FIAP (`#ED145B`, `#FF2E73`, `#030305`, `#10B981`, `#38BDF8`) e o padrão de engenharia já estabelecido na Fase 3.
