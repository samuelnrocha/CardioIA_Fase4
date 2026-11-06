# 03 — Parte 1: Dataset e Pipeline de Pré-processamento
## Organização de Dados Médicos e Preparação para Redes Convolucionais

---

## 1. Escolha e Estrutura do Dataset

O edital da FIAP sugere o uso de datasets públicos consolidados de imagens médicas, destacando:
- **NIH Chest X-ray Dataset** (<https://www.kaggle.com/datasets/nih-chest-xrays/data>): Contém mais de 100 mil radiografias frontais de tórax de mais de 30 mil pacientes com anotações de 14 condições torácicas.
- **Foco Clínico Selecionado para o CardioIA:**
  Para manter o foco estrito em **Cardiologia e Doenças Cardiovasculares**, o pipeline deve focar primordialmente na detecção de:
  1. **Cardiomegalia (Aumento da silhueta cardíaca)**: Condição clínica direta em que o Índice Cardiotorácico (ICT) ultrapassa 0,50, indicativo clássico de insuficiência cardíaca congestiva ou miocardiopatia dilatada.
  2. **Controle / Normalidade**: Radiografias sem achados patológicos cardíacos.
  3. *(Extensão opcional)*: **Edema / Infiltrado Pulmonar**: Complicações cardiorrespiratórias imediatas de falência ventricular esquerda.

---

## 2. O Pipeline de Pré-processamento Passo a Passo

O pré-processamento de imagens médicas possui particularidades cruciais que devem ser documentadas para garantir nota máxima:

```
┌────────────────────────────────────────────────────────────────────────┐
│               FLUXO COMPLETO DE PRÉ-PROCESSAMENTO                      │
│                                                                        │
│   Imagens Originais (Dicom / PNG em alta resolução 1024x1024)          │
│                       │                                                │
│                       ▼                                                │
│   1. Filtragem e Curadoria de Metadados (Patient ID, Rótulos, Sexo)    │
│                       │                                                │
│                       ▼                                                │
│   2. Divisão Estratificada Sem Vazamento de Pacientes (GroupKFold)     │
│      Treino (70%) | Validação (15%) | Teste (15%)                      │
│                       │                                                │
│                       ▼                                                │
│   3. Redimensionamento Espacial (224 x 224 pixels)                     │
│                       │                                                │
│                       ▼                                                │
│   4. Normalização de Intensidade (Z-Score / Estatísticas ImageNet)     │
│                       │                                                │
│                       ▼                                                │
│   5. Aumento de Dados Clínico (Data Augmentation Restrito)             │
│      - Rotações suaves (± 7 graus)                                     │
│      - Zoom sutil (0.95 a 1.05)                                        │
│      - Ajuste leve de contraste (CLAHE / Gamma)                        │
│      - PROIBIDO: Flip vertical (inversão anatômica irreal)             │
│                       │                                                │
│                       ▼                                                │
│   Tensores Prontos para Treinamento (PyTorch DataLoader / TF Dataset)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Decisões Críticas de Engenharia de Dados

### 3.1 Prevenção de *Patient Leakage* (Vazamento de Pacientes)
Em radiografias de tórax, um mesmo paciente frequentemente realiza múltiplos exames ao longo dos meses. 
- **O Erro Comum:** Realizar um `train_test_split` aleatório simples. Isso faz com que imagens do mesmo indivíduo caiam no treino e no teste, inflando artificialmente as métricas (o modelo decora as características ósseas ou do marca-passo daquele paciente específico em vez de aprender a patologia).
- **A Solução Mandatória:** Agrupar pela coluna `Patient ID` (ou equivalente) garantindo que nenhum paciente presente no conjunto de treino apareça no teste ou validação (`GroupShuffleSplit`).

### 3.2 Redimensionamento e Canais
- As redes clássicas de Transfer Learning (ResNet, VGG) foram pré-treinadas com imagens $224 \times 224 \times 3$.
- Como radiografias são em escala de cinza (1 canal), a imagem deve ser replicada nos 3 canais RGB ($C \times H \times W = 3 \times 224 \times 224$) para aproveitar os pesos originais do ImageNet.

### 3.3 Normalização
Aplicar normalização padronizada com a média e desvio padrão do ImageNet:
- $\mu = [0.485, 0.456, 0.406]$
- $\sigma = [0.229, 0.224, 0.225]$

---

## 4. Estrutura do Notebook `01_preprocessamento_e_organizacao_dados.ipynb`

O notebook da Parte 1 deve conter:
1. **Introdução Teórica:** Conceito de índice cardiotorácico e relevância do diagnóstico assistido por IA.
2. **Setup e Carga dos Metadados:** Exibição da distribuição de classes (gráfico de barras da proporção de Cardiomegalia vs. Normal).
3. **Inspeção Visual:** Grid com 8 radiografias de exemplo, com anotação da classe, gênero e idade do paciente.
4. **Execução do Pipeline:** Código modular com funções claras (`carregar_imagem`, `aplicar_transformacoes`, `criar_dataloaders`).
5. **Verificação de Splits:** Teste unitário confirmando que `interseção(pacientes_treino, pacientes_teste) == 0`.
6. **Exportação:** Salvamento de amostras processadas em `data/processed/` e arquivos de índice `.csv`.

---

## 5. Entregável Escrito: Relatório da Parte 1

O edital exige um relatório curto (1 a 2 páginas). O agente deve gerar este documento em:
- Arquivo Markdown: `relatorios/relatorio_parte1_preprocessamento.md`
- Exportação em PDF: `relatorios/relatorio_parte1_preprocessamento.pdf` (gerado via script headless ou ferramenta PDF).
- **Conteúdo Obrigatório:**
  - Descrição do dataset e justificativa de escolha.
  - Tabela comparativa dos conjuntos de Treino, Validação e Teste.
  - Explicação matemática da normalização e técnicas de aumento de dados.
  - Análise dos cuidados com vazamento de dados de pacientes.
