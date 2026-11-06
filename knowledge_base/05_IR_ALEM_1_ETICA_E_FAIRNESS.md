# 05 — Ir Além 1: Ética, Governança e Fairness
## Auditoria de Vieses Algorítmicos em Visão Computacional Médica

---

## 1. O Desafio Ético em Inteligência Artificial para a Saúde

A aplicação de Redes Neurais Convolucionais em radiografias de tórax envolve riscos críticos de saúde pública e justiça social. Estudos seminais (como *Seyyed-Kalantari et al., Nature Medicine 2021*) comprovaram que modelos de visão treinados no NIH Chest X-ray frequentemente apresentam **subdiagnóstico sistemático (*underdiagnosis*) em populações vulneráveis**, particularmente mulheres jovens e minorias étnicas, gerando taxas inaceitáveis de falsos negativos.

O objetivo do **Ir Além 1** é elevar o CardioIA ao nível dos mais altos padrões de Governança Médica e Bioética preconizados pela **OMS (Organização Mundial da Saúde)**, **FDA** e **LGPD**.

---

## 2. Identificação de Vieses no Dataset NIH Chest X-ray

O agente implementador deve analisar as três principais fontes de viés presentes na base:

### 2.1 Viés Demográfico por Gênero (*Sex/Gender Imbalance*)
- A proporção de pacientes do sexo masculino versus feminino no dataset.
- Diferenças fisiológicas e anatômicas da silhueta cardíaca (em média, a razão cardiotorácica varia entre homens e mulheres e a presença de tecido mamário pode afetar a atenuação dos raios X).

### 2.2 Viés Etário (*Age-Group Disparities*)
- A cardiomegalia é expressivamente mais prevalente em pacientes idosos (> 65 anos).
- Se o modelo for treinado sem controle etário, ele pode aprender a classificar a idade do paciente através da densidade óssea torácica como atalho (*spurious shortcut*) para predizer cardiomegalia, falhando gravemente em jovens com cardiopatia congênita.

### 2.3 Viés de Aquisição Técnica (*AP vs. PA Viewpoint*)
- **Incidência PA (Posteroanterior):** Paciente em pé encostado no detector. Projeção padrão onde o tamanho do coração não sofre distorção geométrica.
- **Incidência AP (Anteroposterior):** Realizada frequentemente em leito hospitalar em pacientes acamados e mais graves. Pela geometria de dispersão dos raios X, o coração aparenta estar artificialmente maior (falso aumento da silhueta).
- Um modelo ingênuo associa a etiqueta "AP" (ou artefatos de leito/eletrodos) à cardiomegalia.

---

## 3. Métricas Matemáticas de Justiça Algorítmica (Fairness)

O agente deve implementar as seguintes métricas formais no notebook `notebooks/03_etica_e_fairness_visao_computacional.ipynb`:

### 3.1 Paridade Demográfica (*Demographic Parity*)
Exige que a probabilidade de uma predição positiva de cardiomegalia seja idêntica entre os grupos protegidos ($A = 0$ e $A = 1$):

$$|P(\hat{Y} = 1 \mid A = 0) - P(\hat{Y} = 1 \mid A = 1)| \le \epsilon$$

- **Disparate Impact Ratio (DIR):**

$$\text{DIR} = \frac{P(\hat{Y} = 1 \mid A = \text{desfavorecido})}{P(\hat{Y} = 1 \mid A = \text{favorecido})}$$

*Regra de Ouro da Legislação (EEOC):* Um modelo é considerado equitativo se $\text{DIR} \ge 0.80$ (regra dos quatro quintos).

### 3.2 Igualdade de Oportunidades (*Equal Opportunity*)
No contexto médico, esta é a métrica **mais crítica**: a taxa de verdadeiros positivos (Sensibilidade / Recall) deve ser igual entre todos os grupos protegidos:

$$P(\hat{Y} = 1 \mid Y = 1, A = 0) \approx P(\hat{Y} = 1 \mid Y = 1, A = 1)$$

*Impacto Clínico:* Garante que a probabilidade de um paciente doente ter sua cardiomegalia detectada não dependa do seu gênero ou idade.

### 3.3 Paridade Preditiva (*Predictive Parity*)
A precisão (valor preditivo positivo) deve ser estável entre os grupos:

$$P(Y = 1 \mid \hat{Y} = 1, A = 0) \approx P(Y = 1 \mid \hat{Y} = 1, A = 1)$$

---

## 4. Plano de Mitigação Proposto

O agente deve formular estratégias práticas em 3 níveis:

1. **Pré-processamento (Dados):**
   - *Re-sampling balanceado*: Sobreamostragem de grupos sub-representados com a patologia.
   - *Data Augmentation adaptativo*: Aumento direcionado aos subgrupos com menor taxa de acerto.
2. **Em Treinamento (Algoritmo):**
   - *Função de perda balanceada por grupo (Fairness-penalized Loss)*: Penalização por divergência de sensibilidade entre gêneros.
   - *Remoção de atalhos*: Mascaramento de etiquetas e artefatos de texto gravados na radiografia.
3. **Pós-processamento (Decisão Clínica):**
   - *Ajuste diferenciado de limiar de corte ($\tau$ threshold)*: Calibrar limiares de decisão específicos para cada subpopulação para atingir sensibilidade idêntica (Equalized Odds Thresholding).

---

## 5. Entregável Escrito: Relatório Técnico de Ética e Governança

O agente deve gerar o documento:
- `relatorios/relatorio_ir_alem1_governanca_etica.md` / `.pdf` (máximo de 2 páginas).
- **Estrutura do Relatório:**
  1. *Introdução e Princípios Bioéticos:* Autonomia, Não-Maleficência, Beneficência e Justiça (Beauchamp & Childress).
  2. *Auditoria Quantitativa de Vieses no Dataset NIH:* Tabelas com métricas estratificadas por gênero e idade.
  3. *Resultados de Fairness do CardioIA:* Comparação antes e depois da mitigação.
  4. *Conformidade Regulatória:* Alinhamento com a Resolução CFM sobre telemedicina, LGPD (Art. 20 sobre revisão de decisões automatizadas) e recomendações da OMS para IA na saúde.
