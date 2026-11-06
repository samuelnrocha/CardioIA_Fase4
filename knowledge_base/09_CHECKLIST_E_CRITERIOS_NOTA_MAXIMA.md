# 09 — Critérios de Avaliação e Checklist Nota Máxima (10 + 1 Extra)
## Matriz de Conformidade Rigorosa com o Edital FIAP

---

## 1. Tabela de Rubrica Oficial do Edital

| Critério Avaliado | Pontuação Oficial | Requisito de Aceite / Evidência Verificável | Onde Está no Projeto |
| :--- | :---: | :--- | :--- |
| **Pipeline de Pré-processamento** | **3,0 pts** | Dataset selecionado (NIH / ECG), resize, normalização, ausência de leakage por paciente, splits Treino/Val/Teste e documentação clara. | `notebooks/01_preprocessamento...ipynb`<br>`relatorios/relatorio_parte1...pdf` |
| **CNN Construída do Zero** | **2,0 pts** | Rede convolucional customizada (Conv2D, BatchNorm, Pooling, Dropout, Dense) treinada com convergência comprovada por curvas de perda/acurácia. | `notebooks/02_classificacao...ipynb`<br>`src/models.py` |
| **Transfer Learning Funcional** | **2,0 pts** | Rede pré-treinada (ResNet18/50 ou VGG16) adaptada com nova cabeça de classificação, comparada quantitativamente contra a CNN do zero. | `notebooks/02_classificacao...ipynb`<br>`backend/models/best_cardio_resnet.pt` |
| **Protótipo de Apresentação** | **2,0 pts** | Interface acessível funcional (Interface web Flask ou notebook interativo) permitindo upload e classificação com barra de confiança. | `backend/app.py`<br>`backend/templates/index.html` |
| **Documentação Clara** | **1,0 pt** | Código modular com docstrings e tipagem, README.md impecável, relatórios técnicos bem estruturados e referências bibliográficas. | `README.md`<br>`relatorios/` |
| **Trabalho em Equipe e Governança** | **+1,0 pt (Extra)** | Documentação do grupo multidisciplinar (Data Health Solutions), papéis distribuídos, governança e conformidade com diretrizes Alura/FIAP. | `README.md` (Seção Equipe e Papéis) |
| **Total Base + Bônus** | **11,0 pts** | **Nota 10 Garantida com margem de segurança contra qualquer penalização.** | — |

---

## 2. Checklist dos Desafios "Ir Além"

### 🎯 Ir Além 1 — Ética e Governança em Visão Computacional
- [ ] Análise quantitativa de desbalanceamento por sexo/gênero e faixas etárias no dataset.
- [ ] Cálculo matemático formal de métricas de fairness (*Disparate Impact Ratio*, *Demographic Parity*, *Equal Opportunity Difference*).
- [ ] Relatório técnico de até 2 páginas (`relatorios/relatorio_ir_alem1_governanca_etica.pdf`) abordando bioética, LGPD e diretrizes da OMS para IA na saúde.
- [ ] *(Opcional enriquecedor)*: Notebook executável `notebooks/03_etica_e_fairness_visao_computacional.ipynb`.

### 📱 Ir Além 2 — Aplicativo Mobile em React Native
- [ ] Projeto mobile completo e funcional em `mobile/` (React Native com Expo).
- [ ] Tela de upload de imagem médica com captura por câmera e seleção da galeria.
- [ ] Comunicação HTTP com o backend Flask (`POST /api/v1/predict`).
- [ ] Exibição visual rica: categoria detectada, percentual de confiança, alerta de risco e recomendações clínicas.
- [ ] Modo Mock offline embutido para permitir avaliação mesmo se o corretor não iniciar o backend.
- [ ] Roteiro estruturado para gravação do vídeo demonstrativo de até 3 minutos (`mobile/ROTEIRO_VIDEO_MOBILE.md`).

---

## 3. Checklist Específico da Regra Colab-MCP

- [ ] Todos os notebooks `.ipynb` abrem e executam de ponta a ponta no Google Colab em 1 clique.
- [ ] Badge oficial `Open In Colab` presente no topo de todos os notebooks.
- [ ] Célula 1 de bootstrap autônomo presente (clona repositório / instala dependências se `'google.colab' in sys.modules`).
- [ ] Resiliência de dataset: se o download de dados externos falhar no runtime do Colab, o script ativa o gerador sintético de alta fidelidade e não quebra a execução.
- [ ] O agente usou ou documentou o comando `/colab-mcp` (`open_colab_browser_connection`) para orquestrar sessões de GPU na nuvem.

---

## 4. Auditoria Final Pré-Finalização

Antes de emitir o comando de conclusão, o agente implementador deve validar:
1. `git status`: Repositório limpo, sem arquivos temporários indesejados (`.pyc`, `__pycache__`, arquivos `.tmp`).
2. Todos os links internos e referências de arquivos utilizam caminhos relativos ou sintaxe padrão `file://`.
3. Os notebooks foram salvos com saídas de gráficos geradas e limpas (ou prontas para visualização imediata no GitHub).
4. O `README.md` principal na raiz do repositório reflete fielmente todas as entregas com instruções claras de "Como Rodar em 5 Minutos".
