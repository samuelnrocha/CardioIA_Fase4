# 02 — Diretrizes do Colab-MCP e Padrões de Notebooks
## Guia Operacional para Treinamento em Nuvem e Compatibilidade 100% Google Colab

> **Importante:** Este documento estabelece a regra mandatória de treinamento e desenvolvimento de notebooks estipulada pelo usuário.

---

## 1. O Servidor MCP: `colab-mcp`

### 1.1 O Que É e Onde Está
O ambiente possui o servidor MCP **`colab-mcp`** registrado como ferramenta sob demanda (*lazy-loaded*). 
- **Schema do Tool:** `/home/samuelrocha/.gemini/antigravity-cli/mcp/colab-mcp/open_colab_browser_connection.json`
- **Nome da Ferramenta:** `open_colab_browser_connection`

### 1.2 Como Invocar via Tool Call
Sempre que o agente implementador precisar iniciar, orquestrar ou validar um treinamento remoto com aceleração gráfica no Google Colab, ele deve invocar:

```json
{
  "ServerName": "colab-mcp",
  "ToolName": "open_colab_browser_connection",
  "Arguments": {}
}
```

- **Retorno Esperado:** Um booleano (`true`/`false`) informando se a conexão com a sessão de navegador do Google Colab foi estabelecida com sucesso e as ferramentas de edição remota de notebooks foram desbloqueadas.
- **Workflow do Agente com Colab-MCP:**
  1. O agente prepara o código modular em `src/` e estrutura o notebook `.ipynb` em `notebooks/`.
  2. O agente sobe as alterações para o repositório GitHub público (`CardioIA_Fase4`).
  3. O agente invoca `open_colab_browser_connection` para abrir a interface do Colab.
  4. O agente executa e monitora o treinamento das CNNs no ambiente de GPU (T4/V100) do Google Colab.
  5. Os pesos treinados (`best_model.pt` ou `.h5`) são validados e persistidos na pasta `backend/models/`.

---

## 2. Padrão Inegociável para Notebooks Google Colab

Todo notebook `.ipynb` gerado no projeto **DEVE ser 100% autônomo e executável em um clique** por qualquer pessoa (aluno, professor ou corretor da FIAP) sem gerar erros de "módulo não encontrado" ou "caminho de arquivo inexistente".

### 2.1 Célula 0: Badge Oficial do Google Colab
A primeira célula de todo notebook deve ser em Markdown contendo o badge oficial apontando para o link público do repositório:

```markdown
# CardioIA Fase 4 — [Título do Notebook]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samuelnrocha/CardioIA_Fase4/blob/main/notebooks/NOME_DO_NOTEBOOK.ipynb)
```

### 2.2 Célula 1: Inicialização Inteligente e Auto-Download (Bootstrap Universal)
A primeira célula de código Python deve conter o mecanismo de auto-detecção de ambiente e clonagem/download das dependências:

```python
import os
import sys
import subprocess

# 1. Detecção automática de execução dentro do Google Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🚀 Detectado ambiente Google Colab! Inicializando bootstrap automático...")
    
    # Clona o repositório caso ainda não exista no runtime
    if not os.path.exists("CardioIA_Fase4"):
        !git clone https://github.com/samuelnrocha/CardioIA_Fase4.git
        %cd CardioIA_Fase4
    else:
        %cd CardioIA_Fase4
        !git pull origin main
    
    # Instala dependências específicas se necessário
    !pip install -q -r requirements.txt
    
    # Adiciona o diretório raiz e src ao sys.path
    sys.path.insert(0, os.path.abspath("."))
    sys.path.insert(0, os.path.abspath("src"))
else:
    print("💻 Executando em ambiente local. Configurando caminhos...")
    sys.path.insert(0, os.path.abspath("../"))
    sys.path.insert(0, os.path.abspath("../src"))

print("✅ Ambiente configurado e pronto para execução!")
```

### 2.3 Célula 2: Detecção de Aceleração por Hardware (GPU/TPU/CPU)
O notebook deve sempre selecionar o melhor dispositivo disponível sem travar caso o usuário esteja em CPU:

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de execução selecionado: {device}")
if device.type == "cuda":
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memória VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ Nenhuma GPU detectada. Executando em CPU (recomenda-se ativar GPU em: Ambiente de Execução -> Alterar tipo).")
```

*(Se o agente optar por TensorFlow/Keras, utilizar a detecção correspondente com `tf.config.list_physical_devices('GPU')`).*

---

## 3. Gestão e Download Autônomo de Datasets

O maior erro em notebooks avaliados na FIAP é o notebook depender de arquivos zip locais que o professor não possui.
Para evitar isso, o pipeline de dados no notebook deve obedecer à seguinte regra de resiliência:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE RESILIENTE DE DATASET                       │
│                                                                        │
│  Tentar download automático do dataset curado (Kaggle/GitHub Release)   │
│       │                                                                │
│       ├── Sucesso? ──► Descompactar em data/processed e prosseguir      │
│       │                                                                │
│       └── Falha de rede / Timeout / Sem credenciais Kaggle?            │
│               │                                                        │
│               ▼                                                        │
│       Mecanismo de Fallback Autônomo:                                  │
│       Gerar subset sintético fisiológico em alta fidelidade com        │
│       padrões radiológicos reais (Cardiomegalia vs Normal)             │
│       Garantindo que 100% das células do notebook rodem até o fim!     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Estrutura Visual e Didática dos Notebooks

Para manter a nota máxima nos critérios de documentação e clareza:
1. **Estrutura de Tópicos Ricos:** Dividir em seções numeradas (1. Introdução Teórica, 2. Importações, 3. Carregamento dos Dados, 4. Análise Exploratória, 5. Treinamento, 6. Avaliação e Métricas, 7. Conclusão Clínica).
2. **Gráficos no Padrão Dark/FIAP:** Utilizar paleta coerente com Seaborn / Matplotlib (`#ED145B`, `#FF2E73`, `#38BDF8`, `#10B981` sobre fundo escuro ou limpo).
3. **Explicação Textual:** Cada gráfico deve ser seguido por um parágrafo de **análise clínica e crítica**, demonstrando ao tutor o significado do resultado (ex.: por que um falso negativo de cardiomegalia é mais grave do que um falso positivo).
