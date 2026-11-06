# 06 — Ir Além 2: Backend e API REST de Inferência
## Arquitetura de Serviços, Endpoints de Visão Computacional e Fusão Multimodal

---

## 1. Visão Geral do Backend

O edital da Fase 4 solicita a integração do modelo de classificação em um protótipo acessível, especificando um backend em **Flask** (ou Node.js) que servirá tanto a interface web básica quanto o aplicativo mobile em React Native.

### Tecnologias Recomendadas
- **Framework Principal:** Flask (com Flask-CORS) ou FastAPI.
- **Motor de Inferência:** PyTorch (`torch.jit` / `state_dict`) ou ONNX Runtime (`onnxruntime` para inferência leve em CPU sem sobrecarregar a memória).
- **Tratamento de Imagem:** Pillow (`PIL`) e `torchvision.transforms`.

---

## 2. Estrutura de Arquivos do Módulo `backend/`

```text
backend/
├── app.py                     # Inicialização da aplicação Flask/FastAPI e rotas
├── config.py                  # Parâmetros (porta, caminhos dos modelos, thresholds)
├── model_service.py           # Carregamento do modelo na memória e pré-processamento
├── requirements.txt           # Dependências leves para o servidor
├── models/                    # Modelos treinados exportados
│   ├── best_cardio_resnet.pt
│   └── cardio_resnet.onnx
├── static/                    # Arquivos estáticos da interface web
│   ├── css/style.css
│   └── js/main.js
├── templates/                 # Interface web simples (HTML5)
│   └── index.html
└── tests/                     # Testes automatizados da API
    └── test_api.py
```

---

## 3. Especificação dos Endpoints REST

### 3.1 `GET /api/v1/health`
Verifica se o backend está ativo e qual modelo de Visão Computacional está carregado na memória.

- **Resposta (200 OK):**
  ```json
  {
    "status": "online",
    "servico": "CardioIA - Assistente Virtual de Visão Computacional",
    "versao": "4.0.0",
    "modelo_ativo": "ResNet18-CardioTransfer",
    "dispositivo_inferencia": "cuda:0",
    "timestamp": "2026-11-06T00:30:00Z"
  }
  ```

---

### 3.2 `POST /api/v1/predict`
Endpoint principal consumido pelo aplicativo mobile e pela interface web. Recebe a imagem médica em `multipart/form-data`.

- **Parâmetros da Requisição:**
  - `file`: Arquivo de imagem (`.png`, `.jpg`, `.jpeg`, `.dicom`).
  - *(Opcional)* `limiar`: Float com o limiar de decisão de cardiomegalia (padrão: `0.50`).
- **Pipeline Interno:**
  1. Validação de extensão e integridade de arquivo.
  2. Abertura via PIL e conversão para RGB.
  3. Redimensionamento para $224 \times 224$ e normalização com médias do ImageNet.
  4. Execução da inferência no tensor sem cálculo de gradientes (`torch.no_grad()`).
  5. Aplicação de Softmax / Sigmoid para obter probabilidades calibradas.
- **Resposta (200 OK):**
  ```json
  {
    "diagnostico": "Cardiomegalia",
    "classe_id": 1,
    "probabilidade": 0.9412,
    "confianca_pct": 94.12,
    "status_clinico": "ALERTA_CRITICO",
    "severidade": "Alta",
    "tempo_processamento_ms": 32.4,
    "recomendacoes": [
      "Notificar equipe médica de plantão.",
      "Recomenda-se ecocardiograma transtorácico de urgência.",
      "Verificar sinais de congestão pulmonar ou dispneia."
    ]
  }
  ```

---

### 3.3 `POST /api/v1/multimodal` (Diferencial da CardioIA)
Permite a fusão da imagem de Raio-X com a telemetria do vestível da Fase 3.

- **Payload JSON / Multipart:**
  - `file`: Imagem do Raio-X.
  - `bpm`: Número inteiro (ex.: `138`).
  - `temperatura`: Float (ex.: `38.4`).
  - `queda_detectada`: Booleano (ex.: `true`).
  - `imobilidade_segundos`: Inteiro (ex.: `180`).
- **Resposta:**
  Combina a probabilidade da CNN com o algoritmo de risco da Fase 3 e emite um laudo completo unificado.

---

## 4. Interface Web Simples (`templates/index.html`)

Para atender integralmente ao requisito de *"protótipo simples em Flask"*, o backend deve renderizar na rota `/` uma interface moderna no padrão escuro da FIAP (`#ED145B`), permitindo:
1. Área de *drag-and-drop* para soltar uma radiografia.
2. Botões de "Exemplos Rápidos" (botão que carrega instantaneamente uma imagem Normal ou com Cardiomegalia para demonstração rápida sem precisar caçar imagens no disco).
3. Exibição animada da probabilidade, barra de progresso colorida e laudo descritivo gerado pela IA.

---

## 5. Como Executar e Testar

O agente implementador deve assegurar que o backend seja executável em uma linha de comando:

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

E os testes automatizados devem ser verificáveis via:
```bash
pytest tests/test_api.py -v
```
