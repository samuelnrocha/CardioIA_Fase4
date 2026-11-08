# 🎥 Roteiro Oficial de Gravação do Vídeo Demonstrativo (Até 3 Minutos)
## Ir Além 2 — Aplicativo Mobile em React Native & Integração Backend Flask REST

- **Projeto:** CardioIA Fase 4 — Assistente Cardiológico Virtual com Visão Computacional
- **Instituição:** FIAP — Faculdade de Informática e Administração Paulista
- **Grupo:** **Data Health Solutions**
  - Gabriel Oliveira dos Santos — RM567166
  - Arthur Bruttel Nascimento — RM568484
  - Jonatan Viotti Rodrigues da Silva — RM566787
  - Samuel Nicolas Oliveira Rocha — RM568552
- **Duração Máxima Exigida pelo Edital:** Até 3 minutos (`03:00`)

---

## 🛠️ Preparação Rápida do Ambiente Antes de Gravar (2 Comandos)

1. **Terminal 1 — Iniciar o Backend Flask com o modelo ResNet18 Fine-Tuned:**
   ```bash
   cd CardioIA_Fase4
   source .venv/bin/activate
   python backend/app.py
   ```
   *(O servidor ficará ativo em `http://localhost:5000` e exibirá os logs HTTP em tempo real).*

2. **Terminal 2 — Iniciar o Aplicativo Mobile React Native (Expo):**
   ```bash
   cd CardioIA_Fase4/mobile
   npx expo start --web
   ```
   *(Ou abrir no emulador Android/iOS / aplicativo Expo Go no smartphone).*

---

## ⏱️ Cronograma Minuto a Minuto da Apresentação

### 🎬 Bloco 1: Abertura e Contextualização Clínica (`0:00` – `0:30`)
- **Tela em exibição:** Tela Inicial (`HomeScreen`) do App Mobile CardioIA no tema escuro oficial FIAP (`#ED145B`) ao lado do terminal com o Backend Flask ativo.
- **Fala sugerida:**
  > *"Olá, professores e avaliadores da FIAP! Nós somos o grupo Data Health Solutions e apresentamos agora o módulo Ir Além 2 da Fase 4 do CardioIA: o nosso Aplicativo Mobile em React Native integrado via API REST ao motor de Visão Computacional em PyTorch. Na Fase 3, construímos o vestível IoT no ESP32; agora, na Fase 4, damos olhos clínicos à solução, permitindo que médicos e enfermeiros fotografem ou selecionem uma radiografia de tórax no celular e recebam em milissegundos a detecção de Cardiomegalia combinada aos sinais vitais do paciente."*

---

### 🟢 Bloco 2: Demonstração de Paciente Saudável — Silhueta Cardíaca Normal (`0:30` – `1:15`)
- **Ação na tela:**
  1. Clicar em **"Iniciar Análise de Exame Radiológico (CNN)"**.
  2. Demonstrar os botões de captura via **Câmera Nativa (`expo-image-picker`)**, **Galeria** e **Exemplos Clínicos Pré-Carregados**.
  3. Selecionar a radiografia de um paciente saudável (**"Caso Normal — ICT 0,43"**).
  4. Clicar no botão magenta **"Analisar Radiografia (CNN + Fusão Multimodal)"**.
- **Resultado exibido:**
  - Card animado em **Verde Esmeralda (`#10B981`)**: `SILHUETA CARDÍACA NORMAL` (`Status: NORMAL`).
  - Barra de confiança acima de `95%`, Índice Cardiotorácico estimado `ICT = 0,43 ≤ 0,50`, latência de inferência `~31 ms`.
- **Fala sugerida:**
  > *"Vamos testar primeiro o exame de um paciente com silhueta cardiomediastínica preservada. Ao enviar a imagem para o endpoint `/api/v1/multimodal`, nossa rede ResNet18 pré-treinada e ajustada retorna em cerca de 30 milissegundos o diagnóstico de Silhueta Cardíaca Normal, com mais de 95% de confiança e Índice Cardiotorácico estimado em 0,43, mantendo o paciente em acompanhamento ambulatorial."*

---

### 🚨 Bloco 3: Demonstração de Caso Patológico — Cardiomegalia Crítica + Fusão IoT (`1:15` – `2:05`)
- **Ação na tela:**
  1. Selecionar uma radiografia com aumento severo da área cardíaca (**"Cardiomegalia — ICT 0,58"**).
  2. Clicar em **"Analisar Radiografia (CNN + Fusão Multimodal)"**.
- **Resultado exibido:**
  - Comutação visual imediata para o alerta **Magenta/Vermelho FIAP (`#ED145B`)**: `CARDIOMEGALIA DETECTADA` (`Status: CRITICO`).
  - Probabilidade elevada (`94,8%+`), `ICT Estimado = 0,58` e **Escore Multimodal Integrado** combinando o achado radiológico com a taquicardia (`134 BPM`) do vestível da Fase 3.
  - Exibição automática da lista de condutas clínicas (solicitação prioritária de ecocardiograma transtorácico e dosagem de BNP).
- **Fala sugerida:**
  > *"Agora submetemos a radiografia de um paciente com aumento ventricular. O aplicativo destaca imediatamente o alerta crítico de Cardiomegalia Detectada com 94,8% de confiança e ICT estimado de 0,58. Além disso, o card realiza a fusão multimodal tardia com os sinais vitais do vestível ESP32 da Fase 3, identificando risco agudo descompensado e recomendando ecocardiograma transtorácico de urgência."*

---

### 🔌 Bloco 4: Comprovação da Integração Backend Flask & Modo Offline Resiliente (`2:05` – `2:45`)
- **Ação na tela:**
  1. Mostrar a janela do terminal com o servidor Flask (`backend/app.py`) exibindo as requisições `POST /api/v1/predict` e `POST /api/v1/multimodal -> HTTP 200 OK` e a interface web espelhada no navegador.
  2. No aplicativo mobile, alternar a chave **"Modo Demonstração Offline (Mock Mode)"** na tela inicial para demonstrar que o app possui resiliência embarcada mesmo quando não há rede hospitalar disponível.
- **Fala sugerida:**
  > *"No terminal vemos o servidor Flask processando os tensores 224x224 em tempo real e respondendo com JSON estruturado. E, seguindo o mesmo princípio de resiliência offline que implementamos no SPIFFS da Fase 3, nosso app React Native conta com um Modo Mock Resiliente que garante funcionamento contínuo em demonstrações ou zonas hospitalares sem conectividade."*

---

### 🏁 Bloco 5: Encerramento (`2:45` – `3:00`)
- **Fala sugerida:**
  > *"Com isso, o CardioIA Fase 4 entrega um ciclo completo: pré-processamento rigoroso sem vazamento de pacientes, modelos CNN e Transfer Learning validados, governança ética com auditoria de Fairness e uma aplicação mobile e web pronta para apoiar decisões médicas reais. Muito obrigado!"*
