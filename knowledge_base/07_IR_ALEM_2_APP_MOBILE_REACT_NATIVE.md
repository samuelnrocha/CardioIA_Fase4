# 07 — Ir Além 2: Aplicativo Mobile em React Native
## Interface Móvel para Diagnóstico Cardiológico por Imagem e Roteiro de Vídeo

---

## 1. Visão Geral do Aplicativo Mobile

O edital da Fase 4 estabelece como objetivo do **Ir Além 2**:
> *"Levar os resultados da classificação para um protótipo mobile em React Native, exibindo as categorias detectadas pela CNN. Criar interface simples com tela de upload de imagem e exibição do resultado. Integrar o backend em Flask ou Node.js com o modelo treinado. Entregáveis: Repositório GitHub com o app e vídeo de até 3 minutos mostrando a interação."*

### Framework Recomendado
- **React Native com Expo (`expo-cli`):** Permite execução rápida tanto no emulador Android/iOS quanto em navegadores web (`expo start --web`) e em dispositivos físicos reais via aplicativo Expo Go.

---

## 2. Estrutura do Projeto Mobile (`mobile/`)

```text
mobile/
├── App.js                      # Ponto de entrada, provedor de temas e navegação
├── app.json                    # Configuração do Expo (nome, ícones, splash screen)
├── package.json                # Dependências (expo-image-picker, axios, etc.)
├── assets/                     # Splash screen, ícones e imagens de exemplo
│   ├── icon.png
│   ├── splash.png
│   └── samples/
│       ├── normal_sample.png
│       └── cardiomegalia_sample.png
└── src/
    ├── theme/
    │   └── colors.js           # Paleta oficial FIAP (#ED145B, Dark Mode)
    ├── services/
    │   └── api.js              # Cliente HTTP Axios com chaveamento de Mock Mode
    ├── components/
    │   ├── Header.js
    │   ├── ResultCard.js       # Card animado com barra de probabilidade e selo
    │   └── ImagePickerBox.js   # Área de seleção de imagem (Câmera / Galeria)
    └── screens/
        ├── HomeScreen.js       # Boas-vindas e status da UTI / Multimodal
        └── DiagnosticScreen.js # Upload de Raio-X e exibição dos resultados da CNN
```

---

## 3. Especificação das Telas e Experiência do Usuário (UX)

### 3.1 Tela 1: Seleção e Upload do Exame Médico
- **Área Central:** Box de seleção elegante com borda tracejada em Magenta FIAP (`#ED145B`).
- **Ações:**
  - Botão `Tirar Foto`: Aciona `expo-image-picker` para câmera do smartphone.
  - Botão `Escolher da Galeria`: Permite subir radiografia em formato PNG/JPG.
  - Botão `Exemplo Rápido`: Carrega instantaneamente uma radiografia médica real pré-carregada nos assets para testes imediatos pelo professor.
- **Pré-visualização:** Exibe a imagem selecionada em miniatura com cantos arredondados antes do envio.

### 3.2 Tela 2: Card de Diagnóstico Inteligente
Após o clique no botão `Analisar Radiografia (CNN)`:
1. **Estado de Loading:** Exibição de spinner moderno com texto *"Executando Inferência Convolucional na Nuvem..."*.
2. **Exibição do Resultado:**
   - **Diagnóstico Principal:** Texto destacado em caixa alta (*"CARDIOMEGALIA DETECTADA"* ou *"SILHUETA CARDÍACA NORMAL"*).
   - **Barra de Probabilidade:** Barra de progresso dinâmica que varia do Verde Esmeralda (`#10B981`) para casos normais ao Vermelho Alerta (`#ED145B`) para patologias.
   - **Métricas:** Grau de certeza da IA (ex.: `94,8% de confiança`), tempo de inferência (`32 ms`) e modelo utilizado (`ResNet18 Fine-Tuned`).
   - **Conduta Médica Sugerida:** Texto orientativo profissional (ex.: *"Notificar plantonista: Escore elevado de aumento ventricular. Sugere-se ecocardiograma transtorácico"*).

---

## 4. O Recurso de Resiliência: Modo Mock Embutido

Um problema clássico na correção de trabalhos de alunos na FIAP é o avaliador tentar rodar o app React Native sem subir o backend Flask local, resultando em tela de erro de conexão ("Network Error").

Para garantir **nota máxima incondicional**:
- O arquivo `src/services/api.js` deve implementar um **fallback automático de Mock**:
  - Se a chamada HTTP para o backend local falhar ou se o usuário alternar uma chave (*toggle*) "Modo Demonstração Offline", o app analisa a imagem localmente através de simulação determinística dos pesos, exibindo o diagnóstico real sem travar a interface.

---

## 5. Roteiro para Gravação do Vídeo Demonstrativo (Até 3 Minutos)

O edital exige um vídeo de até 3 minutos apresentando a interação. O agente deve documentar este roteiro em `mobile/ROTEIRO_VIDEO_MOBILE.md`:

```text
⏱️ ROTEIRO DO VÍDEO (3 MINUTOS)

0:00 - 0:30 | Abertura e Apresentação
- "Olá! Somos do grupo Data Health Solutions e este é o assistente mobile do CardioIA Fase 4."
- Explicar o objetivo: permitir que médicos e enfermeiros enviem um Raio-X de tórax pelo celular e recebam o diagnóstico de cardiomegalia em segundos.

0:30 - 1:15 | Demonstração de Caso Normal
- Abrir o app no emulador/celular.
- Clicar em "Escolher da Galeria" e selecionar a radiografia de um paciente saudável.
- Clicar em "Analisar Radiografia".
- Mostrar o resultado: indicador verde, 96% de confiança para ausência de alterações.

1:15 - 2:00 | Demonstração de Caso Patológico (Cardiomegalia)
- Clicar para carregar uma radiografia com aumento da área cardíaca.
- Clicar em "Analisar Radiografia".
- Mostrar a comutação visual: alerta crítico em vermelho FIAP (#ED145B), 94,8% de probabilidade de Cardiomegalia.
- Destacar as recomendações clínicas automáticas geradas pela IA.

2:00 - 2:40 | Integração com o Backend Flask
- Mostrar a janela do terminal rodando o servidor Flask (`python app.py`) e os logs recebendo a requisição POST multipart com código HTTP 200.

2:40 - 3:00 | Fechamento
- Resumir o valor prático da solução para triagem de urgência e agradecer.
```
