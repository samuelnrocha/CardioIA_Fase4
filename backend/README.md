# Backend REST & API de Inferência — CardioIA Fase 4

Serviço REST em Flask/FastAPI que expõe o modelo de Visão Computacional para o aplicativo mobile e a interface web.

## 🚀 Endpoints
- `GET /api/v1/health`: Verificação de status e modelo ativo.
- `POST /api/v1/predict`: Upload multipart de imagem médica e retorno do diagnóstico.
- `POST /api/v1/multimodal`: Fusão tardia entre imagem torácica e sinais vitais IoT (Fase 3).
- `GET /`: Interface web amigável para testes rápidos no navegador.

Consulte o guia [`knowledge_base/06_IR_ALEM_2_BACKEND_FLASK_API.md`](../knowledge_base/06_IR_ALEM_2_BACKEND_FLASK_API.md).
