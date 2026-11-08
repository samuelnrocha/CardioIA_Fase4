"""
Aplicação Principal Flask REST & Interface Web — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)

Endpoints:
- GET  /                   : Interface Web interativa no padrão FIAP Dark (#ED145B)
- GET  /api/v1/health      : Status do serviço e modelo ativo na memória
- POST /api/v1/predict     : Inferência convolucional (multipart/form-data)
- POST /api/v1/multimodal  : Fusão tardia entre Visão Computacional (Fase 4) e IoT (Fase 3)
"""

import os
import sys
from flask import Flask
from flask_cors import CORS

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.config import HOST, PORT, DEBUG
from backend.model_service import CardioModelService
from backend.routes import api_bp


def create_app() -> Flask:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    CORS(app)

    service = CardioModelService()
    app.config["MODEL_SERVICE"] = service
    app.register_blueprint(api_bp)
    return app


app = create_app()

if __name__ == "__main__":
    print(f"🚀 CardioIA Fase 4 Backend iniciado em http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
