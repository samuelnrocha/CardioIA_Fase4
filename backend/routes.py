"""
Rotas modulares da API REST — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)
"""

from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, render_template, current_app
from backend.config import DEFAULT_THRESHOLD

api_bp = Blueprint("api_bp", __name__)


def _get_service():
    return current_app.config["MODEL_SERVICE"]


@api_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@api_bp.route("/api/v1/health", methods=["GET"])
def health():
    svc = _get_service()
    return jsonify(
        {
            "status": "online",
            "servico": "CardioIA - Assistente Virtual de Visão Computacional",
            "versao": "4.0.0",
            "equipe": "Data Health Solutions",
            "modelo_ativo": svc.model_name,
            "dispositivo_inferencia": str(svc.device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ), 200


@api_bp.route("/api/v1/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado no campo 'file'"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"erro": "Arquivo de imagem vazio ou inválido"}), 400

    try:
        threshold = float(request.form.get("limiar", DEFAULT_THRESHOLD))
        image_bytes = file.read()
        svc = _get_service()
        result = svc.predict_image_bytes(image_bytes, threshold=threshold)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"erro": f"Falha no processamento da imagem: {str(exc)}"}), 500


@api_bp.route("/api/v1/multimodal", methods=["POST"])
def multimodal():
    if "file" not in request.files:
        return jsonify({"erro": "Envie a imagem radiológica no campo 'file'"}), 400

    file = request.files["file"]
    try:
        bpm = int(float(request.form.get("bpm", 78)))
        temperatura = float(request.form.get("temperatura", 36.6))
        queda_raw = str(request.form.get("queda_detectada", "false")).lower()
        queda_detectada = queda_raw in ("true", "1", "yes", "sim")
        imobilidade_segundos = int(float(request.form.get("imobilidade_segundos", 0)))
        threshold = float(request.form.get("limiar", DEFAULT_THRESHOLD))

        image_bytes = file.read()
        svc = _get_service()
        result = svc.compute_multimodal_fusion(
            image_bytes=image_bytes,
            bpm=bpm,
            temperatura=temperatura,
            queda_detectada=queda_detectada,
            imobilidade_segundos=imobilidade_segundos,
            threshold=threshold,
        )
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"erro": f"Falha na fusão multimodal: {str(exc)}"}), 500
