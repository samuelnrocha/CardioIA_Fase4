"""
Configurações Globais do Servidor Backend REST — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

HOST = os.getenv("CARDIOIA_HOST", "0.0.0.0")
PORT = int(os.getenv("CARDIOIA_PORT", "5000"))
DEBUG = os.getenv("CARDIOIA_DEBUG", "false").lower() == "true"

RESNET_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cardio_resnet.pt")
SCRATCH_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_cardio_cnn_scratch.pt")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_resnet.onnx")

DEFAULT_THRESHOLD = 0.50
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "dicom"}

# Pesos de Fusão Tardia (Late Fusion) Multimodal com Telemetria IoT da Fase 3
ALPHA_IOT_WEIGHT = 0.40
BETA_CNN_WEIGHT = 0.60
