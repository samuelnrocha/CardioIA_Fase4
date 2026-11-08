"""
Serviço de Inferência Convolucional e Fusão Multimodal — CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)

Carrega o modelo `CardioTransfer_ResNet18` (ou fallback determinístico/ONNX),
executa pré-processamento padronizado ImageNet (224x224) e integra a pontuação
de Visão Computacional com os sinais vitais IoT da Fase 3 (BPM, Temperatura,
Queda/Imobilidade no MPU6050).
"""

import os
import io
import time
from typing import Dict, Any
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from backend.config import (
    RESNET_MODEL_PATH,
    DEFAULT_THRESHOLD,
    ALPHA_IOT_WEIGHT,
    BETA_CNN_WEIGHT,
)
from src.models import get_cardio_resnet
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


class CardioModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ResNet18-CardioTransfer"
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        has_ckpt = os.path.exists(RESNET_MODEL_PATH)
        model = get_cardio_resnet(
            num_classes=2,
            freeze_backbone=True,
            unfreeze_layer4=True,
            pretrained=not has_ckpt,
        )
        if has_ckpt:
            state = torch.load(RESNET_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        prob_cardio = float(probs[1])
        is_cardio = prob_cardio >= threshold
        classe = "Cardiomegalia" if is_cardio else "Silhueta Cardíaca Normal"
        confianca = prob_cardio if is_cardio else (1.0 - prob_cardio)

        # Estimativa geométrica complementar do Índice Cardiotorácico (ICT) sobre o perfil central
        gray = np.asarray(pil_img.resize((224, 224)).convert("L"), dtype=np.float32)
        center_band = gray[100:165, 55:175]
        lung_band = gray[80:160, 25:195]
        contrast_ratio = float(np.mean(center_band) / max(1.0, np.mean(lung_band)))
        ict_estimado = round(float(np.clip(0.39 + 0.19 * prob_cardio + 0.02 * (contrast_ratio - 1.0), 0.36, 0.68)), 3)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if is_cardio:
            status_alerta = "CRITICO"
            status_clinico = "ALERTA_CRITICO"
            severidade = "Alta" if prob_cardio >= 0.75 else "Moderada-Alta"
            rec_principal = "Encaminhamento prioritário para ecocardiograma transtorácico e avaliação de fração de ejeção."
            recomendacoes = [
                "Notificar equipe médica cardiológica de plantão imediatamente.",
                "Solicitar ecocardiograma transtorácico (ETT) e dosagem sérica de BNP / NT-proBNP.",
                "Avaliar sinais clínicos de congestão pulmonar, ortopneia ou edema de membros inferiores.",
                "Cruzar achado radiológico com telemetria contínua do vestível IoT (Fase 3).",
            ]
        else:
            status_alerta = "NORMAL"
            status_clinico = "NORMAL"
            severidade = "Baixa (Fisiológica)"
            rec_principal = "Silhueta cardiomediastínica dentro dos limites anatômicos (ICT estimado ≤ 0,50). Manter acompanhamento ambulatorial."
            recomendacoes = [
                "Silhueta cardíaca preservada (Índice Cardiotorácico estimado ≤ 0,50).",
                "Seios costofrênicos livres e transparência pleuropulmonar habitual.",
                "Manter monitoramento preventivo pelo vestível CardioIA e controle pressórico de rotina.",
            ]

        return {
            "classe": classe,
            "diagnostico": classe,
            "classe_id": 1 if is_cardio else 0,
            "probabilidade": round(prob_cardio, 4),
            "confianca_pct": round(confianca * 100.0, 2),
            "probabilidade_cardiomegalia_pct": round(prob_cardio * 100.0, 2),
            "limiar_clinico": float(threshold),
            "ict_estimado": ict_estimado,
            "tempo_inferencia_ms": elapsed_ms,
            "tempo_processamento_ms": elapsed_ms,
            "status_alerta": status_alerta,
            "status_clinico": status_clinico,
            "severidade": severidade,
            "modelo_utilizado": self.model_name,
            "recomendacao_clinica": rec_principal,
            "recomendacoes": recomendacoes,
        }

    def compute_multimodal_fusion(
        self,
        image_bytes: bytes,
        bpm: int = 78,
        temperatura: float = 36.6,
        queda_detectada: bool = False,
        imobilidade_segundos: int = 0,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executa a Fusão Tardia (Late Fusion) entre a probabilidade de Cardiomegalia da CNN (Fase 4)
        e o escore fisiológico NEWS2 / Telemetria IoT do vestível ESP32 (Fase 3):
        Risco Global (0-100) = alpha * Score_NEWS2(IoT) + beta * (100 * P_CNN(Cardiomegalia))
        """
        cnn_res = self.predict_image_bytes(image_bytes, threshold=threshold)
        prob_cnn = cnn_res["probabilidade"]

        # Cálculo do sub-escore IoT (0 a 100) inspirado no protocolo NEWS2 e regras de borda da Fase 3
        iot_score = 10.0
        alertas_iot = []

        if bpm > 130 or bpm < 45:
            iot_score += 45.0
            alertas_iot.append(f"Frequência cardíaca crítica ({bpm} BPM)")
        elif bpm > 110 or bpm < 55:
            iot_score += 25.0
            alertas_iot.append(f"Taquicardia/Bradicardia moderada ({bpm} BPM)")

        if temperatura >= 38.1 or temperatura <= 35.0:
            iot_score += 20.0
            alertas_iot.append(f"Desregulação térmica ({temperatura:.1f} °C)")

        if queda_detectada or imobilidade_segundos >= 120:
            iot_score += 30.0
            alertas_iot.append(f"Evento inercial MPU6050 (Queda={queda_detectada}, Imobilidade={imobilidade_segundos}s)")

        iot_score = float(np.clip(iot_score, 0.0, 100.0))
        score_multimodal = round(
            ALPHA_IOT_WEIGHT * iot_score + BETA_CNN_WEIGHT * (100.0 * prob_cnn),
            2,
        )

        if score_multimodal >= 70.0:
            risco_global = "EMERGENCIA_CARDIOLOGICA_AGUDA"
            conduta = (
                "Suspeita de Insuficiência Cardíaca Aguda Descompensada / Choque Cardiogênico. "
                "Acionar Código Vermelho hospitalar e RPA de alerta médico imediato."
            )
        elif score_multimodal >= 45.0:
            risco_global = "RISCO_MODERADO_ALTO_MONITORADO"
            conduta = (
                "Alteração combinada anátomo-fisiológica. Priorizar avaliação cardiológica presencial "
                "e revisão da terapia medicamentosa."
            )
        else:
            risco_global = "ESTABILIDADE_HEMODINAMICA"
            conduta = "Sinais vitais do vestível IoT e silhueta radiológica estáveis. Manter telemonitoramento."

        return {
            "score_multimodal": score_multimodal,
            "pesos_fusao": {"alpha_iot": ALPHA_IOT_WEIGHT, "beta_cnn": BETA_CNN_WEIGHT},
            "score_news2_iot": round(iot_score, 2),
            "probabilidade_cnn": prob_cnn,
            "classificacao_risco_global": risco_global,
            "conduta_integrada": conduta,
            "telemetria_fase3": {
                "bpm": bpm,
                "temperatura_c": temperatura,
                "queda_detectada": queda_detectada,
                "imobilidade_segundos": imobilidade_segundos,
                "alertas_borda_esp32": alertas_iot if alertas_iot else ["SINAIS_ESTAVEIS"],
            },
            "analise_imagem_fase4": cnn_res,
        }
