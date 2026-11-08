import io
import pytest
from PIL import Image

from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_sample_png(bright_center: bool = True) -> bytes:
    img = Image.new("RGB", (224, 224), color=(25, 25, 25))
    if bright_center:
        for y in range(70, 175):
            for x in range(60, 185):
                img.putpixel((x, y), (210, 210, 210))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_health_endpoint(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "online"
    assert "modelo_ativo" in data
    assert "versao" in data


def test_index_web_ui(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "CardioIA" in html
    assert "#ED145B" in html or "Data Health Solutions" in html


def test_predict_endpoint_multipart(client):
    png_bytes = _make_sample_png(bright_center=True)
    data = {"file": (io.BytesIO(png_bytes), "test_cxr.png"), "limiar": "0.50"}
    resp = client.post("/api/v1/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res = resp.get_json()

    # Verifica compatibilidade de chaves exigidas no RUNBOOK e no KB 06
    for key in [
        "classe",
        "diagnostico",
        "probabilidade",
        "confianca_pct",
        "limiar_clinico",
        "tempo_inferencia_ms",
        "tempo_processamento_ms",
        "status_alerta",
        "status_clinico",
        "recomendacao_clinica",
        "recomendacoes",
    ]:
        assert key in res
    assert 0.0 <= res["probabilidade"] <= 1.0


def test_multimodal_endpoint_fusion(client):
    png_bytes = _make_sample_png(bright_center=True)
    data = {
        "file": (io.BytesIO(png_bytes), "test_cxr.png"),
        "bpm": "138",
        "temperatura": "38.4",
        "queda_detectada": "true",
        "imobilidade_segundos": "180",
    }
    resp = client.post("/api/v1/multimodal", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    res = resp.get_json()
    assert "score_multimodal" in res
    assert "score_news2_iot" in res
    assert "probabilidade_cnn" in res
    assert "classificacao_risco_global" in res
    assert res["score_multimodal"] >= 50.0
