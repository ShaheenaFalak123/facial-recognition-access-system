"""FastAPI endpoint tests via TestClient. Requires models/face_classifier.joblib
to exist (produced by `python src/train.py`) since the app loads it on startup.
"""
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _jpeg_bytes(width=200, height=200):
    image = np.random.RandomState(0).randint(0, 255, (height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_rejects_non_image_upload(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_returns_422_when_no_face_detected(client):
    # Random noise won't trigger the Haar cascade's face detector.
    response = client.post(
        "/predict", files={"file": ("random.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert response.status_code == 422


def test_predict_happy_path_with_mocked_face_detection(client, monkeypatch):
    monkeypatch.setattr(
        "inference.detect_and_crop_face",
        lambda image_bgr: np.random.RandomState(2).randint(0, 255, (80, 60), dtype=np.uint8),
    )

    response = client.post(
        "/predict", files={"file": ("face.jpg", _jpeg_bytes(), "image/jpeg")}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"is_schwarzenegger", "confidence", "threshold"}
    assert isinstance(body["is_schwarzenegger"], bool)
    assert 0.0 <= body["confidence"] <= 1.0
