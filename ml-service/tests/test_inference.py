"""Unit tests for preprocessing and the prediction seam.

Deliberately avoid depending on any real face photo (no LFW data is
bundled in the repo -- redistributing individual face crops raises
licensing/consent questions the local-only training use doesn't). Where
a "detected face" is needed, we monkeypatch detect_and_crop_face rather
than relying on the Haar cascade actually finding a face in a synthetic
image.
"""
import numpy as np
import cv2
import pytest

from inference import (
    NoFaceDetectedError,
    detect_and_crop_face,
    predict_from_image_bytes,
    preprocess_face,
)


def test_preprocess_face_shape_dtype_and_range():
    raw = (np.random.RandomState(0).rand(80, 60) * 255).astype(np.uint8)
    result = preprocess_face(raw, image_shape=(50, 37))

    assert result.shape == (1, 50 * 37)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_detect_and_crop_face_raises_on_blank_image():
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    with pytest.raises(NoFaceDetectedError):
        detect_and_crop_face(blank)


class _StubModel:
    """Minimal stand-in for the trained sklearn pipeline."""

    def predict_proba(self, X):
        # Fixed confidence regardless of input -- we're testing the
        # plumbing (decode -> crop -> preprocess -> predict -> format),
        # not classifier accuracy (that's evaluated in train.py).
        return np.array([[0.7, 0.3]])


def test_predict_from_image_bytes_happy_path(monkeypatch):
    monkeypatch.setattr(
        "inference.detect_and_crop_face",
        lambda image_bgr: np.random.RandomState(1).randint(0, 255, (80, 60), dtype=np.uint8),
    )

    # Any validly-encoded image bytes work here since detect_and_crop_face
    # is mocked; imdecode just needs to succeed.
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", dummy_image)
    assert success

    result = predict_from_image_bytes(
        encoded.tobytes(), model=_StubModel(), image_shape=(50, 37), threshold=0.5
    )

    assert result["is_schwarzenegger"] is False  # 0.3 < 0.5
    assert result["confidence"] == 0.3
    assert result["threshold"] == 0.5


def test_predict_from_image_bytes_raises_on_undecodable_bytes():
    with pytest.raises(ValueError):
        predict_from_image_bytes(
            b"not an image", model=_StubModel(), image_shape=(50, 37), threshold=0.5
        )
