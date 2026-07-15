"""Preprocessing and prediction for the trained face classifier.

Loads the joblib artifact produced by train.py (pipeline + image_shape +
tuned decision threshold) and exposes a single entry point,
`predict_from_image_bytes`, that FastAPI's endpoint calls.
"""
import os

import cv2
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_classifier.joblib")

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class NoFaceDetectedError(Exception):
    pass


def load_artifact(path=MODEL_PATH):
    artifact = joblib.load(path)
    return artifact["model"], artifact["image_shape"], artifact["threshold"]


_BORDER_PAD = 60  # pixels


def detect_and_crop_face(image_bgr):
    """Find the largest detected face in a BGR image and return a
    grayscale crop of just that face. Raises NoFaceDetectedError if
    none is found.

    We pad the image with replicated border pixels before running the
    cascade. Haar cascade features need some margin around a face to
    evaluate correctly; a tightly-cropped photo where the face fills
    nearly the whole frame (common with already-cropped identity
    photos, which is exactly what our LFW-derived smoke tests use) can
    otherwise go completely undetected even though a face is clearly
    present -- confirmed by testing against a real LFW photo during
    development.

    Note: LFW's own images went through a specific "funneled" alignment
    pipeline during dataset creation. A Haar-cascade crop of an
    arbitrary uploaded photo won't be aligned identically, so real-world
    accuracy on new photos can differ somewhat from the offline
    evaluation metrics in train.py -- worth stating plainly rather than
    implying the two are equivalent.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(
        gray, _BORDER_PAD, _BORDER_PAD, _BORDER_PAD, _BORDER_PAD, cv2.BORDER_REPLICATE
    )
    faces = _FACE_CASCADE.detectMultiScale(padded, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        raise NoFaceDetectedError("No face detected in the uploaded image.")

    # Largest bounding box by area, in case of multiple faces.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return padded[y : y + h, x : x + w]


def preprocess_face(face_gray, image_shape):
    """Resize a grayscale face crop to match training data and flatten,
    matching sklearn's fetch_lfw_people convention: float32 in [0, 1].
    """
    height, width = image_shape
    resized = cv2.resize(face_gray, (width, height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, -1)


def predict_from_image_bytes(image_bytes, model, image_shape, threshold):
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode the uploaded file as an image.")

    face_gray = detect_and_crop_face(image_bgr)
    X = preprocess_face(face_gray, image_shape)

    proba = float(model.predict_proba(X)[0, 1])
    is_match = proba >= threshold

    return {
        "is_schwarzenegger": bool(is_match),
        "confidence": round(proba, 4),
        "threshold": round(threshold, 4),
    }
