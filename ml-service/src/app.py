"""FastAPI service exposing the face verification model.

POST /predict with an image file -> { is_schwarzenegger, confidence, threshold }
GET  /health   -> liveness check
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from inference import NoFaceDetectedError, load_artifact, predict_from_image_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-verification-api")

_model, _image_shape, _threshold = None, None, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _image_shape, _threshold
    _model, _image_shape, _threshold = load_artifact()
    logger.info(f"Model loaded. image_shape={_image_shape}, threshold={_threshold:.4f}")
    yield


app = FastAPI(title="Face Verification API", version="1.0.0", lifespan=lifespan)


class PredictionResponse(BaseModel):
    is_schwarzenegger: bool
    confidence: float
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    start = time.perf_counter()
    try:
        result = predict_from_image_bytes(image_bytes, _model, _image_shape, _threshold)
    except NoFaceDetectedError:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"prediction confidence={result['confidence']:.4f} "
        f"is_match={result['is_schwarzenegger']} latency_ms={latency_ms:.1f}"
    )
    return result
