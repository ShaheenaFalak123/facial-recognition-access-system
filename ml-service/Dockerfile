FROM python:3.11-slim

WORKDIR /app

# System deps for opencv-python-headless (still needs libGL/glib at runtime
# even though it's the "headless" build).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY models/ models/

EXPOSE 7860

# 7860 matches Hugging Face Spaces' expected container port.
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
