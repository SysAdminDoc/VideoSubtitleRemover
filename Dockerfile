FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VSR_LOCAL_SMOKE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        "numpy>=1.26" \
        "opencv-python-headless>=4.12.0" \
        "Pillow>=12.2.0" \
        "onnxruntime>=1.21.0"

COPY . .

CMD ["python", "tools/local_smoke.py"]
