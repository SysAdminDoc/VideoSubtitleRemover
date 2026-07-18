FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VSR_LOCAL_SMOKE=1 \
    VSR_DEPENDENCY_PROFILE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY dependency_profiles ./dependency_profiles

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        --constraint dependency_profiles/cpu.txt \
        "numpy>=1.26" \
        "opencv-python-headless>=5.0.0.93" \
        "Pillow>=12.3.0" \
        "onnxruntime>=1.26.0"

COPY . .

RUN python -m backend.dependency_profiles check

CMD ["python", "tools/local_smoke.py"]
