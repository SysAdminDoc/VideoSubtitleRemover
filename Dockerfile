FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a AS ffmpeg-build

ARG FFMPEG_VERSION=9.0.1
ARG FFMPEG_SHA256=cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
        xz-utils \
    && curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
        -o /tmp/ffmpeg.tar.xz \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && mkdir /tmp/ffmpeg-src \
    && tar -xJf /tmp/ffmpeg.tar.xz --strip-components=1 -C /tmp/ffmpeg-src \
    && cd /tmp/ffmpeg-src \
    && ./configure \
        --prefix=/usr/local \
        --disable-shared \
        --enable-static \
        --disable-debug \
        --disable-doc \
        --disable-ffplay \
        --disable-x86asm \
    && make -j2 \
    && make install \
    && ffmpeg -version | grep -q "^ffmpeg version ${FFMPEG_VERSION}" \
    && ffprobe -version | grep -q "^ffprobe version ${FFMPEG_VERSION}" \
    && rm -rf /var/lib/apt/lists/* /tmp/ffmpeg.tar.xz /tmp/ffmpeg-src

FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VSR_LOCAL_SMOKE=1 \
    VSR_DEPENDENCY_PROFILE=cpu \
    VSR_FFMPEG_REVIEWED_VERSION=9.0.1 \
    VSR_FFMPEG_REVIEWED_SOURCE=https://ffmpeg.org/download.html

WORKDIR /app

COPY --from=ffmpeg-build /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-build /usr/local/bin/ffprobe /usr/local/bin/ffprobe

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && ffmpeg -version | grep -q "^ffmpeg version 9.0.1" \
    && ffprobe -version | grep -q "^ffprobe version 9.0.1" \
    && rm -rf /var/lib/apt/lists/*

COPY dependency_profiles ./dependency_profiles
COPY requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        --constraint dependency_profiles/cpu.txt \
        -r requirements.txt \
        "onnxruntime>=1.26.0"

COPY . .

RUN python -m backend.dependency_profiles check \
    && python tools/local_smoke.py --skip-self-test

ENTRYPOINT ["python", "-m", "backend.cli"]
CMD ["--help"]
