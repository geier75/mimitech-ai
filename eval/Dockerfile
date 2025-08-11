# vXor Evaluator Pack Dockerfile
# Schlanke, reproduzierbare Umgebung für Benchmarks (ARC, GLUE, IMO)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Minimal notwendige System-Pakete
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
COPY eval/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Projektcode hinein kopieren
COPY . /app

# Standardkommando zeigt Makefile-Hilfe
CMD ["bash", "-lc", "make help"]
