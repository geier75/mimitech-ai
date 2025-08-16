# MISO_Ultimate 15.32.28 - Multi-Architecture Dockerfile
# Unterstützt AMD64 und ARM64 Architekturen

ARG PY_VER=3.12.3

# Basis-Image mit spezifizierter Python-Version
FROM python:${PY_VER}-slim AS base

# Metadaten gemäß OCI-Image-Spezifikation
LABEL org.opencontainers.image.title="MISO Ultimate"
LABEL org.opencontainers.image.description="MISO Ultimate AI System - Release Candidate 1"
LABEL org.opencontainers.image.version="15.32.28-rc1"
LABEL org.opencontainers.image.vendor="VXOR Labs"
LABEL org.opencontainers.image.authors="MISO Team <miso@vxorlabs.com>"
LABEL org.opencontainers.image.licenses="Proprietary"
LABEL org.opencontainers.image.created="2025-04-30"

# Umgebungsvariablen
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV MISO_ENV="production"
ENV MISO_SECURITY_LEVEL="ULTRA"

# Arbeitsverzeichnis
WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Installiere je nach Architektur die spezifischen Abhängigkeiten
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        pip install --no-cache-dir mlx; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir torch-cuda; \
    fi

# Kopiere den Anwendungscode
COPY . .

# Führe Berechtigungsanpassungen durch
RUN chmod +x /app/scripts/*.py /app/tools/*.py
RUN mkdir -p /app/data /app/logs /app/security/keystore

# Benutzer ohne Root-Rechte erstellen
RUN groupadd -r miso && useradd -r -g miso -d /app miso
RUN chown -R miso:miso /app

# Wechsle zum nicht-privilegierten Benutzer
USER miso

# Container-Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from tools.health_check import run_health_check; sys.exit(0 if run_health_check() else 1)"

# Entrypoint und Standardkommando
ENTRYPOINT ["python"]
CMD ["vxor_launcher.py", "--benchmark", "smoke"]
