# ════════════════════════════════════════════════════════
# Dockerfile — Flight Delay Inference Container
# Project Lecture 2: Approval + Blue/Green Deploy
#
# Multi-stage build:
#   Stage 1 (builder): install dependencies
#   Stage 2 (runtime): copy only what is needed
#
# SageMaker expects:
#   - The container to listen on port 8080
#   - GET /ping    → health check
#   - POST /invocations → prediction
#
# Build locally:
#   docker build -t flight-delay-inference:latest .
#
# Test locally:
#   docker run -p 8080:8080 \
#     -v $(pwd)/models:/opt/ml/model \
#     flight-delay-inference:latest
#
#   curl http://localhost:8080/health
# ════════════════════════════════════════════════════════

# ── Stage 1: Builder ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy only requirements first — Docker layer cache
# If requirements.txt does not change, this layer is cached
COPY requirements-inference.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-inference.txt \
    && pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.8.0

# ── Stage 2: Runtime ────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /opt/program

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
# features.py must be alongside app.py so the import works
COPY src/features.py          ./features.py
COPY src/inference/app.py     ./app.py
COPY src/inference/serve      ./serve

# SageMaker model artefacts are mounted at /opt/ml/model/
# The app looks there first for model.pkl or model.tar.gz
# Do NOT copy model here — it is provided at runtime by SageMaker

# Make serve executable while still root
RUN chmod +x ./serve

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# SageMaker requires port 8080
EXPOSE 8080

# Health check — SageMaker polls /ping before routing traffic
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')"

# SageMaker invokes the container as: docker run <image> serve
# /opt/program must be on PATH so the 'serve' command is found
ENV PATH="/opt/program:${PATH}"
CMD ["serve"]
