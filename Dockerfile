# Project Sentinel — single-service deploy image.
# Builds the React dashboard, then bakes it + the FastAPI backend + the
# trained model + the held-out replay split into one Python runtime.
# One container, one public URL, /api and /webhook served alongside the
# dashboard's static files.

# ---- stage 1: build the dashboard ----
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: python runtime ----
FROM python:3.14-slim
WORKDIR /app

# LightGBM needs libgomp (OpenMP) on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY train/ train/
COPY models/ models/
COPY report/ report/
COPY data/splits/test.parquet data/splits/test.parquet
COPY --from=frontend /app/frontend/dist frontend/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Render (and most PaaS hosts) inject $PORT; default to 8000 for `docker run` locally.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
