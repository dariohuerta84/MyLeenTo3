# ==============================================================================
# MyLeenTo3 — Multi-Stage Reproducible Container Build
# Targets:
#   - backend  : FastAPI + Uvicorn + Geometry/Metrics/Interactions API engine
#   - frontend : Vite production bundle served via Nginx with /api reverse proxy
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Backend (Python 3.12 + FastAPI + NumPy/SciPy/Pillow + google-genai)
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema mínimas para Pillow y healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY backend/app /app/app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------------------
# Stage 2: Frontend Builder (Node 22 + Vite + React 19 + Three.js R3F)
# ------------------------------------------------------------------------------
FROM node:22-alpine AS frontend-builder

WORKDIR /web

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --legacy-peer-deps || npm install --legacy-peer-deps

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 3: Frontend Runtime (Nginx serving static SPA + /api -> backend:8000)
# ------------------------------------------------------------------------------
FROM nginx:1.27-alpine AS frontend

COPY --from=frontend-builder /web/dist /usr/share/nginx/html

RUN printf '%s\n' \
'server {' \
'    listen 80;' \
'    server_name _;' \
'    client_max_body_size 25M;' \
'' \
'    root /usr/share/nginx/html;' \
'    index index.html;' \
'' \
'    location /api/ {' \
'        proxy_pass http://backend:8000/api/;' \
'        proxy_http_version 1.1;' \
'        proxy_set_header Host $host;' \
'        proxy_set_header X-Real-IP $remote_addr;' \
'        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;' \
'        proxy_set_header X-Forwarded-Proto $scheme;' \
'        proxy_read_timeout 120s;' \
'    }' \
'' \
'    location / {' \
'        try_files $uri $uri/ /index.html;' \
'    }' \
'}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
