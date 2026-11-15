# Opcional: a API também roda sem Docker (ver "Hospedagem" no README).
# Build a partir da raiz do repositório (contexto "."):
#   docker build -f backend/Dockerfile .
FROM python:3.13-slim

WORKDIR /app/backend

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Mesma estrutura do repositório: a API serve ../frontend/index.html
COPY backend /app/backend
COPY frontend /app/frontend

EXPOSE 8000

# 1 worker: a indexação roda em segundo plano dentro do processo
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
