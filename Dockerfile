# Imagem da API de inferência (FastAPI) para deploy na AWS
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/churn_model.joblib

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY models ./models

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
