FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY seedinfer_worker ./seedinfer_worker

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 3001

CMD ["seedinfer-worker", "--mock-gpu", "--engine", "mock", "--host", "0.0.0.0", "--port", "3001"]
