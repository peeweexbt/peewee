# Scheduler image for a CPU VPS (Fly.io / Railway / any Docker host).
# Painting happens on a GPU box via painter.backend: remote_sd (or replicate).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-scheduler.txt .
RUN pip install --no-cache-dir -r requirements-scheduler.txt
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "peewee.main", "loop"]
