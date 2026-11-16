# Production image: installs the mem0 package from this checkout (so local
# provider patches such as the Voyage embedder are included) plus the REST server.
FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt /tmp/requirements.txt
RUN grep -v '^mem0ai' /tmp/requirements.txt > /tmp/server-requirements.txt \
    && pip install --no-cache-dir -r /tmp/server-requirements.txt

COPY pyproject.toml README.md LICENSE /src/
COPY mem0 /src/mem0
RUN pip install --no-cache-dir /src

COPY server /app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# MEM0_ROOT_PATH is the public path prefix the reverse proxy strips (e.g. /api/v1).
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*' --root-path \"${MEM0_ROOT_PATH:-}\""]
