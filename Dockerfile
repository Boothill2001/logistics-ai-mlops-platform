FROM python:3.11-slim

WORKDIR /app

# Layer-cache dependencies separately from source
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# Static UI served at /
COPY scripts/ scripts/
COPY data/ data/
COPY models/ models/

# Bake the vector index at build time so the container is self-contained
RUN python scripts/ingest_docs.py

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
