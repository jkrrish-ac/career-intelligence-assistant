FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf/python-docx and sentence-transformers' torch backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding + reranker models into the image at BUILD time, not
# request time — per the PRD's risk mitigation, this is what keeps
# `docker-compose up` from stalling on a cold Hugging Face download the
# first time someone chats. Requires network access at build time.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
