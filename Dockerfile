FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["bash", "-c", "\
if [ ! -d embeddings ]; then \
  echo 'Running setup...'; \
  python src/ingestion/fetch_docs.py && \
  python src/ingestion/chunk.py && \
  python src/retrieval/embed_store.py; \
fi && \
python app.py"]