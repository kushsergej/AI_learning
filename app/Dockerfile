FROM huggingface/transformers-pytorch-gpu:latest

WORKDIR /app

COPY app_requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r app_requirements.txt && \
    pip cache purge

COPY main.py .

ENV MODEL_PATH=/app/model_snapshot

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]