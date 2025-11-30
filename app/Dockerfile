FROM python:3.13-slim AS builder

WORKDIR /app

COPY app_requirements.txt .

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir --user -r app_requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --user -r app_requirements.txt


FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.13/site-packages:$PYTHONPATH \
    MODEL_PATH=/app/model_snapshot

COPY main.py .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]