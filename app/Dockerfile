# 1. Use the official vLLM image as base to inherit compiled CUDA kernels
# This saves ~15 minutes of build time and ensures GPU compatibility.
FROM vllm/vllm-openai:latest


# 2. Set working directory
WORKDIR /app


# 3. Install your app-specific dependencies (vLLM, torch, and CUDA libraries are already installed in the base)
COPY app_requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r app_requirements.txt && \
    pip cache purge


# 4. Copy your application code
COPY main.py .


# 5. Environment variables for the model
ENV MODEL_PATH=/app/model_snapshot


# 6. Expose the port (FastAPI default is 8000)
EXPOSE 8000


# 7. Override the base image's entrypoint (which defaults to vllm server)
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]