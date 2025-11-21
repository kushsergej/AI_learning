# Use a slim Python image as a base
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy dependency files and install packages first (for better Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model
COPY download_model.py .
RUN python3 download_model.py

# Copy the application code
COPY main.py .

# Expose the port (FastAPI default is 8000)
EXPOSE 8000

# 4. Set environment variable for the model path
ENV MODEL_PATH=/app/model_cache

# Define the command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]