FROM python:3.10-slim

WORKDIR /app

# Copy model artifacts
COPY mlruns /mlruns

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.1 \
    matplotlib==3.9.2

# Set MLflow model path
ENV MODEL_PATH=/mlruns/0

# Expose port
EXPOSE 5000

# Run MLflow model server
CMD ["mlflow", "models", "serve", "-m", "${MODEL_PATH}/artifacts/model", "-h", "0.0.0.0", "-p", "5000"]
