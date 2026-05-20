# Run from inside final/: docker build -f deploy/Dockerfile -t prereqgraph .
# Or from parent:        docker build -f final/deploy/Dockerfile -t prereqgraph final/

FROM python:3.11-slim

# System deps: git (ogb downloads), build tools for numpy/torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the deployment server code
COPY deploy/ ./deploy/

# Copy the pipeline modules (context root IS final/, so copy everything there)
COPY . ./final/

# Install Python dependencies
RUN pip install --no-cache-dir -r deploy/requirements_deploy.txt

# The dataset/embeddings/outputs directories are expected to be mounted at runtime.
# We create stub directories so the container starts cleanly even without volumes.
RUN mkdir -p final/dataset \
             final/outputs/embeddings \
             final/outputs/metadata

ENV FINAL_DIR=/app/final
ENV SPECTER_PATH=/app/final/outputs/embeddings/specter.npy
ENV GNN_PATH=/app/final/outputs/embeddings/gnn_embeddings.pt
ENV METADATA_PATH=/app/final/outputs/metadata/papers.parquet
ENV DATASET_ROOT=/app/final/dataset
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

WORKDIR /app/deploy

CMD ["python", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log"]
