# Elasticsearch Data Loader Service

A FastAPI web service for managing Elasticsearch data and loading the MS MARCO dataset.

## Features

- **Clear all data** - Delete all indices from Elasticsearch
- **Clear specific index** - Delete a specific index
- **Load MS MARCO dataset** - Download and index the MS MARCO dataset from Hugging Face
- **Search** - Search indexed documents
- **Health checks** - Monitor service and Elasticsearch health

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health check (service + Elasticsearch) |
| GET | `/status` | Get current data loading status |
| GET | `/indices` | List all indices |
| DELETE | `/clear` | Clear all data (delete all indices) |
| DELETE | `/clear/{index_name}` | Clear a specific index |
| POST | `/load/msmarco` | Load MS MARCO dataset (background task) |
| GET | `/search/{index_name}?q=query&size=10` | Search documents |

## Build and Deploy to microk8s

### 1. Build the Docker image

```bash
# Navigate to the DataLoader Service folder
cd "DataLoader Service"

# Build the image using microk8s built-in registry
microk8s ctr image build -t dataloader-service:latest .
```

Or if using Docker:

```bash
docker build -t dataloader-service:latest .

# Import into microk8s
docker save dataloader-service:latest | microk8s ctr image import -
```

### 2. Deploy to Kubernetes

```bash
# Make sure Elasticsearch is deployed first
microk8s kubectl apply -f "../Elastic Search Deployment/elasticsearch-deployment.yaml"
microk8s kubectl apply -f "../Elastic Search Deployment/elasticsearch-service.yaml"

# Wait for Elasticsearch to be ready
microk8s kubectl wait --for=condition=ready pod -l app=elasticsearch --timeout=120s

# Deploy the DataLoader service
microk8s kubectl apply -f k8s/deployment.yaml
microk8s kubectl apply -f k8s/service.yaml
```

### 3. Verify deployment

```bash
microk8s kubectl get pods
microk8s kubectl get services
```

## Usage Examples

### Access the service

The service is available at:
- **Internal (within cluster):** `http://dataloader-service:8000`
- **External (NodePort):** `http://<node-ip>:30800`

### Check health

```bash
curl http://<node-ip>:30800/health
```

### Clear all Elasticsearch data

```bash
curl -X DELETE http://<node-ip>:30800/clear
```

### Load MS MARCO dataset

```bash
# Load with default settings (full dataset)
curl -X POST http://<node-ip>:30800/load/msmarco \
  -H "Content-Type: application/json" \
  -d '{}'

# Load with custom settings (limit documents for testing)
curl -X POST http://<node-ip>:30800/load/msmarco \
  -H "Content-Type: application/json" \
  -d '{
    "index_name": "msmarco",
    "dataset_split": "train",
    "max_documents": 10000,
    "batch_size": 500
  }'
```

### Check loading progress

```bash
curl http://<node-ip>:30800/status
```

### Search documents

```bash
curl "http://<node-ip>:30800/search/msmarco?q=machine+learning&size=5"
```

### List indices

```bash
curl http://<node-ip>:30800/indices
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ELASTICSEARCH_HOST` | `elasticsearch` | Elasticsearch hostname |
| `ELASTICSEARCH_PORT` | `9200` | Elasticsearch port |
| `ELASTICSEARCH_INDEX` | `msmarco` | Default index name |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires Elasticsearch running)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API docs
open http://localhost:8000/docs
```

## MS MARCO Dataset

The [MS MARCO](https://microsoft.github.io/msmarco/) (Microsoft Machine Reading Comprehension) dataset is a large-scale dataset for machine reading comprehension and question answering.

The service loads the dataset from Hugging Face: https://huggingface.co/datasets/ms_marco

**Note:** The full dataset is large and loading can take significant time and resources. Use the `max_documents` parameter to limit the data for testing purposes.
