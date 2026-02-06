# Elasticsearch Web UI

A web-based dashboard for managing Elasticsearch data and loading the MS MARCO dataset.

## Features

- **System Health Monitoring** - Real-time status of Elasticsearch and Data Loader service
- **Index Management** - View all indices with document counts and sizes
- **Data Loading** - Load MS MARCO dataset with configurable options and progress tracking
- **Data Deletion** - Clear all indices or specific indices
- **Search** - Quick search functionality to test indexed data

## Screenshots

The UI provides:
- Status cards showing Elasticsearch and Data Loader health
- List of all indices with statistics
- Form to configure and start MS MARCO data loading
- Progress bar showing loading status
- Delete functionality with confirmation
- Search interface to query indexed documents

## Build and Deploy to microk8s

### Prerequisites

Make sure the following are already deployed:
1. Elasticsearch (from `Elastic Search Deployment/`)
2. DataLoader Service (from `DataLoader Service/`)

### 1. Enable MetalLB for LoadBalancer support (if not already enabled)

```bash
# Enable MetalLB addon for LoadBalancer support
microk8s enable metallb

# When prompted, enter an IP range for the load balancer
# Example: 192.168.1.240-192.168.1.250
```

### 2. Build the Docker image

```bash
# Navigate to the WebUI folder
cd WebUI

# Build the image using Docker
docker build -t elasticsearch-webui:latest .

# Import into microk8s
docker save elasticsearch-webui:latest | microk8s ctr image import -
```

Or using microk8s directly:

```bash
microk8s ctr image build -t elasticsearch-webui:latest .
```

### 3. Deploy to Kubernetes

```bash
microk8s kubectl apply -f k8s/deployment.yaml
microk8s kubectl apply -f k8s/service.yaml
```

### 4. Get the LoadBalancer IP

```bash
microk8s kubectl get service elasticsearch-webui
```

Look for the `EXTERNAL-IP` column. This is the IP address to access the Web UI.

### 5. Access the Web UI

Open your browser and navigate to:
```
http://<EXTERNAL-IP>/
```

## Configuration

The Web UI connects to the DataLoader Service via the NodePort at port 30800. If your setup is different, update the `API_BASE_URL` in [public/app.js](public/app.js):

```javascript
const API_BASE_URL = 'http://<your-node-ip>:30800';
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │   Web UI         │    │  DataLoader      │              │
│  │   (nginx)        │───▶│  Service         │              │
│  │   LoadBalancer   │    │  NodePort:30800  │              │
│  │   Port: 80       │    │                  │              │
│  └──────────────────┘    └────────┬─────────┘              │
│                                   │                         │
│                                   ▼                         │
│                          ┌──────────────────┐              │
│                          │  Elasticsearch   │              │
│                          │  ClusterIP:9200  │              │
│                          └──────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Local Development

To run the Web UI locally for development:

```bash
# Using Python's built-in server
cd public
python -m http.server 3000

# Or using Node.js http-server
npx http-server public -p 3000
```

Then open http://localhost:3000 in your browser.

**Note:** For local development, make sure the DataLoader Service is accessible at `http://localhost:8000` or update the `API_BASE_URL` in `app.js`.

## Troubleshooting

### LoadBalancer shows `<pending>` for EXTERNAL-IP

This means MetalLB is not configured or the IP pool is exhausted:

```bash
# Check if MetalLB is enabled
microk8s status

# Enable MetalLB with an IP range
microk8s enable metallb:192.168.1.240-192.168.1.250
```

### Cannot connect to DataLoader Service

1. Check if the DataLoader service is running:
   ```bash
   microk8s kubectl get pods -l app=dataloader-service
   ```

2. Check if the NodePort service is accessible:
   ```bash
   curl http://<node-ip>:30800/health
   ```

3. If using a different port, update `app.js` with the correct URL.

### CORS errors in browser console

The DataLoader Service should allow CORS by default. If you see CORS errors, you may need to add CORS middleware to the FastAPI application.
