# Harbor Deployment Commands

This document contains all commands needed to build and deploy the Elasticsearch PoC services to Harbor at **harbor.kevin.local** (192.168.86.148).

## Prerequisites

- microk8s cluster running
- buildah installed on cluster machine
- Harbor accessible at harbor.kevin.local
- **Docker daemon NOT required** - buildah builds images directly

---

## DataLoader Service

### Option 1: Automated Deployment (Recommended)

```bash
cd "DataLoader Service"
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh
```

### Option 2: Manual Steps

#### Navigate to directory
```bash
cd "DataLoader Service"
```

#### Build, Tag, and Push using buildah (no Docker required)
```bash
# Build the image with buildah
buildah bud -t dataloader-service:latest -f Dockerfile .

# Tag for Harbor
buildah tag dataloader-service:latest harbor.kevin.local/library/dataloader-service:latest

# Push to Harbor (without TLS verification)
buildah push --tls-verify=false harbor.kevin.local/library/dataloader-service:latest

# Pull from Harbor into microk8s
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/dataloader-service:latest
```

### Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment-gpu.yaml
kubectl apply -f k8s/service-gpu.yaml
```

### Verify
```bash
# Check if pod is running on stormtrooper node
kubectl get pods -l app=dataloader-service-gpu -o wide

# Check logs
kubectl logs -l app=dataloader-service-gpu --tail=50

# Check service
kubectl get svc -l app=dataloader-service-gpu
```

---

## WebUI Service

### Option 1: Automated Deployment (Recommended)

```bash
cd WebUI
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh
```

### Option 2: Manual Steps

#### Navigate to directory
```bash
cd WebUI
```

#### Build, Tag, and Push using buildah (no Docker required)
```bash
# Build the image with buildah
buildah bud -t elasticsearch-webui:latest -f Dockerfile .

# Tag for Harbor
buildah tag elasticsearch-webui:latest harbor.kevin.local/library/elasticsearch-webui:latest

# Push to Harbor (without TLS verification)
buildah push --tls-verify=false harbor.kevin.local/library/elasticsearch-webui:latest

# Pull from Harbor into microk8s
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/elasticsearch-webui:latest
```

#### Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

#### Verify
```bash
kubectl get pods -l app=elasticsearch-webui -o wide
kubectl logs -l app=elasticsearch-webui --tail=50
kubectl get svc elasticsearch-webui
```

---

## Quick Deploy All Services

Deploy everything at once using the automated scripts:

```bash
# From the repo root
cd "DataLoader Service"
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh

cd ../WebUI
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh

cd ..
```

---

## Troubleshooting

### Cannot push to Harbor with buildah
- Using `--tls-verify=false` flag bypasses certificate validation
- Check that the `library` project exists in Harbor
- Verify Harbor is accessible: `curl http://harbor.kevin.local` or `curl http://192.168.86.148`

### Pod fails to pull image
- Check if image exists in Harbor: visit http://harbor.kevin.local or http://192.168.86.148 in browser
- Verify image is in microk8s: `microk8s ctr images ls | grep harbor`
- Check pod events: `kubectl describe pod <pod-name>`
- Ensure image was pulled: `microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/<image-name>:latest`

### Image not found in buildah
- Check tar file was created: `ls -lh *.tar`
- Verify microk8s has the image: `microk8s ctr images ls`
- List buildah images: `buildah images`

### DNS resolution issues
If `harbor.kevin.local` doesn't resolve:
```bash
# Check /etc/hosts has the entry
cat /etc/hosts | grep harbor

# Or use the IP directly
# Replace harbor.kevin.local with 192.168.86.148 in scripts
```

---

## Architecture

```
Build Machine (microk8s node)
    ↓ buildah bud (no Docker daemon)
  Buildah Storage
    ↓ buildah tag
  Tagged Image
    ↓ buildah push --tls-verify=false
Harbor Registry (harbor.kevin.local / 192.168.86.148)
    ├── library/dataloader-service:latest
    └── library/elasticsearch-webui:latest
              ↓ microk8s ctr images pull
    Kubernetes Cluster (microk8s)
    ├── DataLoader Service (stormtrooper node)
    └── WebUI Service
```

---

## Workflow Summary

The deployment process follows this simplified workflow (no Docker required):
1. **Build** with buildah (`buildah bud`)
2. **Tag** for Harbor registry
3. **Push** to Harbor (without TLS verification)
4. **Pull** from Harbor into microk8s
5. **Deploy** Kubernetes manifests

This ensures the image is available both in Harbor (for persistence/sharing) and in microk8s containerd (for immediate pod startup).

**Why buildah instead of Docker?**
- Buildah works without a daemon
- Better suited for CI/CD and rootless builds
- Compatible with Dockerfiles
- Native integration with container registries