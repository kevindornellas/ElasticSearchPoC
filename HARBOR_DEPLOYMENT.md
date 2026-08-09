# Harbor Deployment Commands

This document contains all commands needed to build and deploy the Elasticsearch PoC services to Harbor at **harbor.kevin.local** (192.168.86.147).

## Prerequisites

- microk8s cluster running
- buildah installed on cluster machine
- Harbor accessible at harbor.kevin.local

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

#### Build, Tag, and Push using buildah workflow
```bash
# Build the image with Docker
docker build -t dataloader-service:latest -f Dockerfile .

# Import into microk8s
docker save dataloader-service:latest | microk8s ctr image import -

# Export from microk8s to tar
microk8s ctr images export dataloader-service.tar dataloader-service:latest

# Pull into buildah
buildah pull docker-archive:dataloader-service.tar

# Tag for Harbor
buildah tag dataloader-service:latest harbor.kevin.local/library/dataloader-service:latest

# Push to Harbor (without TLS verification)
buildah push --tls-verify=false harbor.kevin.local/library/dataloader-service:latest

# Clean up tar file
rm dataloader-service.tar

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

#### Build, Tag, and Push using buildah workflow
```bash
# Build the image with Docker
docker build -t elasticsearch-webui:latest -f Dockerfile .

# Import into microk8s
docker save elasticsearch-webui:latest | microk8s ctr image import -

# Export from microk8s to tar
microk8s ctr images export elasticsearch-webui.tar elasticsearch-webui:latest

# Pull into buildah
buildah pull docker-archive:elasticsearch-webui.tar

# Tag for Harbor
buildah tag elasticsearch-webui:latest harbor.kevin.local/library/elasticsearch-webui:latest

# Push to Harbor (without TLS verification)
buildah push --tls-verify=false harbor.kevin.local/library/elasticsearch-webui:latest

# Clean up tar file
rm elasticsearch-webui.tar

# Pull from Harbor into microk8s
Deploy everything at once using the automated scripts:

```bash
# From the repo root
cd "DataLoader Service"
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh

cd ../WebUI
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh
kubectl logs -l app=elasticsearch-webui --tail=50

# Check service and get LoadBalancer IP
kubectl get svc -l app=elasticsearch-webui
```

---

## Quick Deploy All Services

If you want to deploy everything at once:

```bash
# From the repo root
cd "DataLoader Service"
docker build -t dataloader-service:latest .
docker tag dataloader-service:latest 192.168.86.147/library/dataloader-service:latest
docker push 192.168.86.147/library/dataloader-service:latest
kubectl apply -f k8s/deployment-gpu.yaml -f k8s/service-gpu.yaml

cd ../WebUI
docker build -t elasticsearch-webui:latest .
docker tag elasticsearch-webui:latest 192.168.86.147/library/elasticsearch-webui:latest
docker push 192.168.86.147/library/elasticsearch-webui:latest
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml

cd ..
```

---

## Troubleshooting

### Cannot push to Harbor
- Ensure you're logged in: `docker login 192.168.86.147`
- Check that the `library` project exists in Harbor
- Verify Harbor is accessible: `curl http://192.168.86.147`
 with buildah
- Using `--tls-verify=false` flag bypasses certificate validation
- Check that the `library` project exists in Harbor
- Verify Harbor is accessible: `curl http://harbor.kevin.local` or `curl http://192.168.86.147`

### Pod fails to pull image
- Check if image exists in Harbor: visit http://harbor.kevin.local or http://192.168.86.147 in browser
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
# Replace harbor.kevin.local with 192.168.86.147 in scripts
```

---

## Architecture

```
Build Machine
    ↓ docker build
  Local Image
    ↓ microk8s ctr import
  microk8s containerd
    ↓ export to tar
  TAR Archive
    ↓ buildah pull
  Buildah Storage
    ↓ buildah push --tls-verify=false
Harbor Registry (harbor.kevin.local / 192.168.86.147)
    ├── library/dataloader-service:latest
    └── library/elasticsearch-webui:latest
              ↓ microk8s ctr images pull
    Kubernetes Cluster (microk8s)
    ├── DataLoader Service (stormtrooper node)
    └── WebUI Service
```

---

## Workflow Summary

The deployment process follows this workflow:
1. **Build** with Docker
2. **Import** into microk8s containerd
3. **Export** to tar archive
4. **Pull** tar into buildah
5. **Tag** for Harbor registry
6. **Push** to Harbor (without TLS verification)
7. **Pull** from Harbor into microk8s
8. **Deploy** Kubernetes manifests

This ensures the image is available both in Harbor (for persistence/sharing) and in microk8s containerd (for immediate pod startup).