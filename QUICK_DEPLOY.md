# Quick Reference - Harbor Deployment Commands

Run these commands on your cluster machine to deploy to Harbor.

**Prerequisites:**
- Docker daemon is not required (uses buildah)
- Harbor hostname must be resolvable: Add `192.168.86.147 harbor.kevin.local` to `/etc/hosts` on all nodes

## DataLoader Service

```bash
cd "DataLoader Service"

# Build with buildah
buildah bud -t dataloader-service:latest -f Dockerfile .

# Tag and push to Harbor
buildah tag dataloader-service:latest harbor.kevin.local/library/dataloader-service:latest
buildah push --tls-verify=false harbor.kevin.local/library/dataloader-service:latest

# Pull from Harbor into microk8s and deploy
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/dataloader-service:latest
kubectl apply -f k8s/deployment-gpu.yaml
kubectl apply -f k8s/service-gpu.yaml

# Verify
kubectl get pods -l app=dataloader-service-gpu -o wide
```

## WebUI Service

```bash
cd WebUI

# Build with buildah
buildah bud -t elasticsearch-webui:latest -f Dockerfile .

# Tag and push to Harbor
buildah tag elasticsearch-webui:latest harbor.kevin.local/library/elasticsearch-webui:latest
buildah push --tls-verify=false harbor.kevin.local/library/elasticsearch-webui:latest

# Pull from Harbor into microk8s and deploy
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/elasticsearch-webui:latest
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify
kubectl get pods -l app=elasticsearch-webui -o wide
```

## Or Use Automated Scripts

```bash
# DataLoader
cd "DataLoader Service"
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh

# WebUI
cd WebUI
chmod +x deploy-to-harbor-buildah.sh
./deploy-to-harbor-buildah.sh
```
