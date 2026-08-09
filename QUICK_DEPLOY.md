# Quick Reference - Harbor Deployment Commands

Run these commands on your cluster machine to deploy to Harbor.

## DataLoader Service

```bash
cd "DataLoader Service"

# Build
docker build -t dataloader-service:latest -f Dockerfile .
docker save dataloader-service:latest | microk8s ctr image import -

# Export and push to Harbor
microk8s ctr images export dataloader-service.tar dataloader-service:latest
buildah pull docker-archive:dataloader-service.tar
buildah tag dataloader-service:latest harbor.kevin.local/library/dataloader-service:latest
buildah push --tls-verify=false harbor.kevin.local/library/dataloader-service:latest
rm dataloader-service.tar

# Pull from Harbor and deploy
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/dataloader-service:latest
kubectl apply -f k8s/deployment-gpu.yaml
kubectl apply -f k8s/service-gpu.yaml

# Verify
kubectl get pods -l app=dataloader-service-gpu -o wide
```

## WebUI Service

```bash
cd WebUI

# Build
docker build -t elasticsearch-webui:latest -f Dockerfile .
docker save elasticsearch-webui:latest | microk8s ctr image import -

# Export and push to Harbor
microk8s ctr images export elasticsearch-webui.tar elasticsearch-webui:latest
buildah pull docker-archive:elasticsearch-webui.tar
buildah tag elasticsearch-webui:latest harbor.kevin.local/library/elasticsearch-webui:latest
buildah push --tls-verify=false harbor.kevin.local/library/elasticsearch-webui:latest
rm elasticsearch-webui.tar

# Pull from Harbor and deploy
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
