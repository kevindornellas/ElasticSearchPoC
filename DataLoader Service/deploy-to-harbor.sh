#!/bin/bash
# Deploy DataLoader Service to Harbor and Kubernetes
# This script builds, tags, and pushes the Docker image to Harbor registry

set -e

# Configuration
IMAGE_NAME="dataloader-service"
IMAGE_TAG="latest"
HARBOR_REGISTRY="192.168.86.147"
HARBOR_PROJECT="library"
HARBOR_IMAGE="${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .

echo "Tagging image for Harbor..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${HARBOR_IMAGE}"

echo "Pushing image to Harbor registry..."
echo "Note: You may need to login to Harbor first:"
echo "  docker login ${HARBOR_REGISTRY}"
docker push "${HARBOR_IMAGE}"

echo ""
echo "Image successfully pushed to Harbor!"
echo "Image: ${HARBOR_IMAGE}"

echo ""
echo "Applying Kubernetes deployment..."
kubectl apply -f k8s/deployment-gpu.yaml
kubectl apply -f k8s/service-gpu.yaml

echo ""
echo "Deployment successful!"
echo "Check status with: kubectl get pods -l app=dataloader-service-gpu"
