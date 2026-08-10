#!/bin/bash
# Deploy WebUI to Harbor using buildah and microk8s (no Docker required)
# This version uses buildah to build directly from Dockerfile

set -e

# Configuration
IMAGE_NAME="elasticsearch-webui"
IMAGE_TAG="latest"
HARBOR_REGISTRY="192.168.86.148"
HARBOR_PROJECT="library"
HARBOR_IMAGE="${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Step 1: Building image with buildah..."
buildah bud -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .

echo ""
echo "Step 2: Tagging for Harbor..."
buildah tag "${IMAGE_NAME}:${IMAGE_TAG}" "${HARBOR_IMAGE}"

echo ""
echo "Step 3: Pushing to Harbor..."
buildah push --tls-verify=false "${HARBOR_IMAGE}"

echo ""
echo "Step 4: Pulling image into microk8s from Harbor..."
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d "${HARBOR_IMAGE}"

echo ""
echo "Step 5: Applying Kubernetes deployment..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo ""
echo "✅ Deployment successful!"
echo "Image: ${HARBOR_IMAGE}"
echo ""
echo "Check status with:"
echo "  kubectl get pods -l app=elasticsearch-webui -o wide"
echo "  kubectl logs -l app=elasticsearch-webui --tail=50"
echo "  kubectl get svc elasticsearch-webui"
