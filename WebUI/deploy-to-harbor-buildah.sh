#!/bin/bash
# Deploy WebUI to Harbor using buildah and microk8s
# Based on the actual workflow used in the cluster

set -e

# Configuration
IMAGE_NAME="elasticsearch-webui"
IMAGE_TAG="latest"
HARBOR_REGISTRY="harbor.kevin.local"
HARBOR_PROJECT="library"
HARBOR_IMAGE="${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}"
TAR_FILE="${IMAGE_NAME}.tar"

echo "Step 1: Building Docker image..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .

echo ""
echo "Step 2: Importing into microk8s..."
docker save "${IMAGE_NAME}:${IMAGE_TAG}" | microk8s ctr image import -

echo ""
echo "Step 3: Exporting from microk8s to tar..."
microk8s ctr images export "${TAR_FILE}" "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "Step 4: Pulling tar into buildah..."
buildah pull "docker-archive:${TAR_FILE}"

echo ""
echo "Step 5: Tagging for Harbor..."
buildah tag "${IMAGE_NAME}:${IMAGE_TAG}" "${HARBOR_IMAGE}"

echo ""
echo "Step 6: Pushing to Harbor..."
buildah push --tls-verify=false "${HARBOR_IMAGE}"

echo ""
echo "Step 7: Cleaning up tar file..."
rm -f "${TAR_FILE}"

echo ""
echo "Step 8: Pulling image into microk8s from Harbor..."
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d "${HARBOR_IMAGE}"

echo ""
echo "Step 9: Applying Kubernetes deployment..."
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
