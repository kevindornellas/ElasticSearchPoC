# Deploy DataLoader Service to Harbor and Kubernetes
# This script builds, tags, and pushes the Docker image to Harbor registry

# Configuration
$IMAGE_NAME = "dataloader-service"
$IMAGE_TAG = "latest"
$HARBOR_REGISTRY = "192.168.86.147"
$HARBOR_PROJECT = "library"
$HARBOR_IMAGE = "$HARBOR_REGISTRY/$HARBOR_PROJECT/${IMAGE_NAME}:$IMAGE_TAG"

Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Tagging image for Harbor..." -ForegroundColor Cyan
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" $HARBOR_IMAGE

Write-Host "Pushing image to Harbor registry..." -ForegroundColor Cyan
Write-Host "Note: You may need to login to Harbor first:" -ForegroundColor Yellow
Write-Host "  docker login $HARBOR_REGISTRY" -ForegroundColor Yellow
docker push $HARBOR_IMAGE

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker push failed!" -ForegroundColor Red
    Write-Host "Make sure you're logged in: docker login $HARBOR_REGISTRY" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nImage successfully pushed to Harbor!" -ForegroundColor Green
Write-Host "Image: $HARBOR_IMAGE" -ForegroundColor Green

Write-Host "`nApplying Kubernetes deployment..." -ForegroundColor Cyan
kubectl apply -f k8s/deployment-gpu.yaml
kubectl apply -f k8s/service-gpu.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDeployment successful!" -ForegroundColor Green
    Write-Host "Check status with: kubectl get pods -l app=dataloader-service-gpu" -ForegroundColor Cyan
} else {
    Write-Host "`nDeployment failed!" -ForegroundColor Red
    exit 1
}
