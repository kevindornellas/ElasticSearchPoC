# Calico Network Plugin Troubleshooting

## Issue
Pod stuck in `ContainerCreating` with error:
```
failed to setup network for sandbox: plugin type="calico" failed (add): 
error getting ClusterInformation: connection is unauthorized: Unauthorized
```

## Root Cause
Calico CNI plugin cannot authenticate with the Calico API server, typically due to:
- Calico node pods not running properly
- RBAC permissions issues
- Calico API server connectivity problems

## Troubleshooting Steps

### 1. Check Calico Pod Status
```bash
kubectl get pods -n kube-system -l k8s-app=calico-node
kubectl get pods -n kube-system -l k8s-app=calico-kube-controllers
```

Look for pods that are not in `Running` state.

### 2. Check Calico Node Logs
```bash
kubectl logs -n kube-system -l k8s-app=calico-node --tail=100
```

### 3. Restart Calico Nodes
```bash
# Delete Calico node pods to force restart
kubectl delete pod -n kube-system -l k8s-app=calico-node

# Wait for them to restart
kubectl wait --for=condition=ready pod -n kube-system -l k8s-app=calico-node --timeout=60s
```

### 4. Check Calico CRDs and Resources
```bash
# Check if Calico CRDs exist
kubectl get crd | grep calico

# Check ClusterInformation resource
kubectl get clusterinformation -o yaml
```

### 5. For microk8s Specific

#### Check microk8s Calico Status
```bash
microk8s status
microk8s kubectl get pods -n kube-system
```

#### Disable and Re-enable Calico (if necessary)
```bash
# This will reset the Calico configuration
microk8s disable ha-cluster
microk8s enable ha-cluster
```

#### Or try restarting microk8s
```bash
microk8s stop
microk8s start
```

### 6. Check RBAC Permissions
```bash
# Verify Calico service account has proper permissions
kubectl get clusterrolebinding | grep calico
kubectl describe clusterrolebinding calico-node
```

### 7. Delete Failed Pod and Let It Recreate
```bash
kubectl delete pod dataloader-service-gpu-66f6799d65-74z87
```

After Calico is working, the pod should be recreated automatically by the deployment.

### 8. Verify Node is Ready
```bash
# Check if stormtrooper node is ready
kubectl get nodes
kubectl describe node stormtrooper
```

## Quick Fix (Most Common Solution)

```bash
# Restart Calico node pods
kubectl delete pod -n kube-system -l k8s-app=calico-node

# Wait for pods to restart
sleep 10

# Delete the stuck pod
kubectl delete pod dataloader-service-gpu-66f6799d65-74z87

# Check status
kubectl get pods -l app=dataloader-service-gpu -o wide
```

## If Problem Persists

Check if other pods can start on the stormtrooper node:
```bash
# Try creating a simple test pod on stormtrooper
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  nodeName: stormtrooper
  containers:
  - name: nginx
    image: nginx:alpine
EOF

# Check its status
kubectl get pod test-pod
kubectl describe pod test-pod

# Clean up
kubectl delete pod test-pod
```

## Alternative: Try Without Node Pinning

If the stormtrooper node continues to have issues, you can temporarily remove the node pinning:

Edit `k8s/deployment-gpu.yaml` and comment out the `nodeName` line:
```yaml
spec:
  # nodeName: stormtrooper  # Temporarily comment out
  containers:
  ...
```

This will let Kubernetes schedule the pod on any node with GPU resources.
