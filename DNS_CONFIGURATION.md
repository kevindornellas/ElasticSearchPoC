# DNS Configuration for Harbor in Kubernetes Cluster

## Problem
Nodes in the Kubernetes cluster can't resolve `harbor.kevin.local` hostname, causing image pull failures:
```
Failed to pull image: failed to resolve reference "harbor.kevin.local/library/dataloader-service:latest": 
dial tcp: lookup harbor.kevin.local on 127.0.0.53:53: server misbehaving
```

## Solutions

### Solution 1: Use IP Address (Quick Fix) ✅

Changed all deployments to use `192.168.86.148` instead of `harbor.kevin.local`.

**Already applied in:**
- [DataLoader Service/k8s/deployment-gpu.yaml](DataLoader%20Service/k8s/deployment-gpu.yaml)
- [WebUI/k8s/deployment.yaml](WebUI/k8s/deployment.yaml)

**Redeploy after fix:**
```bash
kubectl apply -f "DataLoader Service/k8s/deployment-gpu.yaml"
kubectl apply -f WebUI/k8s/deployment.yaml
```

### Solution 2: Add to /etc/hosts on All Nodes (Better for readability)

Add Harbor hostname to `/etc/hosts` on **every** Kubernetes node:

```bash
# On each worker node (including stormtrooper)
echo "192.168.86.148 harbor.kevin.local" | sudo tee -a /etc/hosts

# Verify
cat /etc/hosts | grep harbor
```

**To apply to all nodes in microk8s cluster:**
```bash
# List all nodes
kubectl get nodes -o wide

# SSH to each node and run:
for node in kevin-ubuntu stormtrooper; do
  ssh $node "echo '192.168.86.148 harbor.kevin.local' | sudo tee -a /etc/hosts"
done
```

### Solution 3: Configure CoreDNS (Cluster-wide DNS)

Add Harbor to CoreDNS ConfigMap so all pods can resolve it:

```bash
# Edit CoreDNS ConfigMap
kubectl edit configmap coredns -n kube-system
```

Add this section in the Corefile:
```
harbor.kevin.local:53 {
    hosts {
        192.168.86.148 harbor.kevin.local
        fallthrough
    }
}
```

Then restart CoreDNS:
```bash
kubectl rollout restart deployment coredns -n kube-system
```

### Solution 4: Create Kubernetes Service for Harbor (Advanced)

Create a Service/Endpoints that maps harbor.kevin.local to the IP:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: harbor-external
  namespace: default
spec:
  type: ExternalName
  externalName: 192.168.86.148
  ports:
  - port: 80
    targetPort: 80
```

## Recommended Approach

**For immediate fix:** Use IP address (Solution 1) - already applied ✅

**For long-term:** Add to /etc/hosts on all nodes (Solution 2) or configure CoreDNS (Solution 3)

## Current Status

✅ Updated to use `192.168.86.148` in all deployments
- Deployments will now pull from Harbor using IP address
- No DNS resolution required
- Works immediately without additional node configuration

## Apply Updated Deployment

```bash
cd "DataLoader Service"
kubectl delete pod -l app=dataloader-service-gpu  # Delete the stuck pod
kubectl apply -f k8s/deployment-gpu.yaml          # Reapply with IP address
kubectl get pods -l app=dataloader-service-gpu -w # Watch it come up
```
