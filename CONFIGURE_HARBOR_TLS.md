# Configure Harbor Registry TLS for microk8s Nodes

## Problem
Nodes can't pull from Harbor due to untrusted certificate:
```
tls: failed to verify certificate: x509: certificate signed by unknown authority
```

## Solution: Configure containerd to skip TLS verification

Run these commands **on the stormtrooper node** (or any worker node that needs to pull from Harbor):

### Create Harbor Registry Configuration

```bash
# Create the directory for Harbor registry configuration
sudo mkdir -p /var/snap/microk8s/current/args/certs.d/harbor.kevin.local

# Create hosts.toml to skip TLS verification
sudo tee /var/snap/microk8s/current/args/certs.d/harbor.kevin.local/hosts.toml > /dev/null <<EOF
server = "https://harbor.kevin.local"

[host."https://harbor.kevin.local"]
  capabilities = ["pull", "resolve"]
  skip_verify = true
EOF

# Verify the file was created
cat /var/snap/microk8s/current/args/certs.d/harbor.kevin.local/hosts.toml
```

### Restart containerd (if needed)

```bash
# Restart microk8s to apply the configuration
sudo systemctl restart snap.microk8s.daemon-containerd.service

# Or restart the entire microk8s service
microk8s stop
microk8s start
```

### Test Manual Pull

```bash
# Test pulling the image manually from the node
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/dataloader-service:latest
```

### Delete and Recreate Pod

```bash
# From any node with kubectl access, delete the failed pod
kubectl delete pod -l app=dataloader-service-gpu

# Watch the new pod come up
kubectl get pods -l app=dataloader-service-gpu -w
```

## Alternative: Use HTTP Instead of HTTPS

If you have control over Harbor's configuration and it's in a trusted network, you can configure Harbor to use HTTP:

**1. On stormtrooper node:**
```bash
sudo mkdir -p /var/snap/microk8s/current/args/certs.d/harbor.kevin.local

sudo tee /var/snap/microk8s/current/args/certs.d/harbor.kevin.local/hosts.toml > /dev/null <<EOF
server = "http://harbor.kevin.local"

[host."http://harbor.kevin.local"]
  capabilities = ["pull", "resolve"]
  skip_verify = true
EOF
```

**2. Update Harbor URL in deployments to use `http://`** (if Harbor is configured for HTTP)

## For All Nodes in the Cluster

Apply the same configuration to **every** worker node that needs to pull from Harbor:

```bash
# List all nodes
kubectl get nodes

# For each node, SSH and run:
for node in kevin-ubuntu stormtrooper; do
  echo "Configuring $node..."
  ssh $node "sudo mkdir -p /var/snap/microk8s/current/args/certs.d/harbor.kevin.local"
  ssh $node "sudo tee /var/snap/microk8s/current/args/certs.d/harbor.kevin.local/hosts.toml > /dev/null" <<EOF
server = "https://harbor.kevin.local"

[host."https://harbor.kevin.local"]
  capabilities = ["pull", "resolve"]
  skip_verify = true
EOF
  ssh $node "sudo systemctl restart snap.microk8s.daemon-containerd.service"
  echo "$node configured!"
done
```

## Verify Configuration

```bash
# Check if the configuration file exists
ls -la /var/snap/microk8s/current/args/certs.d/harbor.kevin.local/

# View the configuration
cat /var/snap/microk8s/current/args/certs.d/harbor.kevin.local/hosts.toml

# Test pulling an image
microk8s ctr images pull --hosts-dir /var/snap/microk8s/current/args/certs.d harbor.kevin.local/library/dataloader-service:latest
```

## Summary

The `hosts.toml` file tells containerd:
- Where the registry server is located
- What capabilities it has (pull, resolve)
- To skip TLS verification (`skip_verify = true`)

This allows pulling images from Harbor without trusting its certificate.
