import torch
from Layers.RKS_Layer import RKS_Layer
from Layers.FastFood_Layer import FastFood_Layer
import matplotlib.pyplot as plt

# Setup
scale = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
identity = torch.eye(1024, device=device)

ff_norms = []
rks_norms = []

for _ in range(50):
    # Initialize layers
    rks = RKS_Layer(input_dim=1024, output_dim=2048, scale=scale, device=device)
    ff = FastFood_Layer(input_dim=1024, output_dim=2048, scale=scale, device=device)

    # Forward pass
    ff_out = ff.forward(identity)
    rks_out = rks.forward(identity)

    # Calculate norms and store them
    ff_norms.append(torch.norm(ff_out, dim=1).cpu())
    rks_norms.append(torch.norm(rks_out, dim=1).cpu())

# Convert lists to tensors
ff_norms = torch.cat(ff_norms)  # Flatten the list into one tensor
rks_norms = torch.cat(rks_norms)

# Print variance
print("FF Norm Mean:", torch.mean(ff_norms))
print("RKS Norm Mean:", torch.mean(rks_norms))
print("\nFF Norm Var:", torch.var(ff_norms))
print("RKS Norm Var:", torch.var(rks_norms))

# Plot histograms side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# FF histogram
axs[0].hist(ff_norms.numpy(), bins=30, edgecolor='black')
axs[0].set_title('Histogram of FF Norms')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

# RKS histogram
axs[1].hist(rks_norms.numpy(), bins=30, edgecolor='black')
axs[1].set_title('Histogram of RKS Norms')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

# Show plot
plt.tight_layout()
plt.show()
