import torch
from Layers.RKS_Layer import RKS_Layer
from Layers.Name_Pending_Layer import BIG_Fastfood_Layer as Big_FastFood
import matplotlib.pyplot as plt

# Setup
scale = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
identity = torch.eye(1024, device=device)

# Initialize layers
rks = RKS_Layer(input_dim=1024, output_dim=2048, scale=scale, device=device)
ff = Big_FastFood(input_dim=1024, output_dim=2048, scale=scale, device=device)

# Forward pass
ff_out = ff.forward(identity)
rks_out = rks.forward(identity)

# Calculate norms
ff_norm = torch.norm(ff_out, dim=1).cpu()
rks_norm = torch.norm(rks_out, dim=1).cpu()

print(torch.var(ff_norm))
print(torch.var(rks_norm))

# Plot histograms side by side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# FF histogram
axs[0].hist(ff_norm.numpy(), bins=30, edgecolor='black')
axs[0].set_title('Histogram FF Norm')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

# RKS histogram
axs[1].hist(rks_norm.numpy(), bins=30, edgecolor='black')
axs[1].set_title('Histogram RKS Norm')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

# Show plot
plt.tight_layout()
plt.show()
