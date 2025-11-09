# load_sae.py
import torch
from sae_hello_world import SparseAutoencoder

# Load checkpoint
checkpoint = torch.load('sae_hello_world.pt')
config = checkpoint['config']

# Recreate SAE
sae = SparseAutoencoder(
    d_model=config['d_model'],
    d_sae=config['d_sae'],
    sparsity_coef=config['sparsity_coef']
)
sae.load_state_dict(checkpoint['sae_state_dict'])
sae.eval()

print("✓ SAE loaded and ready to use!")

# Example: Get features for some activation
x = torch.randn(1, 768)  # Random activation
features = sae.encode(x)  # [1, 2048]
print(f"Active features: {(features > 0).sum().item()} / 2048")