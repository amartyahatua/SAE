"""
SAE Hello World: Train a Sparse Autoencoder on GPT-2
This is the absolute minimal working example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Model settings
MODEL_NAME = "gpt2"  # Smallest GPT-2 (117M params)
LAYER_ID = 6  # Which layer to hook (middle layer)
D_MODEL = 768  # GPT-2 hidden dimension

# SAE settings
D_SAE = 2048  # SAE hidden dimension (2.67x expansion for hello world)
SPARSITY_COEF = 0.01  # L1 penalty strength

# Training settings
N_EXAMPLES = 1000  # Very small for hello world
BATCH_SIZE = 8
LR = 1e-3
N_STEPS = 100  # Just 100 steps to see it work


# ============================================================================
# SPARSE AUTOENCODER DEFINITION
# ============================================================================

class SparseAutoencoder(nn.Module):
    """
    Simple Sparse Autoencoder

    Architecture:
        input (d_model) -> encoder -> ReLU -> hidden (d_sae) -> decoder -> reconstruction (d_model)

    Loss = MSE(reconstruction, input) + sparsity_coef * L1(hidden)
    """

    def __init__(self, d_model, d_sae, sparsity_coef=0.01):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.sparsity_coef = sparsity_coef

        # Encoder: d_model -> d_sae
        self.encoder = nn.Linear(d_model, d_sae, bias=True)

        # Decoder: d_sae -> d_model
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Initialize decoder columns to unit norm (important!)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x):
        """Encode input to sparse representation"""
        return F.relu(self.encoder(x))

    def decode(self, h):
        """Decode sparse representation back to input space"""
        return self.decoder(h)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: [batch, d_model] - activations from transformer

        Returns:
            x_recon: [batch, d_model] - reconstructed activations
            loss: scalar - total loss
            h: [batch, d_sae] - sparse hidden activations
        """
        # Encode
        h = self.encode(x)  # [batch, d_sae]

        # Decode
        x_recon = self.decode(h)  # [batch, d_model]

        # Compute losses
        l_recon = F.mse_loss(x_recon, x)
        l_sparsity = h.abs().mean()  # L1 penalty

        loss = l_recon + self.sparsity_coef * l_sparsity

        return x_recon, loss, h

    def get_feature_activations(self, x):
        """Just return the sparse features (for analysis)"""
        return self.encode(x)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_activations(model, tokenizer, layer_id, n_examples=1000, max_length=128):
    """
    Collect activations from a specific layer of GPT-2

    Returns:
        activations: [n_examples, d_model] tensor
    """
    print(f"Collecting {n_examples} activations from layer {layer_id}...")

    # Hardcoded sample texts (no download needed!)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Paris is the capital of France and a major European city.",
        "Machine learning models require large amounts of training data.",
        "The weather today is sunny and warm with clear skies.",
        "Python is a popular programming language for data science.",
        "Neural networks consist of multiple layers of interconnected nodes.",
        "The movie was entertaining and well-directed by a talented filmmaker.",
        "Coffee contains caffeine which increases alertness and focus.",
        "Books provide knowledge, entertainment, and escape from reality.",
        "The ocean covers more than 70 percent of Earth's surface.",
        "Shakespeare wrote many famous plays including Hamlet and Macbeth.",
        "The human brain contains approximately 86 billion neurons.",
        "Music has the power to evoke strong emotions and memories.",
        "Democracy is a system of government where citizens have voting rights.",
        "Albert Einstein developed the theory of relativity in physics.",
        "Healthy eating involves consuming a balanced diet with vegetables.",
        "The internet has revolutionized communication and information sharing.",
        "Dogs are loyal companions and have been domesticated for thousands of years.",
        "Photography captures moments in time and preserves memories forever.",
        "Mathematics is the foundation of science, engineering, and technology.",
        "Climate change poses significant challenges to ecosystems worldwide.",
        "The Renaissance was a period of cultural rebirth in European history.",
        "Computers process information using binary code of zeros and ones.",
        "Basketball is a popular sport played by millions around the globe.",
        "The Great Wall of China is one of the most famous landmarks.",
        "Meditation can reduce stress and improve mental well-being significantly.",
        "Dinosaurs roamed the Earth millions of years before humans existed.",
        "The microscope allows scientists to observe tiny organisms and cells.",
        "Jazz music originated in New Orleans and spread throughout America.",
    ]

    # Repeat texts to get enough examples
    all_texts = []
    while len(all_texts) < n_examples:
        all_texts.extend(sample_texts)
    all_texts = all_texts[:n_examples]  # Trim to exact number

    activations = []
    model.eval()

    print(f"  Processing {n_examples} text examples...")

    with torch.no_grad():
        for i, text in enumerate(all_texts):
            # Tokenize
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(DEVICE)

            # Skip if empty (shouldn't happen with our texts, but safety check)
            if tokens['input_ids'].shape[1] == 0:
                print(f"  Warning: Empty tokens at index {i}, skipping...")
                continue

            # Get hidden states
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (n_layers+1) tensors

            # Extract layer activations
            layer_acts = hidden_states[layer_id]  # [1, seq_len, d_model]

            # Take mean over sequence (simplification for hello world)
            mean_act = layer_acts.mean(dim=1)  # [1, d_model]

            activations.append(mean_act.cpu())

            if (i + 1) % 100 == 0:
                print(f"  Collected {i + 1}/{n_examples}")

    activations = torch.cat(activations, dim=0)  # [n_examples, d_model]
    print(f"✓ Collected activations: {activations.shape}")

    return activations


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_sae(sae, activations, n_steps=100, batch_size=8, lr=1e-3):
    """
    Train the SAE on collected activations
    """
    print(f"\nTraining SAE for {n_steps} steps...")

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    sae.train()

    n_data = len(activations)

    for step in range(n_steps):
        # Sample random batch
        idx = torch.randint(0, n_data, (batch_size,))
        batch = activations[idx].to(DEVICE)

        # Forward pass
        x_recon, loss, h = sae(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Renormalize decoder weights (important for SAE training!)
        with torch.no_grad():
            sae.decoder.weight.data = F.normalize(sae.decoder.weight.data, dim=0)

        # Log
        if (step + 1) % 10 == 0:
            l0 = (h > 0).float().sum(dim=1).mean().item()  # Average # of active features
            print(f"  Step {step + 1}/{n_steps} | Loss: {loss.item():.4f} | L0: {l0:.1f}")

    print("✓ Training complete!")
    return sae


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_sae(sae, activations, n_samples=100):
    """
    Basic analysis of trained SAE
    """
    print("\n=== SAE Analysis ===")

    sae.eval()
    with torch.no_grad():
        # Sample some data
        idx = torch.randint(0, len(activations), (n_samples,))
        batch = activations[idx].to(DEVICE)

        # Get reconstructions and features
        x_recon, loss, h = sae(batch)

        # Compute metrics
        mse = F.mse_loss(x_recon, batch).item()
        l0 = (h > 0).float().sum(dim=1).mean().item()
        l1 = h.abs().mean().item()

        # Reconstruction quality
        cos_sim = F.cosine_similarity(x_recon, batch, dim=1).mean().item()

        print(f"MSE: {mse:.4f}")
        print(f"L0 (avg active features): {l0:.1f} / {sae.d_sae}")
        print(f"L1 (sparsity): {l1:.4f}")
        print(f"Cosine similarity: {cos_sim:.4f}")

        # Feature statistics
        feature_freq = (h > 0).float().mean(dim=0)  # [d_sae]
        n_dead = (feature_freq == 0).sum().item()
        print(f"Dead features: {n_dead} / {sae.d_sae} ({100 * n_dead / sae.d_sae:.1f}%)")

        # Most active features
        print("\nTop 5 most frequently active features:")
        top_features = torch.argsort(feature_freq, descending=True)[:5]
        for i, feat_id in enumerate(top_features):
            print(f"  Feature {feat_id.item()}: {feature_freq[feat_id].item():.2%} frequency")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("SAE HELLO WORLD")
    print("=" * 60)

    # 1. Load GPT-2
    print("\n[1/5] Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"✓ Loaded {MODEL_NAME}")

    # 2. Collect activations
    print(f"\n[2/5] Collecting activations from layer {LAYER_ID}...")
    activations = collect_activations(model, tokenizer, LAYER_ID, n_examples=N_EXAMPLES)

    # 3. Create SAE
    print(f"\n[3/5] Creating SAE ({D_MODEL} -> {D_SAE})...")
    sae = SparseAutoencoder(
        d_model=D_MODEL,
        d_sae=D_SAE,
        sparsity_coef=SPARSITY_COEF
    ).to(DEVICE)
    print(f"✓ SAE has {sum(p.numel() for p in sae.parameters()):,} parameters")

    # 4. Train SAE
    print(f"\n[4/5] Training SAE...")
    sae = train_sae(sae, activations, n_steps=N_STEPS, batch_size=BATCH_SIZE, lr=LR)

    # 5. Analyze
    print(f"\n[5/5] Analyzing trained SAE...")
    analyze_sae(sae, activations)

    # Save
    print(f"\n💾 Saving SAE to 'sae_hello_world.pt'...")
    torch.save({
        'sae_state_dict': sae.state_dict(),
        'config': {
            'd_model': D_MODEL,
            'd_sae': D_SAE,
            'sparsity_coef': SPARSITY_COEF,
            'layer_id': LAYER_ID,
        }
    }, 'sae_hello_world.pt')

    print("\n" + "=" * 60)
    print("✓ DONE! SAE Hello World complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()