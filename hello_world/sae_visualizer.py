"""
Complete SAE Visualization Suite
Generates comprehensive before/after training visualizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from collections import defaultdict
import copy
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# TRAINING LOGGER
# ============================================================================

class SAETrainingLogger:
    """Log metrics during training"""

    def __init__(self):
        self.metrics = {
            'step': [],
            'total_loss': [],
            'recon_loss': [],
            'sparsity_loss': [],
            'l0': [],
            'l1': [],
        }

    def log(self, step, total_loss, recon_loss, sparsity_loss, l0, l1):
        """Log a training step"""
        self.metrics['step'].append(step)
        self.metrics['total_loss'].append(total_loss)
        self.metrics['recon_loss'].append(recon_loss)
        self.metrics['sparsity_loss'].append(sparsity_loss)
        self.metrics['l0'].append(l0)
        self.metrics['l1'].append(l1)

    def get_metrics(self):
        """Get all logged metrics"""
        return self.metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(logger, save_path='outputs/1_training_curves.png'):
    """Plot training dynamics"""
    print("📊 Generating training curves...")

    metrics = logger.get_metrics()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SAE Training Dynamics', fontsize=16, fontweight='bold')

    # Total loss
    axes[0, 0].plot(metrics['step'], metrics['total_loss'], linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Training Step', fontsize=11)
    axes[0, 0].set_ylabel('Total Loss', fontsize=11)
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(metrics['step'], metrics['recon_loss'], linewidth=2, color='#06A77D')
    axes[0, 1].set_xlabel('Training Step', fontsize=11)
    axes[0, 1].set_ylabel('Reconstruction Loss (MSE)', fontsize=11)
    axes[0, 1].set_title('Reconstruction Quality', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Sparsity loss
    axes[0, 2].plot(metrics['step'], metrics['sparsity_loss'], linewidth=2, color='#D45B5B')
    axes[0, 2].set_xlabel('Training Step', fontsize=11)
    axes[0, 2].set_ylabel('Sparsity Loss (L1)', fontsize=11)
    axes[0, 2].set_title('Sparsity Penalty', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # L0 (active features)
    axes[1, 0].plot(metrics['step'], metrics['l0'], linewidth=2, color='#A23B72')
    axes[1, 0].set_xlabel('Training Step', fontsize=11)
    axes[1, 0].set_ylabel('L0 (Avg Active Features)', fontsize=11)
    axes[1, 0].set_title('Sparsity (L0)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # L1
    axes[1, 1].plot(metrics['step'], metrics['l1'], linewidth=2, color='#F18F01')
    axes[1, 1].set_xlabel('Training Step', fontsize=11)
    axes[1, 1].set_ylabel('L1 (Mean Activation)', fontsize=11)
    axes[1, 1].set_title('Feature Activation Strength', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Loss breakdown (stacked)
    axes[1, 2].fill_between(metrics['step'], 0, metrics['recon_loss'],
                            alpha=0.5, label='Reconstruction', color='#06A77D')
    axes[1, 2].fill_between(metrics['step'], metrics['recon_loss'],
                            np.array(metrics['recon_loss']) + np.array(metrics['sparsity_loss']),
                            alpha=0.5, label='Sparsity', color='#D45B5B')
    axes[1, 2].set_xlabel('Training Step', fontsize=11)
    axes[1, 2].set_ylabel('Loss', fontsize=11)
    axes[1, 2].set_title('Loss Components (Stacked)', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_weight_comparison(sae_before, sae_after, save_path='outputs/2_weight_comparison.png'):
    """Compare decoder weights before and after training"""
    print("📊 Generating weight comparison...")

    W_before = sae_before.decoder.weight.data.cpu().numpy()
    W_after = sae_after.decoder.weight.data.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Decoder Weight Evolution', fontsize=16, fontweight='bold')

    # 1. Weight distribution
    axes[0, 0].hist(W_before.flatten(), bins=50, alpha=0.6, label='Before',
                    color='blue', density=True)
    axes[0, 0].hist(W_after.flatten(), bins=50, alpha=0.6, label='After',
                    color='red', density=True)
    axes[0, 0].set_xlabel('Weight Value', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('Weight Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Column norms (feature magnitudes)
    norms_before = np.linalg.norm(W_before, axis=0)
    norms_after = np.linalg.norm(W_after, axis=0)

    axes[0, 1].hist(norms_before, bins=50, alpha=0.6, label='Before', color='blue')
    axes[0, 1].hist(norms_after, bins=50, alpha=0.6, label='After', color='red')
    axes[0, 1].set_xlabel('Feature Norm', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Feature Magnitude Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Norm change
    norm_change = norms_after - norms_before
    axes[0, 2].hist(norm_change, bins=50, edgecolor='black', color='green', alpha=0.7)
    axes[0, 2].set_xlabel('Norm Change', fontsize=11)
    axes[0, 2].set_ylabel('Frequency', fontsize=11)
    axes[0, 2].set_title('Feature Norm Change Distribution', fontweight='bold')
    axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. PCA visualization
    pca = PCA(n_components=2)
    sample_size = min(500, W_before.shape[1])
    idx = np.random.choice(W_before.shape[1], sample_size, replace=False)

    W_before_sample = W_before[:, idx].T
    W_after_sample = W_after[:, idx].T

    W_before_2d = pca.fit_transform(W_before_sample)
    W_after_2d = pca.transform(W_after_sample)

    axes[1, 0].scatter(W_before_2d[:, 0], W_before_2d[:, 1],
                       alpha=0.4, s=20, label='Before', color='blue')
    axes[1, 0].scatter(W_after_2d[:, 0], W_after_2d[:, 1],
                       alpha=0.4, s=20, label='After', color='red')
    axes[1, 0].set_xlabel('PC1', fontsize=11)
    axes[1, 0].set_ylabel('PC2', fontsize=11)
    axes[1, 0].set_title('Feature Space (PCA)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Movement vectors
    for i in range(min(50, sample_size)):
        axes[1, 0].arrow(W_before_2d[i, 0], W_before_2d[i, 1],
                         W_after_2d[i, 0] - W_before_2d[i, 0],
                         W_after_2d[i, 1] - W_before_2d[i, 1],
                         alpha=0.3, width=0.05, head_width=0.2,
                         color='gray', length_includes_head=True)

    # 6. Dead features
    dead_before = np.sum(norms_before < 0.01)
    dead_after = np.sum(norms_after < 0.01)
    alive_before = W_before.shape[1] - dead_before
    alive_after = W_after.shape[1] - dead_after

    x = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x - width / 2, [alive_before, alive_after], width,
                   label='Alive', color='green', alpha=0.7)
    axes[1, 1].bar(x + width / 2, [dead_before, dead_after], width,
                   label='Dead', color='red', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Before', 'After'])
    axes[1, 1].set_ylabel('Number of Features', fontsize=11)
    axes[1, 1].set_title('Dead vs Alive Features', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add text labels
    for i, (alive, dead) in enumerate([(alive_before, dead_before),
                                       (alive_after, dead_after)]):
        axes[1, 1].text(i - width / 2, alive + 10, str(alive),
                        ha='center', va='bottom', fontweight='bold')
        axes[1, 1].text(i + width / 2, dead + 10, str(dead),
                        ha='center', va='bottom', fontweight='bold')

    # 7. Top changed features
    top_indices = np.argsort(np.abs(norm_change))[-20:]
    top_changes = norm_change[top_indices]

    colors = ['green' if x > 0 else 'red' for x in top_changes]
    axes[1, 2].barh(range(20), top_changes, color=colors, alpha=0.7)
    axes[1, 2].set_xlabel('Norm Change', fontsize=11)
    axes[1, 2].set_ylabel('Feature ID', fontsize=11)
    axes[1, 2].set_title('Top 20 Features by Norm Change', fontweight='bold')
    axes[1, 2].set_yticks(range(20))
    axes[1, 2].set_yticklabels([f"{idx}" for idx in top_indices], fontsize=8)
    axes[1, 2].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1, 2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_activation_analysis(sae, activations, save_path='outputs/3_activation_analysis.png'):
    """Analyze feature activation patterns"""
    print("📊 Generating activation analysis...")

    sae.eval()
    with torch.no_grad():
        n_samples = min(200, len(activations))
        sample_acts = activations[:n_samples]
        features = sae.encode(sample_acts)

    features_np = features.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Activation Patterns', fontsize=16, fontweight='bold')

    # 1. Activation heatmap
    n_display = min(100, features_np.shape[0])
    sns.heatmap(features_np[:n_display, :150].T, ax=axes[0, 0],
                cmap='viridis', cbar_kws={'label': 'Activation'})
    axes[0, 0].set_xlabel('Sample ID', fontsize=11)
    axes[0, 0].set_ylabel('Feature ID', fontsize=11)
    axes[0, 0].set_title(f'Activation Heatmap (first {n_display} samples)', fontweight='bold')

    # 2. L0 histogram
    l0_per_sample = (features_np > 0).sum(axis=1)
    axes[0, 1].hist(l0_per_sample, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
    axes[0, 1].set_xlabel('Number of Active Features (L0)', fontsize=11)
    axes[0, 1].set_ylabel('Number of Samples', fontsize=11)
    axes[0, 1].set_title(f'L0 Distribution (mean: {l0_per_sample.mean():.1f})', fontweight='bold')
    axes[0, 1].axvline(l0_per_sample.mean(), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {l0_per_sample.mean():.1f}')
    axes[0, 1].axvline(np.median(l0_per_sample), color='green',
                       linestyle='--', linewidth=2, label=f'Median: {np.median(l0_per_sample):.1f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Activation strength distribution
    active_values = features_np[features_np > 0]
    axes[0, 2].hist(active_values, bins=50, edgecolor='black', color='lightcoral', alpha=0.7)
    axes[0, 2].set_xlabel('Activation Value', fontsize=11)
    axes[0, 2].set_ylabel('Frequency', fontsize=11)
    axes[0, 2].set_title('Distribution of Active Feature Values', fontweight='bold')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Feature frequency
    feature_freq = (features_np > 0).mean(axis=0)
    axes[1, 0].hist(feature_freq, bins=50, edgecolor='black', color='mediumpurple', alpha=0.7)
    axes[1, 0].set_xlabel('Activation Frequency', fontsize=11)
    axes[1, 0].set_ylabel('Number of Features', fontsize=11)
    axes[1, 0].set_title('Feature Firing Frequency', fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 5. Top active features
    top_n = 20
    top_features = np.argsort(feature_freq)[-top_n:]
    axes[1, 1].barh(range(top_n), feature_freq[top_features],
                    color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Activation Frequency', fontsize=11)
    axes[1, 1].set_ylabel('Feature ID', fontsize=11)
    axes[1, 1].set_title(f'Top {top_n} Most Active Features', fontweight='bold')
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([f"F{idx}" for idx in top_features], fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    # 6. Co-activation analysis (sample)
    sample_size = min(100, features_np.shape[1])
    sample_idx = np.random.choice(features_np.shape[1], sample_size, replace=False)
    features_sample = (features_np[:, sample_idx] > 0).astype(float)
    coactivation = features_sample.T @ features_sample / features_sample.shape[0]

    sns.heatmap(coactivation, ax=axes[1, 2], cmap='YlOrRd',
                xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    axes[1, 2].set_title(f'Co-activation Matrix ({sample_size} random features)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_reconstruction_quality(sae, activations, n_samples=8, save_path='outputs/4_reconstruction_quality.png'):
    """Visualize reconstruction quality"""
    print("📊 Generating reconstruction quality analysis...")

    sae.eval()

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 2.5 * n_samples))
    fig.suptitle('Reconstruction Quality Examples', fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n_samples):
            original = activations[i:i + 1]
            reconstructed, _, features = sae(original)

            original_np = original.cpu().numpy().flatten()
            reconstructed_np = reconstructed.cpu().numpy().flatten()
            features_np = features.cpu().numpy().flatten()
            error = original_np - reconstructed_np

            # 1. Original activation
            axes[i, 0].bar(range(len(original_np)), original_np, width=1.0,
                           color='blue', alpha=0.7, edgecolor='none')
            axes[i, 0].set_xlim(0, len(original_np))
            axes[i, 0].set_ylabel('Value', fontsize=10)
            if i == 0:
                axes[i, 0].set_title('Original Activation', fontweight='bold')
            if i == n_samples - 1:
                axes[i, 0].set_xlabel('Dimension', fontsize=10)
            axes[i, 0].grid(True, alpha=0.3, axis='y')

            # 2. Reconstructed activation
            axes[i, 1].bar(range(len(reconstructed_np)), reconstructed_np, width=1.0,
                           color='red', alpha=0.7, edgecolor='none')
            axes[i, 1].set_xlim(0, len(reconstructed_np))
            if i == 0:
                axes[i, 1].set_title('Reconstructed', fontweight='bold')
            if i == n_samples - 1:
                axes[i, 1].set_xlabel('Dimension', fontsize=10)
            axes[i, 1].grid(True, alpha=0.3, axis='y')

            # Metrics
            mse = np.mean(error ** 2)
            cos_sim = np.dot(original_np, reconstructed_np) / \
                      (np.linalg.norm(original_np) * np.linalg.norm(reconstructed_np) + 1e-8)
            l0 = (features_np > 0).sum()

            axes[i, 1].text(0.98, 0.95,
                            f'MSE: {mse:.4f}\nCos: {cos_sim:.4f}\nL0: {l0}',
                            transform=axes[i, 1].transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            # 3. Error
            axes[i, 2].bar(range(len(error)), error, width=1.0,
                           color='purple', alpha=0.7, edgecolor='none')
            axes[i, 2].set_xlim(0, len(error))
            axes[i, 2].axhline(0, color='black', linestyle='-', linewidth=0.5)
            if i == 0:
                axes[i, 2].set_title('Reconstruction Error', fontweight='bold')
            if i == n_samples - 1:
                axes[i, 2].set_xlabel('Dimension', fontsize=10)
            axes[i, 2].grid(True, alpha=0.3, axis='y')

            # 4. Sparse features
            active_features = features_np[features_np > 0]
            active_indices = np.where(features_np > 0)[0]

            if len(active_features) > 0:
                axes[i, 3].bar(active_indices, active_features, width=3.0,
                               color='green', alpha=0.7, edgecolor='black')
                axes[i, 3].set_xlim(0, len(features_np))
                if i == 0:
                    axes[i, 3].set_title(f'Active Features', fontweight='bold')
                if i == n_samples - 1:
                    axes[i, 3].set_xlabel('Feature ID', fontsize=10)
                axes[i, 3].set_ylabel('Activation', fontsize=10)
                axes[i, 3].grid(True, alpha=0.3, axis='y')
            else:
                axes[i, 3].text(0.5, 0.5, 'No active\nfeatures',
                                ha='center', va='center',
                                transform=axes[i, 3].transAxes,
                                fontsize=11, color='red')
                if i == 0:
                    axes[i, 3].set_title('Active Features', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_superposition_analysis(sae, save_path='outputs/5_superposition_analysis.png'):
    """Analyze feature superposition"""
    print("📊 Generating superposition analysis...")

    # Get decoder weights and compute similarity
    D = sae.decoder.weight.data.cpu().numpy()
    D_norm = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
    S = np.abs(D_norm.T @ D_norm)
    np.fill_diagonal(S, 0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Superposition Analysis', fontsize=16, fontweight='bold')

    # 1. Similarity matrix (sample)
    sample_size = 150
    idx = np.random.choice(S.shape[0], sample_size, replace=False)
    S_sample = S[idx][:, idx]

    im = axes[0, 0].imshow(S_sample, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    axes[0, 0].set_title('Feature Similarity Matrix\n(150 random features)', fontweight='bold')
    axes[0, 0].set_xlabel('Feature ID', fontsize=11)
    axes[0, 0].set_ylabel('Feature ID', fontsize=11)
    plt.colorbar(im, ax=axes[0, 0], label='|Cosine Similarity|')

    # 2. Similarity distribution
    S_flat = S[np.triu_indices_from(S, k=1)]
    axes[0, 1].hist(S_flat, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
    axes[0, 1].set_xlabel('|Cosine Similarity|', fontsize=11)
    axes[0, 1].set_ylabel('Number of Feature Pairs', fontsize=11)
    axes[0, 1].set_title('Distribution of Feature Similarities', fontweight='bold')
    axes[0, 1].axvline(S_flat.mean(), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {S_flat.mean():.3f}')
    axes[0, 1].axvline(np.median(S_flat), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(S_flat):.3f}')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Percentiles
    percentiles = [50, 75, 90, 95, 99]
    perc_values = [np.percentile(S_flat, p) for p in percentiles]

    axes[0, 2].bar(range(len(percentiles)), perc_values,
                   color='coral', alpha=0.7, edgecolor='black')
    axes[0, 2].set_xticks(range(len(percentiles)))
    axes[0, 2].set_xticklabels([f'{p}th' for p in percentiles])
    axes[0, 2].set_ylabel('Similarity Value', fontsize=11)
    axes[0, 2].set_title('Similarity Percentiles', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    for i, val in enumerate(perc_values):
        axes[0, 2].text(i, val + 0.02, f'{val:.3f}',
                        ha='center', va='bottom', fontweight='bold')

    # 4. Degree distribution at multiple thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))

    for tau, color in zip(thresholds, colors):
        degrees = (S > tau).sum(axis=1)
        axes[1, 0].hist(degrees, bins=30, alpha=0.5, label=f'τ={tau}',
                        color=color, edgecolor='black')

    axes[1, 0].set_xlabel('Degree (# of neighbors)', fontsize=11)
    axes[1, 0].set_ylabel('Number of Features', fontsize=11)
    axes[1, 0].set_title('Degree Distribution at Various Thresholds', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Graph properties vs threshold
    tau_range = np.linspace(0.05, 0.9, 50)
    densities = []
    n_edges = []
    mean_degrees = []

    for tau in tau_range:
        adj_matrix = (S > tau).astype(int)
        edges = adj_matrix.sum() / 2
        density = edges / (S.shape[0] * (S.shape[0] - 1) / 2)
        mean_degree = adj_matrix.sum(axis=1).mean()

        densities.append(density)
        n_edges.append(edges)
        mean_degrees.append(mean_degree)

    ax1 = axes[1, 1]
    ax2 = ax1.twinx()

    line1 = ax1.plot(tau_range, densities, 'b-', linewidth=2, label='Density')
    line2 = ax2.plot(tau_range, mean_degrees, 'r-', linewidth=2, label='Mean Degree')

    ax1.set_xlabel('Threshold τ', fontsize=11)
    ax1.set_ylabel('Graph Density', color='b', fontsize=11)
    ax2.set_ylabel('Mean Degree', color='r', fontsize=11)
    ax1.set_title('Graph Properties vs Threshold', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # 6. Hub features (high degree nodes)
    tau_analysis = 0.3
    degrees = (S > tau_analysis).sum(axis=1)
    top_hubs = np.argsort(degrees)[-20:]

    axes[1, 2].barh(range(20), degrees[top_hubs],
                    color='orange', alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Degree', fontsize=11)
    axes[1, 2].set_ylabel('Feature ID', fontsize=11)
    axes[1, 2].set_title(f'Top 20 Hub Features (τ={tau_analysis})', fontweight='bold')
    axes[1, 2].set_yticks(range(20))
    axes[1, 2].set_yticklabels([f"F{idx}" for idx in top_hubs], fontsize=8)
    axes[1, 2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def plot_summary_dashboard(sae_before, sae_after, logger, activations,
                           save_path='outputs/6_summary_dashboard.png'):
    """Create comprehensive summary dashboard"""
    print("📊 Generating summary dashboard...")

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    fig.suptitle('SAE Training Summary Dashboard', fontsize=18, fontweight='bold')

    # Get data
    W_before = sae_before.decoder.weight.data.cpu().numpy()
    W_after = sae_after.decoder.weight.data.cpu().numpy()
    norms_before = np.linalg.norm(W_before, axis=0)
    norms_after = np.linalg.norm(W_after, axis=0)

    sae_after.eval()
    with torch.no_grad():
        n_samples = min(200, len(activations))
        sample_acts = activations[:n_samples]
        features = sae_after.encode(sample_acts)
        x_recon, _, _ = sae_after(sample_acts)

    features_np = features.cpu().numpy()
    sample_acts_np = sample_acts.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()

    # 1. Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = logger.get_metrics()
    ax1.plot(metrics['step'], metrics['total_loss'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Step', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. L0 evolution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics['step'], metrics['l0'], linewidth=2, color='#A23B72')
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('L0', fontsize=10)
    ax2.set_title('Sparsity (L0)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Dead features
    ax3 = fig.add_subplot(gs[0, 2])
    dead_before = (norms_before < 0.01).sum()
    dead_after = (norms_after < 0.01).sum()
    ax3.bar(['Before', 'After'], [dead_before, dead_after],
            color=['blue', 'red'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Dead Features', fontsize=10)
    ax3.set_title('Dead Features', fontweight='bold')
    ax3.text(0, dead_before + 10, str(dead_before), ha='center', fontweight='bold')
    ax3.text(1, dead_after + 10, str(dead_after), ha='center', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Final metrics
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')

    l0_final = metrics['l0'][-1]
    recon_final = metrics['recon_loss'][-1]
    mse = np.mean((sample_acts_np - x_recon_np) ** 2)
    cos_sim = np.mean([
        np.dot(sample_acts_np[i], x_recon_np[i]) /
        (np.linalg.norm(sample_acts_np[i]) * np.linalg.norm(x_recon_np[i]) + 1e-8)
        for i in range(len(sample_acts_np))
    ])

    metrics_text = f"""
    Final Metrics:

    L0 (sparsity): {l0_final:.1f}
    Dead features: {dead_after} ({100 * dead_after / W_after.shape[1]:.1f}%)

    MSE: {mse:.4f}
    Cosine sim: {cos_sim:.4f}

    Total features: {W_after.shape[1]}
    Active features: {W_after.shape[1] - dead_after}
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 5. Weight norm distribution
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(norms_before, bins=30, alpha=0.6, label='Before', color='blue')
    ax5.hist(norms_after, bins=30, alpha=0.6, label='After', color='red')
    ax5.set_xlabel('Feature Norm', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Feature Magnitudes', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. L0 distribution
    ax6 = fig.add_subplot(gs[1, 1])
    l0_per_sample = (features_np > 0).sum(axis=1)
    ax6.hist(l0_per_sample, bins=30, edgecolor='black', color='skyblue', alpha=0.7)
    ax6.set_xlabel('L0 per Sample', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title(f'L0 Distribution (mean={l0_per_sample.mean():.1f})', fontweight='bold')
    ax6.axvline(l0_per_sample.mean(), color='red', linestyle='--', linewidth=2)
    ax6.grid(True, alpha=0.3)

    # 7. Feature frequency
    ax7 = fig.add_subplot(gs[1, 2])
    feature_freq = (features_np > 0).mean(axis=0)
    ax7.hist(feature_freq, bins=30, edgecolor='black', color='lightcoral', alpha=0.7)
    ax7.set_xlabel('Activation Frequency', fontsize=10)
    ax7.set_ylabel('# Features', fontsize=10)
    ax7.set_title('Feature Usage', fontweight='bold')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)

    # 8. Activation heatmap
    ax8 = fig.add_subplot(gs[1, 3])
    n_display = min(50, features_np.shape[0])
    im = ax8.imshow(features_np[:n_display, :100].T, cmap='viridis', aspect='auto')
    ax8.set_xlabel('Sample', fontsize=10)
    ax8.set_ylabel('Feature', fontsize=10)
    ax8.set_title('Activation Patterns', fontweight='bold')
    plt.colorbar(im, ax=ax8, label='Activation')

    # 9. Similarity matrix
    ax9 = fig.add_subplot(gs[2, 0])
    D = sae_after.decoder.weight.data.cpu().numpy()
    D_norm = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
    S = np.abs(D_norm.T @ D_norm)
    np.fill_diagonal(S, 0)

    sample_idx = np.random.choice(S.shape[0], 100, replace=False)
    S_sample = S[sample_idx][:, sample_idx]
    im = ax9.imshow(S_sample, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax9.set_title('Feature Similarity', fontweight='bold')
    plt.colorbar(im, ax=ax9)

    # 10. Similarity distribution
    ax10 = fig.add_subplot(gs[2, 1])
    S_flat = S[np.triu_indices_from(S, k=1)]
    ax10.hist(S_flat, bins=40, edgecolor='black', color='skyblue', alpha=0.7)
    ax10.set_xlabel('Similarity', fontsize=10)
    ax10.set_ylabel('Frequency', fontsize=10)
    ax10.set_title(f'Similarity Dist (mean={S_flat.mean():.3f})', fontweight='bold')
    ax10.set_yscale('log')
    ax10.grid(True, alpha=0.3)

    # 11. Degree distribution
    ax11 = fig.add_subplot(gs[2, 2])
    degrees = (S > 0.3).sum(axis=1)
    ax11.hist(degrees, bins=30, edgecolor='black', color='orange', alpha=0.7)
    ax11.set_xlabel('Degree (τ=0.3)', fontsize=10)
    ax11.set_ylabel('# Features', fontsize=10)
    ax11.set_title('Superposition Graph Degrees', fontweight='bold')
    ax11.grid(True, alpha=0.3)

    # 12. Reconstruction error
    ax12 = fig.add_subplot(gs[2, 3])
    errors = np.mean((sample_acts_np - x_recon_np) ** 2, axis=1)
    ax12.hist(errors, bins=30, edgecolor='black', color='lightgreen', alpha=0.7)
    ax12.set_xlabel('MSE per Sample', fontsize=10)
    ax12.set_ylabel('Frequency', fontsize=10)
    ax12.set_title(f'Reconstruction Error (mean={errors.mean():.4f})', fontweight='bold')
    ax12.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {save_path}")


def generate_text_report(sae_before, sae_after, logger, activations,
                         save_path='outputs/training_report.txt'):
    """Generate text summary report"""
    print("📝 Generating text report...")

    # Collect metrics
    W_before = sae_before.decoder.weight.data.cpu().numpy()
    W_after = sae_after.decoder.weight.data.cpu().numpy()
    norms_before = np.linalg.norm(W_before, axis=0)
    norms_after = np.linalg.norm(W_after, axis=0)

    sae_after.eval()
    with torch.no_grad():
        n_samples = min(200, len(activations))
        sample_acts = activations[:n_samples]
        features = sae_after.encode(sample_acts)
        x_recon, _, _ = sae_after(sample_acts)

    features_np = features.cpu().numpy()
    sample_acts_np = sample_acts.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()

    # Compute superposition metrics
    D = sae_after.decoder.weight.data.cpu().numpy()
    D_norm = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-8)
    S = np.abs(D_norm.T @ D_norm)
    np.fill_diagonal(S, 0)
    S_flat = S[np.triu_indices_from(S, k=1)]

    # Build report
    report = f"""
{'=' * 70}
SAE TRAINING REPORT
{'=' * 70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 70}
CONFIGURATION
{'=' * 70}
Input dimension (d_model):     {sae_after.d_model}
Feature dimension (d_sae):     {sae_after.d_sae}
Sparsity coefficient:          {sae_after.sparsity_coef}
Expansion ratio:               {sae_after.d_sae / sae_after.d_model:.2f}x

Training steps:                {len(logger.metrics['step'])}
Final loss:                    {logger.metrics['total_loss'][-1]:.4f}

{'=' * 70}
SPARSITY METRICS
{'=' * 70}
L0 (avg active features):      {features_np[features_np > 0].shape[0] / len(features_np):.1f} / {sae_after.d_sae}
L0 percentage:                 {100 * (features_np > 0).sum() / (features_np.shape[0] * features_np.shape[1]):.2f}%
L1 (mean activation):          {features_np[features_np > 0].mean():.4f}

Sparsity per sample:
  Mean:                        {(features_np > 0).sum(axis=1).mean():.1f}
  Median:                      {np.median((features_np > 0).sum(axis=1)):.1f}
  Std:                         {(features_np > 0).sum(axis=1).std():.1f}
  Min:                         {(features_np > 0).sum(axis=1).min():.0f}
  Max:                         {(features_np > 0).sum(axis=1).max():.0f}

{'=' * 70}
RECONSTRUCTION QUALITY
{'=' * 70}
MSE:                           {np.mean((sample_acts_np - x_recon_np) ** 2):.4f}
RMSE:                          {np.sqrt(np.mean((sample_acts_np - x_recon_np) ** 2)):.4f}

Cosine similarity:
  Mean:                        {np.mean([np.dot(sample_acts_np[i], x_recon_np[i]) / (np.linalg.norm(sample_acts_np[i]) * np.linalg.norm(x_recon_np[i]) + 1e-8) for i in range(len(sample_acts_np))]):.4f}

{'=' * 70}
FEATURE STATISTICS
{'=' * 70}
Dead features (norm < 0.01):   {(norms_after < 0.01).sum()} / {len(norms_after)} ({100 * (norms_after < 0.01).sum() / len(norms_after):.1f}%)
Active features:               {(norms_after >= 0.01).sum()} / {len(norms_after)} ({100 * (norms_after >= 0.01).sum() / len(norms_after):.1f}%)

Feature activation frequency:
  Mean:                        {(features_np > 0).mean(axis=0).mean():.4f}
  Median:                      {np.median((features_np > 0).mean(axis=0)):.4f}
  Max:                         {(features_np > 0).mean(axis=0).max():.4f}

Never-active features:         {((features_np > 0).mean(axis=0) == 0).sum()}
Always-active features:        {((features_np > 0).mean(axis=0) > 0.99).sum()}

{'=' * 70}
SUPERPOSITION ANALYSIS
{'=' * 70}
Feature similarity statistics:
  Mean:                        {S_flat.mean():.4f}
  Median:                      {np.median(S_flat):.4f}
  Std:                         {S_flat.std():.4f}
  Max:                         {S_flat.max():.4f}

Percentiles:
  50th:                        {np.percentile(S_flat, 50):.4f}
  75th:                        {np.percentile(S_flat, 75):.4f}
  90th:                        {np.percentile(S_flat, 90):.4f}
  95th:                        {np.percentile(S_flat, 95):.4f}
  99th:                        {np.percentile(S_flat, 99):.4f}

Graph properties at τ=0.3:
  Edges:                       {(S > 0.3).sum() / 2:.0f}
  Density:                     {(S > 0.3).sum() / (S.shape[0] * (S.shape[0] - 1)):.6f}
  Mean degree:                 {(S > 0.3).sum(axis=1).mean():.2f}
  Max degree:                  {(S > 0.3).sum(axis=1).max():.0f}

{'=' * 70}
TRAINING DYNAMICS
{'=' * 70}
Initial loss:                  {logger.metrics['total_loss'][0]:.4f}
Final loss:                    {logger.metrics['total_loss'][-1]:.4f}
Loss reduction:                {100 * (1 - logger.metrics['total_loss'][-1] / logger.metrics['total_loss'][0]):.1f}%

Initial L0:                    {logger.metrics['l0'][0]:.1f}
Final L0:                      {logger.metrics['l0'][-1]:.1f}
L0 reduction:                  {100 * (1 - logger.metrics['l0'][-1] / logger.metrics['l0'][0]):.1f}%

{'=' * 70}
QUALITY ASSESSMENT
{'=' * 70}
"""

    # Add quality grades
    mse = np.mean((sample_acts_np - x_recon_np) ** 2)
    l0 = (features_np > 0).sum(axis=1).mean()
    cos_sim = np.mean([np.dot(sample_acts_np[i], x_recon_np[i]) / (
                np.linalg.norm(sample_acts_np[i]) * np.linalg.norm(x_recon_np[i]) + 1e-8) for i in
                       range(len(sample_acts_np))])
    dead_pct = 100 * (norms_after < 0.01).sum() / len(norms_after)

    if mse < 0.3:
        recon_grade = "✅ EXCELLENT"
    elif mse < 0.5:
        recon_grade = "⚠️  GOOD"
    else:
        recon_grade = "❌ NEEDS IMPROVEMENT"

    if l0 < 50:
        sparsity_grade = "✅ EXCELLENT"
    elif l0 < 100:
        sparsity_grade = "⚠️  GOOD"
    else:
        sparsity_grade = "❌ TOO DENSE"

    if cos_sim > 0.95:
        direction_grade = "✅ EXCELLENT"
    elif cos_sim > 0.85:
        direction_grade = "⚠️  GOOD"
    else:
        direction_grade = "❌ POOR"

    if dead_pct < 30:
        utilization_grade = "✅ EXCELLENT"
    elif dead_pct < 50:
        utilization_grade = "⚠️  OK"
    else:
        utilization_grade = "❌ POOR"

    report += f"""
Reconstruction (MSE):          {recon_grade}
Sparsity (L0):                 {sparsity_grade}
Direction preservation:        {direction_grade}
Feature utilization:           {utilization_grade}

{'=' * 70}
RECOMMENDATIONS
{'=' * 70}
"""

    if mse > 0.5:
        report += "• Reconstruction quality is poor. Consider:\n"
        report += "  - Training for more steps\n"
        report += "  - Reducing sparsity coefficient\n"
        report += "  - Increasing SAE capacity\n\n"

    if l0 > 100:
        report += "• Features are not sparse enough. Consider:\n"
        report += "  - Increasing sparsity coefficient\n"
        report += "  - Training for more steps\n\n"

    if dead_pct > 50:
        report += "• Too many dead features. Consider:\n"
        report += "  - Reducing SAE capacity\n"
        report += "  - Better weight initialization\n"
        report += "  - Using more training data\n\n"

    if l0 < 50 and mse < 0.3 and cos_sim > 0.95 and dead_pct < 30:
        report += "• ✅ SAE training successful! Ready for downstream analysis.\n\n"

    report += f"""
{'=' * 70}
NEXT STEPS FOR RESEARCH
{'=' * 70}
1. Feature interpretation: Identify what each feature detects
2. Build superposition graphs at multiple thresholds
3. Select features for unlearning experiments
4. Test correlation between degree and unlearning difficulty

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"  ✓ Saved to {save_path}")
    print("\n" + "=" * 70)
    print(report)


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def visualize_sae_training(sae_before, sae_after, logger, activations, output_dir='outputs'):
    """
    Generate all visualizations

    Args:
        sae_before: SAE before training
        sae_after: SAE after training
        logger: SAETrainingLogger with metrics
        activations: torch.Tensor of activations used for training
        output_dir: directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING SAE VISUALIZATIONS")
    print("=" * 70 + "\n")

    # Generate all plots
    plot_training_curves(logger, f'{output_dir}/1_training_curves.png')
    plot_weight_comparison(sae_before, sae_after, f'{output_dir}/2_weight_comparison.png')
    plot_activation_analysis(sae_after, activations, f'{output_dir}/3_activation_analysis.png')
    plot_reconstruction_quality(sae_after, activations, n_samples=8)
    plot_superposition_analysis(sae_after, f'{output_dir}/5_superposition_analysis.png')
    plot_summary_dashboard(sae_before, sae_after, logger, activations,
                           f'{output_dir}/6_summary_dashboard.png')
    generate_text_report(sae_before, sae_after, logger, activations,
                         f'{output_dir}/training_report.txt')

    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. 1_training_curves.png - Training dynamics")
    print("  2. 2_weight_comparison.png - Weight evolution")
    print("  3. 3_activation_analysis.png - Feature activation patterns")
    print("  4. 4_reconstruction_quality.png - Reconstruction examples")
    print("  5. 5_superposition_analysis.png - Feature superposition")
    print("  6. 6_summary_dashboard.png - Complete overview")
    print("  7. training_report.txt - Detailed text report")
    print("\n" + "=" * 70 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    SAE Visualization Suite

    Usage:

    from sae_visualizer import SAETrainingLogger, visualize_sae_training
    import copy

    # 1. Create logger
    logger = SAETrainingLogger()

    # 2. Save untrained SAE
    sae_before = copy.deepcopy(sae)

    # 3. Train with logging
    for step in range(n_steps):
        # ... training code ...

        if step % 10 == 0:
            logger.log(step, loss.item(), l_recon, l_sparsity, l0, l1)

    # 4. Generate all visualizations
    visualize_sae_training(sae_before, sae, logger, activations)
    """)