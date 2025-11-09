"""
Comprehensive Sparse Autoencoder (SAE) Analysis Tool
Combines loading, analysis, and visualization in a single program
"""

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional, Tuple
import sys


class SAEAnalyzer:
    """Complete SAE analysis with loading and visualization capabilities"""

    def __init__(
        self,
        model_name: str = "gpt2-small",
        sae_release: str = "gpt2-small-res-jb",
        sae_id: str = "blocks.8.hook_resid_pre",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the SAE analyzer

        Args:
            model_name: Name of the transformer model
            sae_release: SAE release identifier
            sae_id: Specific SAE layer/hook identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_id = sae_id

        print(f"Initializing SAE Analyzer on {device}...")
        print(f"Model: {model_name}")
        print(f"SAE Release: {sae_release}")
        print(f"SAE ID: {sae_id}")

        # Load model and SAE
        self.model = None
        self.sae = None
        self.load_components()

    def load_components(self):
        """Load the transformer model and SAE"""
        print("\n1. Loading transformer model...")
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device
        )
        print(f"   Model loaded: {self.model_name}")

        print("\n2. Loading SAE...")
        # Use the new API that doesn't unpack
        try:
            # Try the new method first
            self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device
            )
        except AttributeError:
            # Fall back to old method if new one doesn't exist
            result = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device
            )
            # Handle both old (tuple) and new (single object) return types
            if isinstance(result, tuple):
                self.sae, self.cfg_dict, self.sparsity = result
            else:
                self.sae = result
                self.cfg_dict = None
                self.sparsity = None

        print(f"   SAE loaded: {self.sae_id}")
        print(f"   Dictionary size: {self.sae.cfg.d_sae}")
        print(f"   Input dimension: {self.sae.cfg.d_in}")
        if self.sparsity is not None:
            # Handle sparsity - could be a scalar, 1-element tensor, or multi-element tensor
            if torch.is_tensor(self.sparsity):
                if self.sparsity.numel() == 1:
                    sparsity_val = self.sparsity.item()
                else:
                    # Multiple values - take the mean
                    sparsity_val = self.sparsity.mean().item()
            else:
                sparsity_val = self.sparsity
            print(f"   Average Sparsity: {sparsity_val:.4f}")

    def analyze_text(
        self,
        text: str,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Analyze text and get SAE activations

        Args:
            text: Input text to analyze
            top_k: Number of top features to return

        Returns:
            Tuple of (feature_acts, top_indices, tokens)
        """
        print(f"\n3. Analyzing text: '{text}'")

        # Tokenize
        tokens = self.model.to_tokens(text)
        print(f"   Tokens: {tokens.shape}")

        # Get token strings
        token_strs = [
            self.model.to_string(tokens[0, i])
            for i in range(tokens.shape[1])
        ]

        # Run model and get activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)

            # Get the activations at the SAE hook point
            hook_name = self.sae_id.replace("blocks.", "blocks.")
            sae_in = cache[hook_name]

            # Encode through SAE
            feature_acts = self.sae.encode(sae_in)

            # Decode to verify reconstruction
            sae_out = self.sae.decode(feature_acts)

            # Calculate reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(sae_out, sae_in)
            print(f"   Reconstruction loss: {recon_loss.item():.6f}")

        # Get top activated features
        feature_acts_flat = feature_acts.squeeze()
        if len(feature_acts_flat.shape) > 1:
            # Average over sequence length
            feature_acts_flat = feature_acts_flat.mean(dim=0)

        top_values, top_indices = torch.topk(feature_acts_flat, top_k)

        print(f"\n   Top {top_k} activated features:")
        for idx, (feat_idx, value) in enumerate(zip(top_indices, top_values)):
            print(f"   {idx+1}. Feature {feat_idx.item()}: {value.item():.4f}")

        return feature_acts, top_indices, token_strs

    def visualize_activations(
        self,
        text: str,
        top_k: int = 10,
        show_heatmap: bool = True,
        show_top_features: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Visualize SAE activations for given text

        Args:
            text: Input text to analyze
            top_k: Number of top features to visualize
            show_heatmap: Whether to show activation heatmap
            show_top_features: Whether to show top features bar chart
            save_path: Optional path to save the figure
        """
        # Get activations
        feature_acts, top_indices, tokens = self.analyze_text(text, top_k)

        # Prepare data for visualization
        feature_acts_np = feature_acts.squeeze().cpu().numpy()

        # Handle different shapes
        if len(feature_acts_np.shape) == 1:
            # Single token or averaged
            feature_acts_np = feature_acts_np.reshape(1, -1)

        print(f"\n4. Creating visualizations...")

        # Create subplots
        n_plots = int(show_heatmap) + int(show_top_features)
        if n_plots == 0:
            print("   No visualizations requested")
            return

        specs = [[{"type": "heatmap"}] if show_heatmap else [{"type": "bar"}]]
        if n_plots == 2:
            specs = [[{"type": "heatmap"}], [{"type": "bar"}]]

        fig = make_subplots(
            rows=n_plots,
            cols=1,
            subplot_titles=[
                title for title, show in [
                    ("SAE Feature Activations Heatmap", show_heatmap),
                    (f"Top {top_k} Activated Features", show_top_features)
                ] if show
            ],
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4] if n_plots == 2 else [1.0]
        )

        row = 1

        # Plot 1: Heatmap of activations
        if show_heatmap:
            # Select top features for heatmap
            top_feature_acts = feature_acts_np[:, top_indices.cpu().numpy()]

            heatmap = go.Heatmap(
                z=top_feature_acts.T,
                x=tokens,
                y=[f"Feature {idx.item()}" for idx in top_indices],
                colorscale='Viridis',
                colorbar=dict(title="Activation"),
                hovertemplate=(
                    'Token: %{x}<br>'
                    'Feature: %{y}<br>'
                    'Activation: %{z:.4f}<br>'
                    '<extra></extra>'
                )
            )

            fig.add_trace(heatmap, row=row, col=1)
            fig.update_xaxes(title_text="Tokens", row=row, col=1)
            fig.update_yaxes(title_text="SAE Features", row=row, col=1)
            row += 1

        # Plot 2: Bar chart of top features
        if show_top_features:
            # Average activation across tokens for each feature
            avg_acts = feature_acts_np[:, top_indices.cpu().numpy()].mean(axis=0)

            bar = go.Bar(
                x=[f"F{idx.item()}" for idx in top_indices],
                y=avg_acts,
                marker=dict(
                    color=avg_acts,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Activation", x=1.15)
                ),
                hovertemplate=(
                    'Feature: %{x}<br>'
                    'Avg Activation: %{y:.4f}<br>'
                    '<extra></extra>'
                )
            )

            fig.add_trace(bar, row=row, col=1)
            fig.update_xaxes(title_text="Feature Index", row=row, col=1)
            fig.update_yaxes(title_text="Average Activation", row=row, col=1)

        # Update layout
        fig.update_layout(
            title_text=f"SAE Analysis: {self.sae_id}<br>Text: '{text}'",
            height=400 * n_plots,
            showlegend=False,
            hovermode='closest'
        )

        # Save or show
        if save_path:
            fig.write_html(save_path)
            print(f"   Visualization saved to: {save_path}")
        else:
            fig.show()
            print("   Visualization displayed")

    def compare_texts(
        self,
        texts: List[str],
        top_k: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Compare SAE activations across multiple texts

        Args:
            texts: List of texts to compare
            top_k: Number of top features to track
            save_path: Optional path to save the figure
        """
        print(f"\n5. Comparing {len(texts)} texts...")

        all_activations = []
        all_top_features = []

        for i, text in enumerate(texts):
            print(f"\n   Text {i+1}: '{text}'")
            feature_acts, top_indices, _ = self.analyze_text(text, top_k)

            # Average over sequence length
            avg_acts = feature_acts.squeeze().mean(dim=0) if len(feature_acts.squeeze().shape) > 1 else feature_acts.squeeze()
            all_activations.append(avg_acts.cpu().numpy())
            all_top_features.extend(top_indices.cpu().numpy())

        # Get unique top features across all texts
        unique_features = sorted(set(all_top_features))[:top_k * 2]  # Track more for comparison

        # Create comparison matrix
        comparison_matrix = np.zeros((len(texts), len(unique_features)))
        for i, acts in enumerate(all_activations):
            for j, feat_idx in enumerate(unique_features):
                comparison_matrix[i, j] = acts[feat_idx]

        # Create visualization
        fig = go.Figure(data=go.Heatmap(
            z=comparison_matrix,
            x=[f"F{idx}" for idx in unique_features],
            y=[f"Text {i+1}" for i in range(len(texts))],
            colorscale='Viridis',
            colorbar=dict(title="Activation"),
            hovertemplate=(
                'Feature: %{x}<br>'
                'Text: %{y}<br>'
                'Activation: %{z:.4f}<br>'
                '<extra></extra>'
            )
        ))

        fig.update_layout(
            title="Comparing SAE Activations Across Texts",
            xaxis_title="Feature Index",
            yaxis_title="Text",
            height=max(400, len(texts) * 50)
        )

        # Add text annotations
        annotations_text = "<br>".join([f"Text {i+1}: {text}" for i, text in enumerate(texts)])
        fig.add_annotation(
            text=annotations_text,
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            align="left",
            font=dict(size=10)
        )

        if save_path:
            fig.write_html(save_path)
            print(f"\n   Comparison saved to: {save_path}")
        else:
            fig.show()
            print("\n   Comparison displayed")

    def analyze_feature_weights(
        self,
        feature_indices: List[int],
        top_tokens: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Analyze decoder weights for specific features to understand what they represent

        Args:
            feature_indices: List of feature indices to analyze
            top_tokens: Number of top tokens to show per feature
            save_path: Optional path to save results
        """
        print(f"\n6. Analyzing feature weights...")

        decoder_weights = self.sae.W_dec  # Shape: [d_sae, d_in]

        results = []
        for feat_idx in feature_indices:
            print(f"\n   Feature {feat_idx}:")

            # Get the decoder vector for this feature
            feat_vector = decoder_weights[feat_idx]

            # Project onto model's unembedding to see token associations
            with torch.no_grad():
                token_logits = feat_vector @ self.model.W_U  # [d_vocab]

                # Get top tokens
                top_values, top_token_ids = torch.topk(token_logits, top_tokens)

                print(f"   Top {top_tokens} associated tokens:")
                tokens_info = []
                for i, (token_id, value) in enumerate(zip(top_token_ids, top_values)):
                    token_str = self.model.to_string(token_id)
                    print(f"     {i+1}. '{token_str}': {value.item():.4f}")
                    tokens_info.append((token_str, value.item()))

                results.append({
                    'feature_idx': feat_idx,
                    'top_tokens': tokens_info
                })

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n   Results saved to: {save_path}")

        return results


def main():
    """Main function demonstrating the complete SAE analysis workflow"""

    print("=" * 70)
    print("COMPREHENSIVE SAE ANALYSIS TOOL")
    print("=" * 70)

    # Initialize analyzer
    analyzer = SAEAnalyzer(
        model_name="gpt2-small",
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre"
    )

    print("\n" + "=" * 70)
    print("SINGLE TEXT ANALYSIS")
    print("=" * 70)

    # Analyze a single text
    test_text = "The quick brown fox jumps over the lazy dog"
    analyzer.visualize_activations(
        text=test_text,
        top_k=15,
        show_heatmap=True,
        show_top_features=True,
        save_path="outputs/sae_single_analysis.html"
    )

    print("\n" + "=" * 70)
    print("MULTI-TEXT COMPARISON")
    print("=" * 70)

    # Compare multiple texts
    comparison_texts = [
        "I love machine learning",
        "I hate traffic jams",
        "The weather is beautiful today",
        "Python is a great programming language"
    ]

    analyzer.compare_texts(
        texts=comparison_texts,
        top_k=10,
        save_path="outputs/sae_comparison.html"
    )

    print("\n" + "=" * 70)
    print("FEATURE WEIGHT ANALYSIS")
    print("=" * 70)

    # Analyze specific features
    _, top_indices, _ = analyzer.analyze_text(test_text, top_k=5)
    analyzer.analyze_feature_weights(
        feature_indices=top_indices.cpu().tolist(),
        top_tokens=15,
        save_path="outputs/sae_feature_weights.json"
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - outputs/sae_single_analysis.html")
    print("  - outputs/sae_comparison.html")
    print("  - outputs/sae_feature_weights.json")


if __name__ == "__main__":
    main()