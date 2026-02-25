#!/usr/bin/env python3
"""
Inference script for VLA-SigLIP2 DROID two-view model.

Tests retrieval accuracy on extracted DROID frames by:
1. For each sample, creating an action pool (ground truth + random negatives)
2. Computing similarity scores between (image, text) and all actions in pool
3. Reporting top-1 and top-5 retrieval accuracy

Usage:
    python inference_droid_two_view.py \
        --checkpoint /path/to/checkpoint.pt \
        --frames_dir ./droid_sampled_frames \
        --num_samples 100 \
        --action_pool_size 50
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from open_clip import create_model_from_pretrained, get_tokenizer

from model import AttentionPooling, ModelConfig, TextAwareVisualExtraction, sincos_position_embedding

PADDING_VALUE = -5.0


class VLA_SigLIP2_Droid(nn.Module):
    """
    VLA-SigLIP2 model for DROID two-view dataset.
    (Copied from finetune_droid_two_view_ddp.py for standalone inference)
    """

    def __init__(self, model_config, use_transformer: bool = False):
        super().__init__()
        self.model = model_config.clip_model
        for param in self.model.parameters():
            param.requires_grad = False
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        self.model.eval()

        text_pooling_output_dim = model_config.text_pooling_output_dim
        pooling_heads = model_config.pooling_heads
        pooling_layers = model_config.pooling_layers
        self.num_readouts = model_config.num_readouts

        text_dim = self.model.text.output_dim
        vision_dim = self.model.visual.trunk.num_features
        vision_pooling_output_dim = model_config.vision_pooling_output_dim

        self.visual_patch_size = self.model.visual.trunk.patch_embed.proj.kernel_size[0]
        image_size = self.model.visual.image_size[0] if hasattr(self.model.visual, "image_size") else 224
        self.num_img_patches = (image_size // self.visual_patch_size) ** 2

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            num_img_patches=self.num_img_patches,
            vision_dim=vision_dim,
        )
        self.text_pooling = AttentionPooling(
            text_dim,
            text_pooling_output_dim,
            pooling_heads,
            pooling_layers,
            num_readouts=self.num_readouts,
        )
        self.vision_poolings = AttentionPooling(
            vision_dim,
            vision_pooling_output_dim,
            pooling_heads,
            pooling_layers,
            num_readouts=self.num_readouts,
        )

        self.f_t_dim = text_pooling_output_dim + (2 * vision_pooling_output_dim)
        self.input_projection = nn.Linear(self.f_t_dim, vision_pooling_output_dim)

        self.action_dim = model_config.action_dim
        self.history_length = model_config.history_length
        self.use_transformer = use_transformer

        if self.use_transformer:
            self.single_step_action_encoder = nn.Linear(self.action_dim, vision_pooling_output_dim)
            self.register_buffer(
                "action_pos_embedding",
                sincos_position_embedding(self.history_length, vision_pooling_output_dim).unsqueeze(0),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=vision_pooling_output_dim,
                nhead=8,
                dim_feedforward=vision_pooling_output_dim * 2,
                dropout=0.1,
            )
            self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        else:
            self.complex_action_encoder = nn.Sequential(
                nn.Linear(self.history_length * self.action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, vision_pooling_output_dim),
            )

        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook

        self.hooks = [
            self.model.visual.trunk.blocks[-1].attn.register_forward_hook(
                get_activation("image_patches")
            ),
            self.model.text.transformer.register_forward_hook(
                get_activation("text_features")
            ),
        ]

        self.action_padding_value = PADDING_VALUE

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def extract_features(self, ext_images, wrist_images, text):
        ext_images = ext_images.to(torch.bfloat16)
        wrist_images = wrist_images.to(torch.bfloat16)
        stacked_images = torch.cat([ext_images, wrist_images], dim=0)
        with torch.no_grad():
            _ = self.model.encode_text(text, normalize=False)
            _ = self.model.encode_image(stacked_images, normalize=False)

        text_features = self.activation["text_features"]
        text_features = self.model.text.ln_final(text_features)
        if hasattr(self.model.text, "text_projection") and self.model.text.text_projection is not None:
            batch_size, seq_len, hidden_dim = text_features.shape
            text_features = self.model.text.text_projection(text_features.reshape(-1, hidden_dim))
            text_features = text_features.reshape(batch_size, seq_len, -1)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        patch_features = self.activation["image_patches"]
        b2 = patch_features.shape[0]
        b = b2 // 2
        ext_patch_features = patch_features[:b]
        wrist_patch_features = patch_features[b:]
        if patch_features.shape[1] == self.num_img_patches + 1:
            ext_patch_features = ext_patch_features[:, 1:, :]
            wrist_patch_features = wrist_patch_features[:, 1:, :]
        elif patch_features.shape[1] == self.num_img_patches:
            pass
        else:
            ext_patch_features = ext_patch_features[:, 1:, :]
            wrist_patch_features = wrist_patch_features[:, 1:, :]

        ext_patch_features = ext_patch_features.float()
        wrist_patch_features = wrist_patch_features.float()
        ext_patch_features = ext_patch_features / ext_patch_features.norm(dim=-1, keepdim=True)
        wrist_patch_features = wrist_patch_features / wrist_patch_features.norm(dim=-1, keepdim=True)
        
        self.activation.clear()
        
        return ext_patch_features, wrist_patch_features, text_features

    def forward_features(self, ext_image, wrist_image, text, action_histories):
        ext_patch_features, wrist_patch_features, text_features = self.extract_features(ext_image, wrist_image, text)

        ext_tav_features = self.text_aware_visual_extraction(ext_patch_features, text_features)
        wrist_tav_features = self.text_aware_visual_extraction(wrist_patch_features, text_features)
        ext_vision_token = self.vision_poolings(ext_tav_features)
        wrist_vision_token = self.vision_poolings(wrist_tav_features)
        vision_token = torch.cat([ext_vision_token, wrist_vision_token], dim=-1)

        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, vision_token], dim=-1)
        combined_features = self.input_projection(combined_features)
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

        action_histories = action_histories.float().to(ext_image.device)

        if self.use_transformer:
            padding_mask = (action_histories[:, :, 0] == self.action_padding_value)
            encoded_steps = self.single_step_action_encoder(action_histories)
            # Add positional embedding
            encoded_steps = encoded_steps + self.action_pos_embedding
            encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
            transformer_output_permuted = self.trajectory_encoder(
                encoded_steps_permuted, src_key_padding_mask=padding_mask
            )
            transformer_output = transformer_output_permuted.permute(1, 0, 2)
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            summed_features = (transformer_output * mask_expanded).sum(dim=1)
            num_non_padded = mask_expanded.sum(dim=1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)
            projected_trajectory = summed_features / num_non_padded
        else:
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            projected_trajectory = self.complex_action_encoder(flat_actions)

        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
        
        return combined_features, projected_trajectory

    def forward(self, ext_image, wrist_image, text, action_histories):
        combined_features, projected_trajectory = self.forward_features(ext_image, wrist_image, text, action_histories)
        
        logits_scale = self.logit_scale.exp()
        image_logits = logits_scale * torch.matmul(combined_features, projected_trajectory.T)
        action_logits = logits_scale * torch.matmul(projected_trajectory, combined_features.T)

        return image_logits, action_logits


class DroidTwoViewInference:
    """Inference class for VLA-SigLIP2 DROID two-view model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "hf-hub:timm/ViT-L-16-SigLIP2-384",
        history_length: int = 31,
        action_dim: int = 8,
        use_transformer: bool = True,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading SigLIP2 model: {backbone}")
        
        siglip2_model, self.preprocess = create_model_from_pretrained(backbone)
        siglip2_model = siglip2_model.to(self.device)
        self.tokenizer = get_tokenizer(backbone)
        self.context_length = siglip2_model.context_length
        
        # Initialize model
        model_config = ModelConfig(
            clip_model=siglip2_model,
            history_length=history_length,
            action_dim=action_dim,
        )
        self.model = VLA_SigLIP2_Droid(model_config, use_transformer=use_transformer).to(self.device)
        self.model.action_padding_value = PADDING_VALUE
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Use strict=False to handle old checkpoints missing new buffers (e.g., action_pos_embedding).
        # Fixed buffers like sinusoidal positional embeddings will use freshly initialized values.
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Checkpoint missing keys (using initialized values): {missing_keys}")
        if unexpected_keys:
            print(f"Checkpoint has unexpected keys (ignored): {unexpected_keys}")
        self.model.eval()
        print("Model loaded successfully!")
        
        # Extract normalization params for image loading
        self.mean, self.std = self._extract_preprocess_norm()
    
    def _extract_preprocess_norm(self):
        """Extract normalization mean/std from CLIP-style preprocess."""
        default_mean = [0.48145466, 0.4578275, 0.40821073]
        default_std = [0.26862954, 0.26130258, 0.27577711]
        try:
            transforms = getattr(self.preprocess, "transforms", None)
            if transforms:
                for t in transforms:
                    if hasattr(t, "mean") and hasattr(t, "std"):
                        return [float(m) for m in t.mean], [float(s) for s in t.std]
        except Exception:
            pass
        return default_mean, default_std
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        img = Image.open(image_path).convert("RGB")
        return self.preprocess(img).unsqueeze(0).to(self.device)
    
    def compute_similarity_scores(
        self,
        ext_img: torch.Tensor,
        wrist_img: torch.Tensor,
        text: str,
        action_pool: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute similarity scores between (image, text) and all actions in pool.
        
        Returns:
            scores: [pool_size] array of similarity scores
        """
        with torch.no_grad():
            # Tokenize text
            text_tokens = self.tokenizer([text], context_length=self.context_length).to(self.device)
            
            # Stack all actions
            actions = torch.tensor(np.stack(action_pool), dtype=torch.float32).to(self.device)
            pool_size = actions.shape[0]
            
            # Expand image and text to match pool size
            ext_img_batch = ext_img.expand(pool_size, -1, -1, -1)
            wrist_img_batch = wrist_img.expand(pool_size, -1, -1, -1)
            text_batch = text_tokens.expand(pool_size, -1)
            
            # Get similarity scores
            image_logits, _ = self.model(ext_img_batch, wrist_img_batch, text_batch, actions)
            
            # Diagonal gives similarity between each (image, text) pair and its corresponding action
            scores = torch.diag(image_logits).cpu().numpy()
            similarity_scores = scores.copy()/self.model.logit_scale.exp().item()
        
        return similarity_scores
    
    def predict(
        self,
        ext_img: torch.Tensor,
        wrist_img: torch.Tensor,
        text: str,
        action_pool: list[np.ndarray],
    ) -> tuple[int, np.ndarray]:
        """
        Predict the most likely action from pool.
        
        Returns:
            predicted_idx: Index of predicted action in pool
            scores: [pool_size] array of similarity scores
        """
        scores = self.compute_similarity_scores(ext_img, wrist_img, text, action_pool)
        predicted_idx = int(np.argmax(scores))
        return predicted_idx, scores


def load_frames_from_dir(frames_dir: str) -> list[dict]:
    """Load all extracted frames from directory."""
    frames = []
    
    # Try to load combined metadata
    metadata_path = os.path.join(frames_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        for frame_info in metadata.get("frames", []):
            frame_id = frame_info["frame_id"]
            ext_path = os.path.join(frames_dir, f"example_{frame_id}_ext.png")
            wrist_path = os.path.join(frames_dir, f"example_{frame_id}_wrist.png")
            
            if os.path.exists(ext_path) and os.path.exists(wrist_path):
                frames.append({
                    "frame_id": frame_id,
                    "ext_path": ext_path,
                    "wrist_path": wrist_path,
                    "caption": frame_info["caption"],
                    "actions": np.array(frame_info["actions"], dtype=np.float32),
                })
    else:
        # Fallback: load individual JSONs
        json_files = sorted(Path(frames_dir).glob("example_*.json"))
        for json_path in json_files:
            if json_path.name == "metadata.json":
                continue
            with open(json_path, "r") as f:
                frame_info = json.load(f)
            
            frame_id = frame_info.get("frame_id", int(json_path.stem.split("_")[1]))
            ext_path = os.path.join(frames_dir, f"example_{frame_id}_ext.png")
            wrist_path = os.path.join(frames_dir, f"example_{frame_id}_wrist.png")
            
            if os.path.exists(ext_path) and os.path.exists(wrist_path):
                frames.append({
                    "frame_id": frame_id,
                    "ext_path": ext_path,
                    "wrist_path": wrist_path,
                    "caption": frame_info["caption"],
                    "actions": np.array(frame_info["actions"], dtype=np.float32),
                })
    
    return frames


def evaluate_retrieval(
    inference: DroidTwoViewInference,
    frames: list[dict],
    num_samples: int,
    action_pool_size: int,
    seed: int = 42,
) -> dict:
    """
    Evaluate retrieval accuracy on extracted frames.
    
    Args:
        inference: DroidTwoViewInference instance
        frames: List of frame dictionaries
        num_samples: Number of samples to test
        action_pool_size: Size of action pool (including ground truth)
        seed: Random seed
    
    Returns:
        Dictionary with evaluation metrics
    """
    np.random.seed(seed)
    
    if len(frames) < num_samples:
        print(f"Warning: Only {len(frames)} frames available, using all")
        num_samples = len(frames)
    
    if len(frames) < action_pool_size:
        print(f"Warning: Only {len(frames)} frames available for pool, reducing pool size")
        action_pool_size = len(frames)
    
    # Sample frames for testing
    sample_indices = np.random.choice(len(frames), size=num_samples, replace=False)
    
    # Collect all actions for negative sampling
    all_actions = [f["actions"] for f in frames]
    
    results = {
        "top1_correct": 0,
        "top5_correct": 0,
        "total": num_samples,
        "ranks": [],
        "details": [],
    }
    
    print(f"\nEvaluating {num_samples} samples with pool size {action_pool_size}...")
    
    for idx in tqdm(sample_indices, desc="Evaluating"):
        frame = frames[idx]
        gt_actions = frame["actions"]
        
        # Build action pool: ground truth + random negatives
        action_pool = [gt_actions]
        gt_idx_in_pool = 0
        
        # Sample negative actions
        negative_indices = [i for i in range(len(frames)) if i != idx]
        if len(negative_indices) >= action_pool_size - 1:
            neg_sample_indices = np.random.choice(
                negative_indices, size=action_pool_size - 1, replace=False
            )
        else:
            neg_sample_indices = negative_indices
        
        for neg_idx in neg_sample_indices:
            action_pool.append(all_actions[neg_idx])
        
        # Shuffle pool and track ground truth position
        pool_order = np.random.permutation(len(action_pool))
        action_pool = [action_pool[i] for i in pool_order]
        gt_idx_in_pool = int(np.where(pool_order == 0)[0][0])
        
        # Load images
        ext_img = inference.load_image(frame["ext_path"])
        wrist_img = inference.load_image(frame["wrist_path"])
        
        # Get prediction
        predicted_idx, scores = inference.predict(
            ext_img, wrist_img, frame["caption"], action_pool
        )
        
        # Calculate rank of ground truth
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        gt_rank = int(np.where(sorted_indices == gt_idx_in_pool)[0][0]) + 1
        results["ranks"].append(gt_rank)
        
        # Check top-1 and top-5
        if gt_rank == 1:
            results["top1_correct"] += 1
        if gt_rank <= 5:
            results["top5_correct"] += 1
        
        # Store details
        results["details"].append({
            "frame_id": frame["frame_id"],
            "caption": frame["caption"],
            "gt_idx_in_pool": gt_idx_in_pool,
            "predicted_idx": predicted_idx,
            "gt_rank": gt_rank,
            "pool_size": len(action_pool),
        })
    
    # Calculate metrics
    results["top1_accuracy"] = results["top1_correct"] / results["total"]
    results["top5_accuracy"] = results["top5_correct"] / results["total"]
    results["mean_rank"] = np.mean(results["ranks"])
    results["median_rank"] = np.median(results["ranks"])
    
    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {results['total']}")
    print(f"Action pool size: {results['details'][0]['pool_size'] if results['details'] else 'N/A'}")
    print("-" * 60)
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f} ({results['top1_correct']}/{results['total']})")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_correct']}/{results['total']})")
    print(f"Mean Rank: {results['mean_rank']:.2f}")
    print(f"Median Rank: {results['median_rank']:.2f}")
    print("=" * 60)
    
    # Show rank distribution
    ranks = results["ranks"]
    print("\nRank Distribution:")
    for r in [1, 2, 3, 4, 5, 10, 20]:
        count = sum(1 for rank in ranks if rank <= r)
        pct = count / len(ranks) * 100
        print(f"  Rank <= {r:2d}: {count:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA-SigLIP2 DROID two-view model on extracted frames"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="./droid_sampled_frames",
        help="Directory containing extracted frames",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--action_pool_size",
        type=int,
        default=50,
        help="Size of action pool for retrieval (including ground truth)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="hf-hub:timm/ViT-L-16-SigLIP2-384",
        help="SigLIP2 model backbone",
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=31,
        help="Action history length (default: 31 for action_chunk_size=16)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=8,
        help="Action dimension (default: 8 for DROID)",
    )
    parser.add_argument(
        "--use_transformer",
        action="store_true",
        default=True,
        help="Use transformer action encoder (default: True)",
    )
    parser.add_argument(
        "--no_transformer",
        action="store_true",
        help="Use MLP action encoder instead of transformer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional: save detailed results to JSON file",
    )
    
    args = parser.parse_args()
    
    # Handle transformer flag
    use_transformer = args.use_transformer and not args.no_transformer
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.frames_dir):
        print(f"Error: Frames directory not found: {args.frames_dir}")
        return
    
    # Load frames
    print(f"Loading frames from {args.frames_dir}...")
    frames = load_frames_from_dir(args.frames_dir)
    print(f"Loaded {len(frames)} frames")
    
    if len(frames) == 0:
        print("Error: No valid frames found!")
        return
    
    # Infer history length from first frame if not specified
    first_actions_shape = frames[0]["actions"].shape
    inferred_history = first_actions_shape[0]
    if args.history_length != inferred_history:
        print(f"Note: Using history_length={args.history_length} (frames have {inferred_history})")
    
    # Initialize inference
    inference = DroidTwoViewInference(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        history_length=args.history_length,
        action_dim=args.action_dim,
        use_transformer=use_transformer,
    )
    
    # Run evaluation
    results = evaluate_retrieval(
        inference=inference,
        frames=frames,
        num_samples=args.num_samples,
        action_pool_size=args.action_pool_size,
        seed=args.seed,
    )
    
    # Print results
    print_results(results)
    
    # Optionally save to JSON
    if args.output_json:
        # Convert numpy types for JSON serialization
        results_json = {
            "top1_accuracy": float(results["top1_accuracy"]),
            "top5_accuracy": float(results["top5_accuracy"]),
            "top1_correct": int(results["top1_correct"]),
            "top5_correct": int(results["top5_correct"]),
            "total": int(results["total"]),
            "mean_rank": float(results["mean_rank"]),
            "median_rank": float(results["median_rank"]),
            "ranks": [int(r) for r in results["ranks"]],
            "config": {
                "checkpoint": args.checkpoint,
                "frames_dir": args.frames_dir,
                "num_samples": args.num_samples,
                "action_pool_size": args.action_pool_size,
                "backbone": args.backbone,
                "history_length": args.history_length,
                "use_transformer": use_transformer,
                "seed": args.seed,
            },
            "details": results["details"],
        }
        with open(args.output_json, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nDetailed results saved to {args.output_json}")


if __name__ == "__main__":
    main()



