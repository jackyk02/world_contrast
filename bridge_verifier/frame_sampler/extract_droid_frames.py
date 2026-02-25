#!/usr/bin/env python3
"""
Extract uniformly sampled frames from the DROID dataset using the same
preprocessing pipeline as training (OpenPiRldsTwoViewBatchedIterable).

This script samples N frames uniformly and saves:
- Exterior camera image (after full preprocessing pipeline)
- Wrist camera image (after full preprocessing pipeline)
- Metadata JSON with instruction, actions, and preprocessing info

The output format matches the examples saved by finetune_droid_two_view_ddp.py.

Usage:
    python extract_droid_frames.py \
        --data_dir /root/data \
        --output_dir ./droid_sampled_frames \
        --num_frames 100 \
        --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables before importing heavy libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PADDING_VALUE = -5.0


def _extract_preprocess_norm(preprocess):
    """Extract normalization mean/std from CLIP-style preprocess."""
    default_mean = [0.48145466, 0.4578275, 0.40821073]
    default_std = [0.26862954, 0.26130258, 0.27577711]
    try:
        transforms = getattr(preprocess, "transforms", None)
        if transforms:
            for t in transforms:
                if hasattr(t, "mean") and hasattr(t, "std"):
                    mean = [float(m) for m in t.mean]
                    std = [float(s) for s in t.std]
                    return mean, std
    except Exception:
        pass
    return default_mean, default_std


def _tensor_to_pil(image_tensor, mean, std):
    """Convert preprocessed tensor back to PIL image."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    tensor = image_tensor.detach().cpu().float()
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    tensor = (tensor * std_t) + mean_t
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def extract_frames(
    data_dir: str,
    output_dir: str,
    num_frames: int = 100,
    seed: int = 42,
    backbone: str = "hf-hub:timm/ViT-L-16-SigLIP2-384",
    action_chunk_size: int = 16,
    filter_dict_path: str = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
    instruction_mapping: str | None = None,
    target_height: int = 224,
    target_width: int = 224,
    use_delta_actions: bool = True,
    batch_size: int = 32,
):
    """
    Extract uniformly sampled frames from DROID dataset using training pipeline.
    
    Args:
        data_dir: Parent directory of droid/ TFDS folder
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
        seed: Random seed for reproducibility
        backbone: SigLIP2 model backbone
        action_chunk_size: Action chunk size (same as training)
        filter_dict_path: Idle frame filter JSON path
        instruction_mapping: Optional instruction rephrase mapping
        target_height: Target height for images
        target_width: Target width for images
        use_delta_actions: Whether to use delta actions
        batch_size: Batch size for data loading
    """
    from open_clip import create_model_from_pretrained
    from openpi_rlds_two_view import OpenPiRldsTwoViewBatchedIterable, OpenPiRldsTwoViewConfig
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading SigLIP2 model: {backbone}")
    _, preprocess = create_model_from_pretrained(backbone)
    
    print(f"Creating DROID data loader from {data_dir}...")
    print(f"  Action chunk size: {action_chunk_size} (total window: {2 * action_chunk_size - 1})")
    print(f"  Filter: {filter_dict_path or 'disabled'}")
    print(f"  Delta actions: {use_delta_actions}")
    
    # Create dataset with training pipeline settings
    dataset = OpenPiRldsTwoViewBatchedIterable(
        OpenPiRldsTwoViewConfig(
            data_dir=data_dir,
            per_rank_batch_size=batch_size,
            action_chunk_size=action_chunk_size,
            expected_action_dim=8,
            shuffle=True,  # Shuffle for uniform sampling
            filter_dict_path=filter_dict_path if filter_dict_path else None,
            preprocess=preprocess,
            instruction_mapping_path=instruction_mapping,
            instruction_mode="original",  # Use original instructions only
            max_rephrases=1,
            target_height=target_height,
            target_width=target_width,
            resize_mode="bilinear",
            use_delta_actions=use_delta_actions,
            rank=0,
            world_size=1,
        )
    )
    
    # Get normalization params for saving images
    mean, std = _extract_preprocess_norm(preprocess)
    history_length = 2 * action_chunk_size - 1
    
    print(f"\nExtracting {num_frames} uniformly sampled frames...")
    print(f"  Image preprocessing: SigLIP2 ({backbone})")
    print(f"  History length: {history_length}")
    print(f"  Output directory: {output_dir}")
    
    saved = 0
    iterator = iter(dataset)
    metadata_all = []
    
    # Calculate how many batches we need to process to get enough samples
    # We'll sample uniformly by skipping random numbers of batches
    total_batches_to_scan = max(num_frames * 10 // batch_size, 100)  # Scan enough batches
    
    pbar = tqdm(total=num_frames, desc="Extracting frames")
    
    # Collect candidate frames first, then sample uniformly
    candidates = []
    batches_seen = 0
    
    while len(candidates) < num_frames * 5 and batches_seen < total_batches_to_scan:
        try:
            batch = next(iterator)
        except StopIteration:
            print("Dataset exhausted, restarting iterator...")
            iterator = iter(dataset)
            batch = next(iterator)
        
        if batch is None:
            continue
        
        batches_seen += 1
        
        # Unpack batch
        ext_imgs, wrist_imgs, captions, actions_batch = batch
        current_batch_size = ext_imgs.shape[0]
        
        # Store all items from this batch as candidates
        for i in range(current_batch_size):
            candidates.append({
                'ext_img': ext_imgs[i].clone(),
                'wrist_img': wrist_imgs[i].clone() if wrist_imgs is not None else None,
                'caption': captions[i],
                'actions': actions_batch[i].clone(),
            })
        
        # Update progress
        if len(candidates) >= num_frames:
            break
    
    print(f"\nCollected {len(candidates)} candidates from {batches_seen} batches")
    
    # Uniformly sample from candidates
    if len(candidates) < num_frames:
        print(f"Warning: Only {len(candidates)} frames available, using all of them")
        num_frames = len(candidates)
    
    # Random sample indices
    sample_indices = np.random.choice(len(candidates), size=num_frames, replace=False)
    sample_indices = sorted(sample_indices)
    
    print(f"Saving {num_frames} uniformly sampled frames...")
    
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Saving frames")):
        sample = candidates[sample_idx]
        ext_img = sample['ext_img']
        wrist_img = sample['wrist_img']
        caption = sample['caption']
        actions = sample['actions']
        
        actions_np = actions.detach().cpu().numpy()
        non_padding_mask = ~(np.isclose(actions_np, PADDING_VALUE).all(axis=-1))
        
        # Build metadata (same format as training examples)
        metadata = {
            "frame_id": idx,
            "caption": caption,
            "actions_shape": list(actions_np.shape),
            "non_padding_steps": int(non_padding_mask.sum()),
            "padding_value": PADDING_VALUE,
            "history_length": history_length,
            "actions": actions_np.tolist(),
        }
        
        try:
            # Save exterior image
            ext_pil = _tensor_to_pil(ext_img, mean, std)
            ext_path = os.path.join(output_dir, f"example_{idx}_ext.png")
            ext_pil.save(ext_path)
            metadata["ext_image"] = f"example_{idx}_ext.png"
            
            # Save wrist image
            if wrist_img is not None:
                wrist_pil = _tensor_to_pil(wrist_img, mean, std)
                wrist_path = os.path.join(output_dir, f"example_{idx}_wrist.png")
                wrist_pil.save(wrist_path)
                metadata["wrist_image"] = f"example_{idx}_wrist.png"
            
            # Save individual metadata JSON
            meta_path = os.path.join(output_dir, f"example_{idx}.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            metadata_all.append(metadata)
            saved += 1
            
        except Exception as e:
            print(f"Failed to save frame {idx}: {e}")
            continue
    
    # Save combined metadata
    combined_meta_path = os.path.join(output_dir, "metadata.json")
    combined_meta = {
        "num_frames": saved,
        "seed": seed,
        "data_dir": data_dir,
        "backbone": backbone,
        "action_chunk_size": action_chunk_size,
        "history_length": history_length,
        "target_height": target_height,
        "target_width": target_width,
        "use_delta_actions": use_delta_actions,
        "filter_dict_path": filter_dict_path,
        "padding_value": PADDING_VALUE,
        "preprocess_mean": mean,
        "preprocess_std": std,
        "frames": metadata_all,
    }
    with open(combined_meta_path, "w") as f:
        json.dump(combined_meta, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"  Saved {saved} frames to {output_dir}")
    print(f"  Combined metadata: {combined_meta_path}")
    print(f"\nOutput format matches training examples (example_X_ext.png, example_X_wrist.png, example_X.json)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract uniformly sampled frames from DROID using training preprocessing pipeline"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/data",
        help="Parent directory of droid/ TFDS folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./droid_sampled_frames",
        help="Directory to save extracted frames",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Number of frames to extract",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="hf-hub:timm/ViT-L-16-SigLIP2-384",
        help="SigLIP2 model backbone (same as training)",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=16,
        help="Action chunk size (same as training, creates 2*N-1 window)",
    )
    parser.add_argument(
        "--filter_dict_path",
        type=str,
        default="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        help="Path to idle frame filter JSON (empty string to disable)",
    )
    parser.add_argument(
        "--instruction_mapping",
        type=str,
        default=None,
        help="Optional instruction rephrase mapping JSON",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=224,
        help="Target height for images",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=224,
        help="Target width for images",
    )
    parser.add_argument(
        "--use_delta_actions",
        action="store_true",
        default=True,
        help="Use delta actions (default: True, same as training)",
    )
    parser.add_argument(
        "--no_delta_actions",
        action="store_true",
        help="Disable delta actions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for data loading",
    )
    
    args = parser.parse_args()
    
    # Handle delta actions flag
    use_delta = args.use_delta_actions and not args.no_delta_actions
    
    # Handle empty filter path
    filter_path = args.filter_dict_path if args.filter_dict_path else None
    
    extract_frames(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        seed=args.seed,
        backbone=args.backbone,
        action_chunk_size=args.action_chunk_size,
        filter_dict_path=filter_path,
        instruction_mapping=args.instruction_mapping,
        target_height=args.target_height,
        target_width=args.target_width,
        use_delta_actions=use_delta,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
