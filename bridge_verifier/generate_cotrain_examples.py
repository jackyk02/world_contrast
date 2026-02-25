#!/usr/bin/env python3
"""
Generate example visualizations from the Polaris DROID co-training dataset.

This script loads the TFDS-format co-training dataset and saves example images
and metadata in the same format as the DROID verifier examples.

The preprocessing pipeline matches the training code exactly:
1. Resize with aspect ratio preservation and padding (to pre_resize_height x pre_resize_width)
2. Apply CLIP/SigLIP2 preprocess (resize to model input size, normalize)
3. Reverse normalization to save as PNG

Usage:
    python generate_cotrain_examples.py \
        --data_dir /root/data/polaris_droid_cotrain_dataset \
        --output_dir cotrain_examples \
        --num_examples 10 \
        --action_chunk_size 16
"""

import argparse
import json
import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch

# Reduce TensorFlow log spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow_datasets as tfds

# Import CLIP model loading
from open_clip import create_model_from_pretrained

# Action normalization constants (from DROID/PI0)
DROID_ACTION_Q01 = np.array([
    -0.45159999, -0.79799998, -0.4384, -0.90880001,
    -0.634, -0.63279998, -0.75160003, 0.0
], dtype=np.float32)

DROID_ACTION_Q99 = np.array([
    0.43880001, 0.76440001, 0.44319999, 0.78600001,
    0.63800001, 0.65679997, 0.72439998, 0.9982
], dtype=np.float32)

PADDING_VALUE = -5.0


def normalize_actions_quantile(
    actions: np.ndarray,
    q01: np.ndarray = DROID_ACTION_Q01,
    q99: np.ndarray = DROID_ACTION_Q99,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """
    Normalize actions:
    - Dims 0-6 (joints): quantile normalization to [-1, 1]
    - Dim 7 (gripper): binarized to 0 or 1
    """
    action_dim = actions.shape[-1]
    result = actions.copy()
    
    joint_dims = min(7, action_dim)
    q01_joints = q01[:joint_dims]
    q99_joints = q99[:joint_dims]
    result[..., :joint_dims] = (actions[..., :joint_dims] - q01_joints) / (q99_joints - q01_joints + 1e-6) * 2.0 - 1.0
    
    if action_dim > 7:
        result[..., 7] = (actions[..., 7] >= gripper_threshold).astype(np.float32)
    
    return result


def compute_action_deltas(
    actions: np.ndarray,
    state: np.ndarray,
    delta_mask: tuple = (True, True, True, True, True, True, True, False),
) -> np.ndarray:
    """Convert absolute actions to delta actions (action - state)."""
    actions = np.array(actions, copy=True, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(delta_mask, dtype=bool)
    dims = mask.shape[-1]
    actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    return actions


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image with aspect ratio preserved using padding."""
    pil_img = Image.fromarray(image)
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    new_height = int(cur_height / ratio)
    new_width = int(cur_width / ratio)
    
    pil_img = pil_img.resize((new_width, new_height), Image.BILINEAR)
    
    # Create padded image
    padded = Image.new("RGB", (width, height), (0, 0, 0))
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    padded.paste(pil_img, (paste_x, paste_y))
    
    return np.array(padded)


def extract_preprocess_norm(preprocess: Callable) -> tuple[list[float], list[float]]:
    """Extract normalization mean/std from CLIP preprocess transform."""
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


def tensor_to_pil(image_tensor: torch.Tensor, mean: list[float], std: list[float]) -> Image.Image:
    """
    Reverse CLIP preprocess normalization and convert tensor to PIL image.
    Matches _tensor_to_pil in finetune_droid_two_view_ddp.py
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    tensor = image_tensor.detach().cpu().float()
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    tensor = (tensor * std_t) + mean_t
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def get_first_non_empty_instruction(step) -> Optional[str]:
    """Get first non-empty language instruction from step."""
    for key in ["language_instruction", "language_instruction_2", "language_instruction_3"]:
        if key in step:
            instr = step[key]
            if isinstance(instr, bytes):
                instr = instr.decode("utf-8", errors="replace")
            elif hasattr(instr, "numpy"):
                instr = instr.numpy()
                if isinstance(instr, bytes):
                    instr = instr.decode("utf-8", errors="replace")
            if instr and str(instr).strip():
                return str(instr).strip()
    return None


def extract_episode_data(episode, action_chunk_size: int, use_delta_actions: bool = False):
    """
    Extract steps from an episode and create action windows.
    
    Uses the same clamping approach as the original DROID RLDS loader:
    - Early timesteps repeat the first action for history
    - Late timesteps repeat the last action for future
    
    Returns list of dicts with keys: ext_image, wrist_image, instruction, actions, state
    """
    steps = list(episode["steps"])
    if not steps:
        return []
    
    # Extract all data from steps
    all_actions = []
    all_states = []
    all_ext_images = []
    all_wrist_images = []
    instruction = None
    
    for step in steps:
        # Action: [8] (7 joints + 1 gripper)
        action = step["action"]
        if hasattr(action, "numpy"):
            action = action.numpy()
        all_actions.append(np.array(action, dtype=np.float32))
        
        # State: joint_position [7] + gripper_position [1]
        obs = step["observation"]
        joint_pos = obs["joint_position"]
        gripper_pos = obs["gripper_position"]
        if hasattr(joint_pos, "numpy"):
            joint_pos = joint_pos.numpy()
        if hasattr(gripper_pos, "numpy"):
            gripper_pos = gripper_pos.numpy()
        state = np.concatenate([joint_pos, gripper_pos], axis=-1).astype(np.float32)
        all_states.append(state)
        
        # Images
        ext_img = obs.get("exterior_image_1_left")
        if ext_img is None or (hasattr(ext_img, "numpy") and ext_img.numpy().size == 0):
            ext_img = obs.get("exterior_image_2_left")
        if hasattr(ext_img, "numpy"):
            ext_img = ext_img.numpy()
        all_ext_images.append(ext_img)
        
        wrist_img = obs["wrist_image_left"]
        if hasattr(wrist_img, "numpy"):
            wrist_img = wrist_img.numpy()
        all_wrist_images.append(wrist_img)
        
        # Instruction (use first found)
        if instruction is None:
            instruction = get_first_non_empty_instruction(step)
    
    if instruction is None:
        return []
    
    all_actions = np.array(all_actions, dtype=np.float32)
    all_states = np.array(all_states, dtype=np.float32)
    
    # Create action windows: [history + current + future]
    # Total window size = 2 * action_chunk_size - 1
    # Using clamping approach like original DROID loader (repeat first/last actions)
    history_length = action_chunk_size - 1
    total_window = 2 * action_chunk_size - 1
    num_steps = len(all_actions)
    
    samples = []
    for t in range(num_steps):
        # Build indices for the action window centered at t
        # Range: [t - history_length, t + action_chunk_size)
        indices = np.arange(t - history_length, t + action_chunk_size)
        
        # Clamp indices to valid range [0, num_steps-1]
        # This repeats first action for early timesteps and last action for late timesteps
        clamped_indices = np.clip(indices, 0, num_steps - 1)
        
        # Gather actions using clamped indices
        action_window = all_actions[clamped_indices].copy()
        
        if action_window.shape[0] != total_window:
            continue
        
        # Get current state for delta computation
        current_state = all_states[t]
        
        # Apply delta actions if requested (before normalization)
        if use_delta_actions:
            delta_mask = np.array([True, True, True, True, True, True, True, False])
            # Apply delta: action[i] - current_state (for joint dims only)
            action_window[:, :7] = action_window[:, :7] - np.where(delta_mask[:7], current_state[:7], 0)
        
        # Always normalize (quantile normalization for joints, binarize gripper)
        action_window = normalize_actions_quantile(action_window)
        
        samples.append({
            "ext_image": all_ext_images[t],
            "wrist_image": all_wrist_images[t],
            "instruction": instruction,
            "actions": action_window,
            "timestep": t,
            "episode_length": num_steps,
        })
    
    return samples


def save_example(
    sample: dict,
    output_dir: str,
    example_idx: int,
    preprocess: Optional[Callable] = None,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    pre_resize_height: int = 224,
    pre_resize_width: int = 224,
):
    """
    Save a single example (images + JSON metadata).
    
    Pipeline (matching training code):
    1. Resize with aspect ratio preservation and padding (to pre_resize_height x pre_resize_width)
    2. Apply CLIP preprocess (resize to model input size, normalize)
    3. Reverse normalization to save as PNG
    """
    ext_img = sample["ext_image"]
    wrist_img = sample["wrist_image"]
    actions = sample["actions"]
    instruction = sample["instruction"]
    
    # Step 1: Resize with padding (before CLIP preprocess, matching DROID RLDS loader)
    ext_img = resize_with_pad(ext_img, pre_resize_height, pre_resize_width)
    wrist_img = resize_with_pad(wrist_img, pre_resize_height, pre_resize_width)
    
    # Step 2 & 3: Apply CLIP preprocess then reverse to get final images
    if preprocess is not None and mean is not None and std is not None:
        # Convert to PIL for preprocess
        ext_pil = Image.fromarray(ext_img)
        wrist_pil = Image.fromarray(wrist_img)
        
        # Apply CLIP preprocess (resizes to model input size + normalizes)
        ext_tensor = preprocess(ext_pil)
        wrist_tensor = preprocess(wrist_pil)
        
        # Reverse normalization to get back to image
        ext_pil = tensor_to_pil(ext_tensor, mean, std)
        wrist_pil = tensor_to_pil(wrist_tensor, mean, std)
    else:
        # Fallback: just convert to PIL
        ext_pil = Image.fromarray(ext_img)
        wrist_pil = Image.fromarray(wrist_img)
    
    # Save images
    ext_path = os.path.join(output_dir, f"example_{example_idx}_ext.png")
    wrist_path = os.path.join(output_dir, f"example_{example_idx}_wrist.png")
    ext_pil.save(ext_path)
    wrist_pil.save(wrist_path)
    
    # Count non-padding steps (all 31 should be non-padding with clamping approach)
    non_padding_mask = ~np.all(np.isclose(actions, PADDING_VALUE), axis=-1)
    
    # Save metadata
    metadata = {
        "caption": instruction,
        "actions_shape": list(actions.shape),
        "non_padding_steps": int(non_padding_mask.sum()),
        "padding_value": PADDING_VALUE,
        "history_length": actions.shape[0],
        "actions": actions.tolist(),
        "ext_image": ext_path,
        "wrist_image": wrist_path,
        "timestep_in_episode": sample.get("timestep", -1),
        "episode_length": sample.get("episode_length", -1),
    }
    
    meta_path = os.path.join(output_dir, f"example_{example_idx}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return meta_path


def main():
    parser = argparse.ArgumentParser(description="Generate examples from Polaris DROID co-training dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/data/polaris_droid_cotrain_dataset",
        help="Path to the parent directory containing polaris_droid_cotrain_dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cotrain_examples",
        help="Directory to save examples",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=16,
        help="Action chunk size (total window = 2*chunk-1)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="hf-hub:timm/ViT-L-16-SigLIP2-384",
        help="CLIP/SigLIP2 model backbone (determines image size and preprocessing)",
    )
    parser.add_argument(
        "--pre_resize_height",
        type=int,
        default=224,
        help="Pre-resize height (before CLIP preprocess, matching DROID RLDS loader)",
    )
    parser.add_argument(
        "--pre_resize_width",
        type=int,
        default=224,
        help="Pre-resize width (before CLIP preprocess, matching DROID RLDS loader)",
    )
    parser.add_argument(
        "--use_delta_actions",
        action="store_true",
        help="Convert actions to deltas (action - state) before normalization",
    )
    parser.add_argument(
        "--samples_per_episode",
        type=int,
        default=3,
        help="Number of samples to extract per episode (spread across timesteps)",
    )
    parser.add_argument(
        "--skip_episodes",
        type=int,
        default=0,
        help="Number of episodes to skip before starting",
    )
    args = parser.parse_args()
    
    # Configure TensorFlow
    tf.config.set_visible_devices([], "GPU")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading SigLIP2 model: {args.backbone}")
    siglip_model, preprocess = create_model_from_pretrained(args.backbone)
    mean, std = extract_preprocess_norm(preprocess)
    print(f"  Preprocess normalization: mean={mean}, std={std}")
    
    # Get model image size from preprocess
    try:
        for t in preprocess.transforms:
            if hasattr(t, "size"):
                model_img_size = t.size
                print(f"  Model input size: {model_img_size}")
                break
    except Exception:
        pass
    
    print(f"Loading dataset from {args.data_dir}...")
    print(f"Action chunk size: {args.action_chunk_size} (total window: {2*args.action_chunk_size - 1})")
    print(f"Pre-resize size: {args.pre_resize_height}x{args.pre_resize_width}")
    print(f"Delta actions: {args.use_delta_actions}")
    
    # Load TFDS dataset
    builder = tfds.builder(
        "polaris_droid_cotrain_dataset",
        data_dir=args.data_dir,
        version="1.0.0",
    )
    dataset = builder.as_dataset(split="train")
    
    saved_count = 0
    episode_count = 0
    
    for episode in dataset:
        episode_count += 1
        
        if episode_count <= args.skip_episodes:
            continue
        
        samples = extract_episode_data(
            episode,
            action_chunk_size=args.action_chunk_size,
            use_delta_actions=args.use_delta_actions,
        )
        
        if not samples:
            continue
        
        # Select samples spread across the episode
        num_to_take = min(args.samples_per_episode, len(samples))
        if num_to_take > 0:
            indices = np.linspace(0, len(samples) - 1, num_to_take, dtype=int)
            selected = [samples[i] for i in indices]
        else:
            selected = []
        
        for sample in selected:
            if saved_count >= args.num_examples:
                break
            
            meta_path = save_example(
                sample,
                args.output_dir,
                saved_count,
                preprocess=preprocess,
                mean=mean,
                std=std,
                pre_resize_height=args.pre_resize_height,
                pre_resize_width=args.pre_resize_width,
            )
            print(f"Saved example {saved_count}: {sample['instruction'][:50]}... (t={sample['timestep']}/{sample['episode_length']})")
            saved_count += 1
        
        if saved_count >= args.num_examples:
            break
    
    print(f"\nDone! Saved {saved_count} examples to {args.output_dir}")
    print(f"Processed {episode_count} episodes")


if __name__ == "__main__":
    main()

