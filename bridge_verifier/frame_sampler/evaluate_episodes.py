#!/usr/bin/env python3
"""
Evaluate VLA-SigLIP2 DROID model by computing frame-level scores per episode.

This script:
1. Uniformly samples N episodes from DROID
2. For each episode, computes the similarity score for each frame
3. Reports per-episode average scores and overall statistics
4. Optionally saves episodes as videos (all or top/bottom N)

Usage:
    # Single GPU
    python evaluate_episodes.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_dir /root/data \
        --num_episodes 100 \
        --seed 42
    
    # Multi-GPU (8x faster with 8 GPUs)
    python evaluate_episodes.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_dir /root/data \
        --num_episodes 100 \
        --num_gpus 8 \
        --save_videos_dir ./episode_videos \
        --save_top_bottom 5 \
        --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables before importing heavy libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Reuse the inference class from inference_droid_two_view.py
from inference_droid_two_view import VLA_SigLIP2_Droid, PADDING_VALUE
from model import ModelConfig

# Import RLDS dataset components
from openpi_rlds_two_view import (
    DROID_ACTION_Q01,
    DROID_ACTION_Q99,
    _normalize_actions_quantile,
    _compute_action_deltas,
    _to_uint8_hwc,
)


class EpisodeEvaluator:
    """Evaluator that computes frame-level scores for episodes."""
    
    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "hf-hub:timm/ViT-L-16-SigLIP2-384",
        action_chunk_size: int = 16,
        action_dim: int = 8,
        use_transformer: bool = True,
        use_delta_actions: bool = True,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.action_chunk_size = action_chunk_size
        self.history_length = 2 * action_chunk_size - 1
        self.action_dim = action_dim
        self.use_delta_actions = use_delta_actions
        
        print(f"Loading SigLIP2 model: {backbone}")
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        siglip2_model, self.preprocess = create_model_from_pretrained(backbone)
        siglip2_model = siglip2_model.to(self.device)
        self.tokenizer = get_tokenizer(backbone)
        self.context_length = siglip2_model.context_length
        
        # Initialize model
        model_config = ModelConfig(
            clip_model=siglip2_model,
            history_length=self.history_length,
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
        
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print("Model loaded successfully!")
    
    def compute_frame_score(
        self,
        ext_img: np.ndarray,
        wrist_img: np.ndarray,
        text: str,
        actions: np.ndarray,
    ) -> float:
        """
        Compute similarity score for a single frame.
        
        Args:
            ext_img: Exterior image (uint8 HWC)
            wrist_img: Wrist image (uint8 HWC)
            text: Instruction text
            actions: Action history [T, A]
        
        Returns:
            Similarity score (float)
        """
        with torch.no_grad():
            # Preprocess images
            ext_pil = Image.fromarray(ext_img)
            wrist_pil = Image.fromarray(wrist_img)
            ext_tensor = self.preprocess(ext_pil).unsqueeze(0).to(self.device)
            wrist_tensor = self.preprocess(wrist_pil).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_tokens = self.tokenizer([text], context_length=self.context_length).to(self.device)
            
            # Prepare actions
            actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get similarity score
            image_logits, _ = self.model(ext_tensor, wrist_tensor, text_tokens, actions_tensor)
            score = image_logits[0, 0].item()
        
        return score


def load_episodes_from_rlds(
    data_dir: str,
    num_episodes: int,
    action_chunk_size: int,
    use_delta_actions: bool,
    seed: int = 42,
    target_height: int = 224,
    target_width: int = 224,
) -> list[dict]:
    """
    Load episodes from DROID RLDS dataset.
    
    Returns list of episode dicts, each containing:
        - episode_id: str
        - instruction: str
        - frames: list of (ext_img, wrist_img, actions) tuples
    """
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import dlimp as dl
    
    np.random.seed(seed)
    
    history_length = 2 * action_chunk_size - 1
    
    print(f"Loading DROID dataset from {data_dir}...")
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=True, num_parallel_reads=4)
    
    # Filter for successful trajectories only
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
        )
    )
    
    # Take more than needed to ensure we get enough valid episodes
    dataset = dataset.take(num_episodes * 2)
    
    episodes = []
    
    print(f"Processing episodes...")
    for traj in tqdm(dataset, desc="Loading episodes", total=num_episodes * 2):
        if len(episodes) >= num_episodes:
            break
        
        traj_len = int(tf.shape(traj["action"])[0].numpy())
        if traj_len < history_length:
            continue
        
        # Extract instruction
        instruction = ""
        for lang_key in ["language_instruction", "language_instruction_2", "language_instruction_3"]:
            if lang_key in traj:
                lang_val = traj[lang_key].numpy()
                if isinstance(lang_val, bytes):
                    lang_val = lang_val.decode("utf-8")
                elif isinstance(lang_val, np.ndarray) and lang_val.size > 0:
                    lang_val = lang_val.flat[0]
                    if isinstance(lang_val, bytes):
                        lang_val = lang_val.decode("utf-8")
                if lang_val and str(lang_val).strip():
                    instruction = str(lang_val).strip()
                    break
        
        if not instruction:
            continue
        
        # Get episode metadata
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0].numpy()
        if isinstance(file_path, bytes):
            file_path = file_path.decode("utf-8")
        
        # Get actions and state
        actions_raw = traj["action_dict"]["joint_position"].numpy()  # [T, 7]
        gripper_raw = traj["action_dict"]["gripper_position"].numpy()  # [T, 1]
        actions_full = np.concatenate([actions_raw, gripper_raw], axis=-1)  # [T, 8]
        
        # Get state for delta actions
        joint_state = traj["observation"]["joint_position"].numpy()  # [T, 7]
        gripper_state = traj["observation"]["gripper_position"].numpy()  # [T, 1]
        state_full = np.concatenate([joint_state, gripper_state], axis=-1)  # [T, 8]
        
        # Get images (encoded)
        ext_images = traj["observation"]["exterior_image_1_left"].numpy()
        if ext_images is None or len(ext_images) == 0:
            ext_images = traj["observation"]["exterior_image_2_left"].numpy()
        wrist_images = traj["observation"]["wrist_image_left"].numpy()
        
        if ext_images is None or wrist_images is None:
            continue
        
        # Process frames
        frames = []
        for t in range(traj_len):
            # Build action chunk window centered at t
            # History: [t - (chunk_size-1), ..., t-1, t, t+1, ..., t + (chunk_size-1)]
            history_start = t - (action_chunk_size - 1)
            history_end = t + action_chunk_size
            
            # Clip indices and build action chunk
            action_chunk = []
            for i in range(history_start, history_end):
                if i < 0:
                    action_chunk.append(actions_full[0])  # Pad with first
                elif i >= traj_len:
                    action_chunk.append(actions_full[-1])  # Pad with last
                else:
                    action_chunk.append(actions_full[i])
            
            action_chunk = np.stack(action_chunk, axis=0).astype(np.float32)  # [2*chunk-1, 8]
            
            # Apply delta actions if enabled
            if use_delta_actions:
                state_t = state_full[t:t+1]  # [1, 8]
                action_chunk = _compute_action_deltas(
                    action_chunk[np.newaxis, ...],  # [1, T, 8]
                    state_t,  # [1, 8]
                )[0]  # [T, 8]
            
            # Normalize actions
            action_chunk = _normalize_actions_quantile(action_chunk)
            
            # Decode images
            try:
                ext_img = tf.io.decode_image(ext_images[t], expand_animations=False, dtype=tf.uint8).numpy()
                wrist_img = tf.io.decode_image(wrist_images[t], expand_animations=False, dtype=tf.uint8).numpy()
            except Exception:
                continue
            
            # Resize with aspect ratio preservation
            ext_img = _resize_image(ext_img, target_height, target_width)
            wrist_img = _resize_image(wrist_img, target_height, target_width)
            
            frames.append({
                'ext_img': ext_img,
                'wrist_img': wrist_img,
                'actions': action_chunk,
                'timestep': t,
            })
        
        if len(frames) < 5:  # Skip very short episodes
            continue
        
        episodes.append({
            'episode_id': file_path,
            'instruction': instruction,
            'frames': frames,
            'num_frames': len(frames),
        })
    
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def _resize_image(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize image with aspect ratio preservation and padding."""
    cur_height, cur_width = img.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    new_height = int(cur_height / ratio)
    new_width = int(cur_width / ratio)
    
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_width, new_height), Image.BILINEAR)
    resized = np.array(pil_img)
    
    pad_h = height - new_height
    pad_w = width - new_width
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if len(resized.shape) == 3:
        padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                       mode='constant', constant_values=0)
    else:
        padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                       mode='constant', constant_values=0)
    
    return padded


def save_episode_video(
    episode: dict,
    output_dir: str,
    episode_idx: int,
    avg_score: float = None,
    frame_scores: list[float] = None,
    fps: int = 15,
) -> None:
    """
    Save an episode as video files (exterior, wrist, and combined side-by-side).
    
    Args:
        episode: Episode dict with frames
        output_dir: Directory to save videos
        episode_idx: Index for naming
        avg_score: Optional average score to include in filename
        frame_scores: Optional list of scores for each frame to draw on video
        fps: Frames per second for video
    """
    import imageio.v2 as iio
    
    # Create safe filename from instruction
    instruction = episode['instruction']
    safe_instruction = "".join(c if c.isalnum() or c in ' _-' else '' for c in instruction)[:50]
    safe_instruction = safe_instruction.strip().replace(' ', '_')
    
    # Create episode directory
    if avg_score is not None:
        ep_dir = os.path.join(output_dir, f"episode_{episode_idx:02d}_{safe_instruction}_score_{avg_score:.2f}")
    else:
        ep_dir = os.path.join(output_dir, f"episode_{episode_idx:02d}_{safe_instruction}")
    os.makedirs(ep_dir, exist_ok=True)
    
    frames = episode['frames']
    
    # Collect frames
    ext_frames = []
    wrist_frames = []
    combined_frames = []
    
    for i, frame in enumerate(frames):
        ext_img = frame['ext_img']
        wrist_img = frame['wrist_img']
        
        # Mark score on exterior image if provided
        if frame_scores is not None and i < len(frame_scores) and avg_score is not None:
            score = frame_scores[i]
            # Red for scores below mean, green for scores above mean
            color = (0, 255, 0) if score >= avg_score else (255, 0, 0)
            
            # Use PIL to draw on image
            pil_img = Image.fromarray(ext_img)
            draw = ImageDraw.Draw(pil_img)
            try:
                # Try to find a standard font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), f"Score: {score:.2f}", fill=color, font=font)
            ext_img = np.array(pil_img)
        
        ext_frames.append(ext_img)
        wrist_frames.append(wrist_img)
        
        # Create side-by-side combined frame
        combined = np.concatenate([ext_img, wrist_img], axis=1)
        combined_frames.append(combined)
    
    # Save videos
    ext_path = os.path.join(ep_dir, "exterior.mp4")
    wrist_path = os.path.join(ep_dir, "wrist.mp4")
    combined_path = os.path.join(ep_dir, "combined.mp4")
    
    # Use imageio to write videos
    iio.mimwrite(ext_path, ext_frames, fps=fps, codec='libx264', pixelformat='yuv420p')
    iio.mimwrite(wrist_path, wrist_frames, fps=fps, codec='libx264', pixelformat='yuv420p')
    iio.mimwrite(combined_path, combined_frames, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    # Save metadata
    metadata = {
        'episode_id': episode['episode_id'],
        'instruction': instruction,
        'num_frames': len(frames),
        'avg_score': avg_score,
        'fps': fps,
    }
    meta_path = os.path.join(ep_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved episode {episode_idx}: {safe_instruction[:40]}... ({len(frames)} frames)")


def save_episodes_as_videos(
    episodes: list[dict],
    results: dict,
    output_dir: str,
    save_all: bool = False,
    save_top_bottom: int = 5,
    fps: int = 15,
) -> None:
    """
    Save episodes as videos.
    
    Args:
        episodes: List of episode dicts
        results: Evaluation results with scores
        output_dir: Directory to save videos
        save_all: If True, save all episodes
        save_top_bottom: Number of top and bottom episodes to save (ignored if save_all=True)
        fps: Frames per second
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get episode details sorted by score
    details = results['episode_details']
    sorted_indices = sorted(range(len(details)), key=lambda i: details[i]['avg_score'], reverse=True)
    
    if save_all:
        print(f"\nSaving all {len(episodes)} episodes as videos...")
        indices_to_save = list(range(len(episodes)))
    else:
        # Save top N and bottom N
        top_indices = sorted_indices[:save_top_bottom]
        bottom_indices = sorted_indices[-save_top_bottom:]
        indices_to_save = list(set(top_indices + bottom_indices))
        print(f"\nSaving top {save_top_bottom} and bottom {save_top_bottom} episodes as videos...")
    
    # Create subdirectories
    if not save_all:
        top_dir = os.path.join(output_dir, "top_episodes")
        bottom_dir = os.path.join(output_dir, "bottom_episodes")
        os.makedirs(top_dir, exist_ok=True)
        os.makedirs(bottom_dir, exist_ok=True)
    
    for rank, orig_idx in enumerate(sorted_indices):
        if orig_idx not in indices_to_save:
            continue
        
        episode = episodes[orig_idx]
        ep_details = details[orig_idx]
        avg_score = ep_details['avg_score']
        frame_scores = ep_details.get('frame_scores')
        
        if save_all:
            save_dir = output_dir
            ep_idx = rank
        else:
            # Determine if top or bottom
            if rank < save_top_bottom:
                save_dir = top_dir
                ep_idx = rank
            elif rank >= len(sorted_indices) - save_top_bottom:
                save_dir = bottom_dir
                ep_idx = rank - (len(sorted_indices) - save_top_bottom)
            else:
                continue
        
        save_episode_video(
            episode=episode,
            output_dir=save_dir,
            episode_idx=ep_idx,
            avg_score=avg_score,
            frame_scores=frame_scores,
            fps=fps,
        )


def evaluate_episodes(
    evaluator: EpisodeEvaluator,
    episodes: list[dict],
) -> dict:
    """
    Evaluate frame-level scores for each episode (single GPU).
    
    Args:
        evaluator: EpisodeEvaluator instance
        episodes: List of episode dicts
    
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'episode_scores': [],
        'all_frame_scores': [],
        'episode_details': [],
    }
    
    for ep_idx, episode in enumerate(tqdm(episodes, desc="Evaluating episodes")):
        instruction = episode['instruction']
        frames = episode['frames']
        
        frame_scores = []
        for frame in frames:
            score = evaluator.compute_frame_score(
                ext_img=frame['ext_img'],
                wrist_img=frame['wrist_img'],
                text=instruction,
                actions=frame['actions'],
            )
            frame_scores.append(score)
        
        episode_avg = np.mean(frame_scores)
        results['episode_scores'].append(episode_avg)
        results['all_frame_scores'].extend(frame_scores)
        results['episode_details'].append({
            'episode_id': episode['episode_id'],
            'instruction': instruction,
            'num_frames': len(frame_scores),
            'avg_score': float(episode_avg),
            'min_score': float(np.min(frame_scores)),
            'max_score': float(np.max(frame_scores)),
            'std_score': float(np.std(frame_scores)),
            'frame_scores': frame_scores,
        })
    
    # Compute overall statistics
    results['overall'] = {
        'num_episodes': len(episodes),
        'total_frames': len(results['all_frame_scores']),
        'avg_episode_score': float(np.mean(results['episode_scores'])),
        'std_episode_score': float(np.std(results['episode_scores'])),
        'min_episode_score': float(np.min(results['episode_scores'])),
        'max_episode_score': float(np.max(results['episode_scores'])),
        'median_episode_score': float(np.median(results['episode_scores'])),
        'avg_frame_score': float(np.mean(results['all_frame_scores'])),
        'std_frame_score': float(np.std(results['all_frame_scores'])),
    }
    
    return results


def _evaluate_episodes_on_gpu(args_tuple):
    """
    Worker function to evaluate episodes on a specific GPU.
    Called by multiprocessing pool.
    """
    (
        gpu_id,
        episode_indices,
        episodes_data,
        checkpoint_path,
        backbone,
        action_chunk_size,
        use_transformer,
        use_delta_actions,
    ) = args_tuple
    
    device = f"cuda:{gpu_id}"
    
    # Initialize evaluator on this GPU
    evaluator = EpisodeEvaluator(
        checkpoint_path=checkpoint_path,
        backbone=backbone,
        action_chunk_size=action_chunk_size,
        action_dim=8,
        use_transformer=use_transformer,
        use_delta_actions=use_delta_actions,
        device=device,
    )
    
    results = []
    for ep_idx in tqdm(episode_indices, desc=f"GPU {gpu_id}", position=gpu_id):
        episode = episodes_data[ep_idx]
        instruction = episode['instruction']
        frames = episode['frames']
        
        frame_scores = []
        for frame in frames:
            score = evaluator.compute_frame_score(
                ext_img=frame['ext_img'],
                wrist_img=frame['wrist_img'],
                text=instruction,
                actions=frame['actions'],
            )
            frame_scores.append(score)
        
        episode_avg = np.mean(frame_scores)
        results.append({
            'original_idx': ep_idx,
            'episode_id': episode['episode_id'],
            'instruction': instruction,
            'num_frames': len(frame_scores),
            'avg_score': float(episode_avg),
            'min_score': float(np.min(frame_scores)),
            'max_score': float(np.max(frame_scores)),
            'std_score': float(np.std(frame_scores)),
            'frame_scores': frame_scores,
        })
    
    return results


def evaluate_episodes_multi_gpu(
    episodes: list[dict],
    checkpoint_path: str,
    backbone: str,
    action_chunk_size: int,
    use_transformer: bool,
    use_delta_actions: bool,
    num_gpus: int = 8,
) -> dict:
    """
    Evaluate episodes using multiple GPUs in parallel.
    
    Args:
        episodes: List of episode dicts
        checkpoint_path: Path to model checkpoint
        backbone: SigLIP2 backbone model
        action_chunk_size: Action chunk size
        use_transformer: Use transformer encoder
        use_delta_actions: Use delta actions
        num_gpus: Number of GPUs to use
    
    Returns:
        Dictionary with evaluation results
    """
    num_episodes = len(episodes)
    available_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, available_gpus, num_episodes)
    
    print(f"\nUsing {num_gpus} GPUs for parallel evaluation...")
    
    # Distribute episodes across GPUs
    episode_indices = list(range(num_episodes))
    indices_per_gpu = [[] for _ in range(num_gpus)]
    for i, idx in enumerate(episode_indices):
        indices_per_gpu[i % num_gpus].append(idx)
    
    # Print distribution
    for gpu_id, indices in enumerate(indices_per_gpu):
        print(f"  GPU {gpu_id}: {len(indices)} episodes")
    
    # Prepare arguments for each GPU worker
    worker_args = []
    for gpu_id in range(num_gpus):
        worker_args.append((
            gpu_id,
            indices_per_gpu[gpu_id],
            episodes,
            checkpoint_path,
            backbone,
            action_chunk_size,
            use_transformer,
            use_delta_actions,
        ))
    
    # Run evaluation in parallel using multiprocessing
    all_results = []
    
    # Use spawn method for CUDA compatibility
    ctx = mp.get_context('spawn')
    with ctx.Pool(num_gpus) as pool:
        gpu_results = pool.map(_evaluate_episodes_on_gpu, worker_args)
    
    # Flatten results from all GPUs
    for gpu_result in gpu_results:
        all_results.extend(gpu_result)
    
    # Sort by original index to maintain order
    all_results.sort(key=lambda x: x['original_idx'])
    
    # Aggregate results
    results = {
        'episode_scores': [],
        'all_frame_scores': [],
        'episode_details': [],
    }
    
    for r in all_results:
        results['episode_scores'].append(r['avg_score'])
        results['all_frame_scores'].extend(r['frame_scores'])
        results['episode_details'].append({
            'episode_id': r['episode_id'],
            'instruction': r['instruction'],
            'num_frames': r['num_frames'],
            'avg_score': r['avg_score'],
            'min_score': r['min_score'],
            'max_score': r['max_score'],
            'std_score': r['std_score'],
            'frame_scores': r['frame_scores'],
        })
    
    # Compute overall statistics
    results['overall'] = {
        'num_episodes': len(episodes),
        'total_frames': len(results['all_frame_scores']),
        'avg_episode_score': float(np.mean(results['episode_scores'])),
        'std_episode_score': float(np.std(results['episode_scores'])),
        'min_episode_score': float(np.min(results['episode_scores'])),
        'max_episode_score': float(np.max(results['episode_scores'])),
        'median_episode_score': float(np.median(results['episode_scores'])),
        'avg_frame_score': float(np.mean(results['all_frame_scores'])),
        'std_frame_score': float(np.std(results['all_frame_scores'])),
        'num_gpus_used': num_gpus,
    }
    
    return results


def print_results(results: dict):
    """Print evaluation results."""
    overall = results['overall']
    
    print("\n" + "=" * 70)
    print("EPISODE EVALUATION RESULTS")
    print("=" * 70)
    print(f"Number of episodes: {overall['num_episodes']}")
    print(f"Total frames evaluated: {overall['total_frames']}")
    print("-" * 70)
    print("Per-Episode Score Statistics:")
    print(f"  Average: {overall['avg_episode_score']:.4f}")
    print(f"  Std Dev: {overall['std_episode_score']:.4f}")
    print(f"  Median:  {overall['median_episode_score']:.4f}")
    print(f"  Min:     {overall['min_episode_score']:.4f}")
    print(f"  Max:     {overall['max_episode_score']:.4f}")
    print("-" * 70)
    print("Per-Frame Score Statistics:")
    print(f"  Average: {overall['avg_frame_score']:.4f}")
    print(f"  Std Dev: {overall['std_frame_score']:.4f}")
    print("=" * 70)
    
    # Show distribution of episode scores
    scores = results['episode_scores']
    print("\nEpisode Score Distribution:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(scores, p)
        print(f"  {p}th percentile: {val:.4f}")
    
    # Show top and bottom episodes
    details = results['episode_details']
    sorted_details = sorted(details, key=lambda x: x['avg_score'], reverse=True)
    
    print("\nTop 5 Episodes (highest avg score):")
    for i, ep in enumerate(sorted_details[:5]):
        print(f"  {i+1}. Score={ep['avg_score']:.4f} ({ep['num_frames']} frames): {ep['instruction'][:60]}...")
    
    print("\nBottom 5 Episodes (lowest avg score):")
    for i, ep in enumerate(sorted_details[-5:]):
        print(f"  {i+1}. Score={ep['avg_score']:.4f} ({ep['num_frames']} frames): {ep['instruction'][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA-SigLIP2 DROID model with per-episode frame scores"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root/data",
        help="Parent directory of droid/ TFDS folder",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=0,
        help="Max frames to evaluate per episode (for speed). Set to 0 for all frames.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="hf-hub:timm/ViT-L-16-SigLIP2-384",
        help="SigLIP2 model backbone",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=16,
        help="Action chunk size (creates 2*N-1 action window)",
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
        "--use_delta_actions",
        action="store_true",
        default=True,
        help="Use delta actions (default: True)",
    )
    parser.add_argument(
        "--no_delta_actions",
        action="store_true",
        help="Disable delta actions",
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
    parser.add_argument(
        "--save_videos_dir",
        type=str,
        default=None,
        help="Optional: directory to save episode videos",
    )
    parser.add_argument(
        "--save_all_videos",
        action="store_true",
        help="Save all episodes as videos (default: only top/bottom)",
    )
    parser.add_argument(
        "--save_top_bottom",
        type=int,
        default=5,
        help="Number of top and bottom episodes to save as videos (default: 5)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=15,
        help="FPS for saved videos (default: 15)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel evaluation (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Handle flags
    use_transformer = args.use_transformer and not args.no_transformer
    use_delta_actions = args.use_delta_actions and not args.no_delta_actions
    max_frames = args.max_frames_per_episode if args.max_frames_per_episode > 0 else None
    
    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Load episodes from RLDS
    episodes = load_episodes_from_rlds(
        data_dir=args.data_dir,
        num_episodes=args.num_episodes,
        action_chunk_size=args.action_chunk_size,
        use_delta_actions=use_delta_actions,
        seed=args.seed,
    )
    
    if len(episodes) == 0:
        print("Error: No valid episodes loaded!")
        return
    
    # Uniformly sample frames per episode if requested
    if max_frames:
        print(f"Sampling max {max_frames} frames per episode...")
        for ep in episodes:
            if len(ep['frames']) > max_frames:
                indices = np.linspace(0, len(ep['frames']) - 1, max_frames, dtype=int)
                ep['frames'] = [ep['frames'][i] for i in indices]
                ep['num_frames'] = len(ep['frames'])
    
    # Run evaluation (single or multi-GPU)
    if args.num_gpus > 1:
        # Multi-GPU evaluation
        results = evaluate_episodes_multi_gpu(
            episodes=episodes,
            checkpoint_path=args.checkpoint,
            backbone=args.backbone,
            action_chunk_size=args.action_chunk_size,
            use_transformer=use_transformer,
            use_delta_actions=use_delta_actions,
            num_gpus=args.num_gpus,
        )
    else:
        # Single GPU evaluation
        evaluator = EpisodeEvaluator(
            checkpoint_path=args.checkpoint,
            backbone=args.backbone,
            action_chunk_size=args.action_chunk_size,
            action_dim=8,
            use_transformer=use_transformer,
            use_delta_actions=use_delta_actions,
        )
        results = evaluate_episodes(
            evaluator=evaluator,
            episodes=episodes,
        )
    
    # Print results
    print_results(results)
    
    # Optionally save to JSON
    if args.output_json:
        # Convert numpy types for JSON serialization
        results_json = {
            'overall': results['overall'],
            'episode_scores': [float(s) for s in results['episode_scores']],
            'episode_details': results['episode_details'],
            'config': {
                'checkpoint': args.checkpoint,
                'data_dir': args.data_dir,
                'num_episodes': args.num_episodes,
                'max_frames_per_episode': max_frames,
                'action_chunk_size': args.action_chunk_size,
                'use_transformer': use_transformer,
                'use_delta_actions': use_delta_actions,
                'seed': args.seed,
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nDetailed results saved to {args.output_json}")
    
    # Optionally save episodes as videos
    if args.save_videos_dir:
        save_episodes_as_videos(
            episodes=episodes,
            results=results,
            output_dir=args.save_videos_dir,
            save_all=args.save_all_videos,
            save_top_bottom=args.save_top_bottom,
            fps=args.video_fps,
        )
        print(f"\nVideos saved to {args.save_videos_dir}")


if __name__ == "__main__":
    main()

