#!/usr/bin/env python3
"""
Preprocess DROID (parquet + mp4) into a Bridge-style normalized JSON and JPGs.

Updated for droid_1.0.1 structure:
- Reads episodes via `meta/episodes/chunk-*/file-*.parquet`
- Loads step data from `data/chunk-XXX/file-XXX.parquet` (uses `dataset_from_index` / `dataset_to_index`)
- Loads two views (observation.images.exterior_1_left + observation.images.wrist_left),
  concatenates horizontally, saves JPG
- Uses per-step language fields when available; otherwise falls back to task text from `meta/tasks.parquet`
- Builds action windows using explicit past + future counts with padding (-5.0)
- Deduplicates action histories, builds instructions + optional rephrases (Bridge-format instruction_mapping.json)
- Outputs:
    action_histories: {id: list[list[float]]}
    instructions: {id: text}
    samples: [{action_history_id, instruction_id, agent_view_image_file, episode_id, timestep}]
    _metadata: action_dim, window_before/after, total_window_size, counts, padding_value, format_version
"""

import argparse
import glob
import json
import os
import re
import hashlib
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import imageio.v2 as iio
from PIL import Image
from tqdm import tqdm


PADDING_VALUE = -5.0

CAMERA_EXT = "observation.images.exterior_1_left"
CAMERA_WRIST = "observation.images.wrist_left"


def normalize_instruction(instr: Optional[str]) -> Optional[str]:
    if not instr or not isinstance(instr, str):
        return None
    instr = instr.strip().lower()
    instr = re.sub(r"[.?!]+$", "", instr).strip()
    return instr or None


def load_tasks_map(tasks_path: str) -> Dict[int, str]:
    """Return mapping task_index -> normalized text (if available)."""
    mapping: Dict[int, str] = {}
    parquet_path = tasks_path
    jsonl_path = tasks_path.replace(".parquet", ".jsonl")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path).reset_index()
        for _, row in df.iterrows():
            text = row.get("index", None)
            norm = normalize_instruction(text) if isinstance(text, str) else None
            if norm is not None:
                mapping[int(row["task_index"])] = norm
        return mapping

    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                row = json.loads(line)
                if "task_index" not in row:
                    continue
                text = row.get("task")
                norm = normalize_instruction(text) if isinstance(text, str) else None
                if norm is not None:
                    mapping[int(row["task_index"])] = norm
        return mapping

    return {}


def first_non_empty(series: pd.Series) -> Optional[str]:
    for val in series:
        if isinstance(val, str) and val.strip():
            return normalize_instruction(val)
    return None


def load_instruction_mapping(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    mapping = {}
    if isinstance(data, dict):
        for entry in data.values():
            if isinstance(entry, dict) and "original" in entry and "rephrases" in entry:
                orig = normalize_instruction(entry["original"])
                if not orig:
                    continue
                rephs = [r.strip() for r in entry["rephrases"] if isinstance(r, str) and r.strip()]
                if rephs:
                    mapping[orig] = rephs
    return mapping


def parse_chunk_filter(chunk_filter: Optional[str]) -> Optional[Set[int]]:
    """
    Parse comma-separated chunk indices (e.g., "0,1") into a set of ints.
    Returns None if no filter provided or parsing yields no indices.
    """
    if not chunk_filter:
        return None
    parts = [p.strip() for p in chunk_filter.split(",") if p.strip()]
    indices: Set[int] = set()
    for p in parts:
        if p.isdigit():
            indices.add(int(p))
    return indices or None


def hash_action_history(action_hist: np.ndarray) -> str:
    return hashlib.md5(action_hist.tobytes()).hexdigest()


def resize_with_pad_torch(
    images: torch.Tensor, height: int, width: int, mode: str = "bilinear"
) -> torch.Tensor:
    """
    Resize to target size with aspect ratio preserved by padding. If float,
    expects range [-1, 1].
    """
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=constant_value
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)

    return padded_images


def resize_with_pad_numpy(image: np.ndarray, height: int, width: int, mode: str = "bilinear") -> np.ndarray:
    tensor = torch.from_numpy(image)
    resized = resize_with_pad_torch(tensor, height, width, mode=mode)
    return resized.cpu().numpy()


def save_image_pair(
    ext_frame: np.ndarray,
    wrist_frame: np.ndarray,
    out_path: str,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
    resize_mode: str = "bilinear",
):
    if ext_frame.dtype != np.uint8:
        ext_frame = ext_frame.astype(np.uint8)
    if wrist_frame.dtype != np.uint8:
        wrist_frame = wrist_frame.astype(np.uint8)
    concat = np.concatenate([ext_frame, wrist_frame], axis=1)
    if target_height is not None and target_width is not None:
        concat = resize_with_pad_numpy(concat, target_height, target_width, mode=resize_mode)
    img = Image.fromarray(concat)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "JPEG", quality=95)


def preprocess(
    dataset_root: str,
    output_json: str,
    images_folder: str,
    past_actions: int,
    future_actions_including_current: int,
    instruction_mapping_path: Optional[str],
    chunk_filter: Optional[str],
    ffmpeg_threads: int,
    target_height: Optional[int],
    target_width: Optional[int],
    resize_mode: str,
):
    rephrases_map = load_instruction_mapping(instruction_mapping_path)
    default_instruction = next(iter(rephrases_map.keys()), "instruction") if rephrases_map else "instruction"
    chunk_filter_set = parse_chunk_filter(chunk_filter)
    reader_kwargs = {}
    if ffmpeg_threads and ffmpeg_threads > 0:
        reader_kwargs["ffmpeg_params"] = ["-threads", str(ffmpeg_threads)]

    info_path = os.path.join(dataset_root, "meta", "info.json")
    fps = 15
    chunk_size = 1000
    total_episodes = None
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        fps = int(info.get("fps", fps))
        chunk_size = int(info.get("chunks_size", chunk_size))
        total_episodes = info.get("total_episodes")
    tasks_map = load_tasks_map(os.path.join(dataset_root, "meta", "tasks.parquet"))

    # Window definition: collect `past_actions` steps before current, and
    # `future_actions_including_current` steps starting at current (current + future).
    if past_actions < 0 or future_actions_including_current <= 0:
        raise ValueError("past_actions must be >=0 and future_actions_including_current must be >0")
    history_length = past_actions + future_actions_including_current
    window_before = past_actions
    window_after = max(0, future_actions_including_current - 1)

    action_histories: Dict[str, List[List[float]]] = {}
    instructions: Dict[str, str] = {}
    instruction_to_id: Dict[str, str] = {}
    samples: List[Dict[str, str]] = []
    action_hash_to_id: Dict[str, str] = {}

    next_action_id = 0
    next_instr_id = 0
    total_images = 0

    episode_meta_paths = sorted(
        glob.glob(os.path.join(dataset_root, "meta", "episodes", "chunk-*", "file-*.parquet"))
    )
    if chunk_filter_set:
        filtered_paths = []
        for path in episode_meta_paths:
            match = re.search(r"chunk-(\d+)", path)
            if match and int(match.group(1)) in chunk_filter_set:
                filtered_paths.append(path)
        episode_meta_paths = filtered_paths
    episodes_jsonl_path = os.path.join(dataset_root, "meta", "episodes.jsonl")
    use_jsonl = False
    if not episode_meta_paths:
        if os.path.exists(episodes_jsonl_path):
            use_jsonl = True
        else:
            print("No episode metadata found under meta/episodes or meta/episodes.jsonl.")
            return

    # Cache dataframes and video readers per (chunk, file)
    data_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
    video_cache: Dict[Tuple[str, int, int], Any] = {}
    video_cache_episode: Dict[Tuple[str, int, int], Any] = {}

    def load_data(chunk_idx: int, file_idx: int) -> pd.DataFrame:
        key = (chunk_idx, file_idx)
        if key in data_cache:
            return data_cache[key]
        path = os.path.join(dataset_root, "data", f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")
        df = pd.read_parquet(path)
        data_cache[key] = df
        return df

    def get_reader(cam: str, chunk_idx: int, file_idx: int) -> Optional[Any]:
        key = (cam, chunk_idx, file_idx)
        if key in video_cache:
            return video_cache[key]
        path = os.path.join(
            dataset_root,
            "videos",
            cam,
            f"chunk-{chunk_idx:03d}",
            f"file-{file_idx:03d}.mp4",
        )
        if not os.path.exists(path):
            return None
        try:
            reader = iio.get_reader(path, **reader_kwargs)
        except (OSError, RuntimeError) as e:
            print(f"Warning: could not open video {path}: {e}")
            return None
        video_cache[key] = reader
        return reader

    def get_reader_episode(cam: str, chunk_idx: int, episode_id: int) -> Optional[Any]:
        key = (cam, chunk_idx, episode_id)
        if key in video_cache_episode:
            return video_cache_episode[key]
        path = os.path.join(
            dataset_root,
            "videos",
            f"chunk-{chunk_idx:03d}",
            cam,
            f"episode_{episode_id:06d}.mp4",
        )
        if not os.path.exists(path):
            return None
        try:
            reader = iio.get_reader(path, **reader_kwargs)
        except (OSError, RuntimeError) as e:
            print(f"Warning: could not open video {path}: {e}")
            return None
        video_cache_episode[key] = reader
        return reader

    if use_jsonl:
        max_chunk = max(chunk_filter_set) if chunk_filter_set else None
        tqdm_total = chunk_size * len(chunk_filter_set) if chunk_filter_set else total_episodes
        with open(episodes_jsonl_path, "r") as f:
            for line in tqdm(f, desc="Episodes", total=tqdm_total):
                rec = json.loads(line)
                if "episode_index" not in rec:
                    continue
                episode_id = int(rec["episode_index"])
                chunk_idx = episode_id // chunk_size
                if chunk_filter_set and chunk_idx not in chunk_filter_set:
                    if max_chunk is not None and chunk_idx > max_chunk:
                        break
                    continue
                data_path = os.path.join(
                    dataset_root, "data", f"chunk-{chunk_idx:03d}", f"episode_{episode_id:06d}.parquet"
                )
                if not os.path.exists(data_path):
                    print(f"Warning: data parquet missing for episode {episode_id} at {data_path}")
                    continue
                ep_slice = pd.read_parquet(data_path)
                if ep_slice.empty:
                    continue

                # Skip unsuccessful episodes if flag exists
                if "is_episode_successful" in ep_slice.columns:
                    success = bool(ep_slice["is_episode_successful"].iloc[0])
                    if not success:
                        continue

                actions = np.array([np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0) for a in ep_slice["action"].to_numpy()])
                frame_indices = (
                    ep_slice["frame_index"].to_numpy()
                    if "frame_index" in ep_slice.columns
                    else np.arange(len(actions))
                )
                task_indices = ep_slice["task_index"].to_numpy() if "task_index" in ep_slice.columns else []
                instr_col1 = ep_slice["language_instruction"] if "language_instruction" in ep_slice else []

                action_dim = actions[0].shape[0]

                instr_text = first_non_empty(instr_col1)
                if instr_text is None:
                    continue

                if instr_text in instruction_to_id:
                    instr_id = instruction_to_id[instr_text]
                else:
                    instr_id = f"instr_{next_instr_id}"
                    next_instr_id += 1
                    instruction_to_id[instr_text] = instr_id
                    instructions[instr_id] = instr_text

                wrist_reader = get_reader_episode(CAMERA_WRIST, chunk_idx, episode_id)
                ext_reader = get_reader_episode(CAMERA_EXT, chunk_idx, episode_id)
                if wrist_reader is None or ext_reader is None:
                    continue

                wrist_offset = 0
                ext_offset = 0

                for idx_in_episode, frame_idx in enumerate(frame_indices):
                    start = idx_in_episode - past_actions
                    end = idx_in_episode + future_actions_including_current
                    pad_before = max(0, -start)
                    pad_after = max(0, end - len(actions))
                    actual_start = max(0, start)
                    actual_end = min(len(actions), end)
                    actual_actions = actions[actual_start:actual_end]

                    parts = []
                    if pad_before > 0:
                        parts.append(np.full((pad_before, action_dim), PADDING_VALUE, dtype=np.float32))
                    if len(actual_actions) > 0:
                        parts.append(np.stack(actual_actions, axis=0))
                    if pad_after > 0:
                        parts.append(np.full((pad_after, action_dim), PADDING_VALUE, dtype=np.float32))
                    action_hist = np.concatenate(parts, axis=0).astype(np.float32)
                    if action_hist.shape[0] != history_length:
                        continue

                    action_hash = hash_action_history(action_hist)
                    if action_hash in action_hash_to_id:
                        action_id = action_hash_to_id[action_hash]
                    else:
                        action_id = f"action_{next_action_id}"
                        next_action_id += 1
                        action_hash_to_id[action_hash] = action_id
                        action_histories[action_id] = action_hist.tolist()

                    wrist_frame_idx = wrist_offset + int(frame_idx)
                    ext_frame_idx = ext_offset + int(frame_idx)
                    ext_frame = ext_reader.get_data(ext_frame_idx)
                    wrist_frame = wrist_reader.get_data(wrist_frame_idx)

                    img_filename = f"{episode_id:06d}_{int(frame_idx):06d}.jpg"
                    img_path = os.path.join(images_folder, img_filename)
                    save_image_pair(
                        ext_frame,
                        wrist_frame,
                        img_path,
                        target_height=target_height,
                        target_width=target_width,
                        resize_mode=resize_mode,
                    )
                    total_images += 1

                    samples.append(
                        {
                            "action_history_id": action_id,
                            "agent_view_image_file": img_filename,
                            "instruction_id": instr_id,
                            "episode_id": episode_id,
                            "timestep": int(frame_idx),
                        }
                    )

                    base_text = instructions[instr_id]
                    if base_text in rephrases_map:
                        for reph in rephrases_map[base_text]:
                            norm_reph = normalize_instruction(reph)
                            if not norm_reph:
                                continue
                            if norm_reph in instruction_to_id:
                                reph_id = instruction_to_id[norm_reph]
                            else:
                                reph_id = f"instr_{next_instr_id}"
                                next_instr_id += 1
                                instruction_to_id[norm_reph] = reph_id
                                instructions[reph_id] = norm_reph
                            samples.append(
                                {
                                    "action_history_id": action_id,
                                    "agent_view_image_file": img_filename,
                                    "instruction_id": reph_id,
                                    "episode_id": episode_id,
                                    "timestep": int(frame_idx),
                                }
                            )
    else:
        pbar_total = total_episodes if not chunk_filter_set else None
        pbar = tqdm(total=pbar_total, desc="Episodes")
        for ep_meta_path in episode_meta_paths:
            if chunk_filter_set:
                match = re.search(r"chunk-(\d+)", ep_meta_path)
                if not match or int(match.group(1)) not in chunk_filter_set:
                    continue
            ep_df = pd.read_parquet(ep_meta_path)

            for _, ep_row in ep_df.iterrows():
                episode_id = int(ep_row["episode_index"])
                data_chunk = int(ep_row["data/chunk_index"])
                data_file = int(ep_row["data/file_index"])
                data_from = int(ep_row["dataset_from_index"])
                data_to = int(ep_row["dataset_to_index"])
                if chunk_filter_set and data_chunk not in chunk_filter_set:
                    continue
                pbar.update(1)

                df = load_data(data_chunk, data_file)

                ep_slice = df.iloc[data_from:data_to]
                if ep_slice.empty:
                    continue

                actions = np.array([np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0) for a in ep_slice["action"].to_numpy()])
                frame_indices = ep_slice["frame_index"].to_numpy()
                task_indices = ep_slice["task_index"].to_numpy()
                instr_col1 = ep_slice["language_instruction"]

                action_dim = actions[0].shape[0]

                # Instruction resolution: language fields first, then task mapping, else default
                instr_text = first_non_empty(instr_col1)
                if instr_text is None:
                    continue

                if instr_text in instruction_to_id:
                    instr_id = instruction_to_id[instr_text]
                else:
                    instr_id = f"instr_{next_instr_id}"
                    next_instr_id += 1
                    instruction_to_id[instr_text] = instr_id
                    instructions[instr_id] = instr_text

                wrist_chunk = int(ep_row[f"videos/{CAMERA_WRIST}/chunk_index"])
                wrist_file = int(ep_row[f"videos/{CAMERA_WRIST}/file_index"])
                wrist_from_ts = float(ep_row[f"videos/{CAMERA_WRIST}/from_timestamp"])

                ext_chunk = int(ep_row[f"videos/{CAMERA_EXT}/chunk_index"])
                ext_file = int(ep_row[f"videos/{CAMERA_EXT}/file_index"])
                ext_from_ts = float(ep_row[f"videos/{CAMERA_EXT}/from_timestamp"])

                wrist_reader = get_reader(CAMERA_WRIST, wrist_chunk, wrist_file)
                ext_reader = get_reader(CAMERA_EXT, ext_chunk, ext_file)
                if wrist_reader is None or ext_reader is None:
                    continue

                wrist_offset = int(round(wrist_from_ts * fps))
                ext_offset = int(round(ext_from_ts * fps))

                for idx_in_episode, frame_idx in enumerate(frame_indices):
                    start = idx_in_episode - past_actions
                    end = idx_in_episode + future_actions_including_current
                    pad_before = max(0, -start)
                    pad_after = max(0, end - len(actions))
                    actual_start = max(0, start)
                    actual_end = min(len(actions), end)
                    actual_actions = actions[actual_start:actual_end]

                    parts = []
                    if pad_before > 0:
                        parts.append(np.full((pad_before, action_dim), PADDING_VALUE, dtype=np.float32))
                    if len(actual_actions) > 0:
                        parts.append(np.stack(actual_actions, axis=0))
                    if pad_after > 0:
                        parts.append(np.full((pad_after, action_dim), PADDING_VALUE, dtype=np.float32))
                    action_hist = np.concatenate(parts, axis=0).astype(np.float32)
                    if action_hist.shape[0] != history_length:
                        continue

                    action_hash = hash_action_history(action_hist)
                    if action_hash in action_hash_to_id:
                        action_id = action_hash_to_id[action_hash]
                    else:
                        action_id = f"action_{next_action_id}"
                        next_action_id += 1
                        action_hash_to_id[action_hash] = action_id
                        action_histories[action_id] = action_hist.tolist()

                    # Save concatenated image
                    wrist_frame_idx = wrist_offset + int(frame_idx)
                    ext_frame_idx = ext_offset + int(frame_idx)
                    ext_frame = ext_reader.get_data(ext_frame_idx)
                    wrist_frame = wrist_reader.get_data(wrist_frame_idx)

                    img_filename = f"{episode_id:06d}_{int(frame_idx):06d}.jpg"
                    img_path = os.path.join(images_folder, img_filename)
                    save_image_pair(
                        ext_frame,
                        wrist_frame,
                        img_path,
                        target_height=target_height,
                        target_width=target_width,
                        resize_mode=resize_mode,
                    )
                    total_images += 1

                    samples.append(
                        {
                            "action_history_id": action_id,
                            "agent_view_image_file": img_filename,
                            "instruction_id": instr_id,
                            "episode_id": episode_id,
                            "timestep": int(frame_idx),
                        }
                    )

                    # Rephrases
                    base_text = instructions[instr_id]
                    if base_text in rephrases_map:
                        for reph in rephrases_map[base_text]:
                            norm_reph = normalize_instruction(reph)
                            if not norm_reph:
                                continue
                            if norm_reph in instruction_to_id:
                                reph_id = instruction_to_id[norm_reph]
                            else:
                                reph_id = f"instr_{next_instr_id}"
                                next_instr_id += 1
                                instruction_to_id[norm_reph] = reph_id
                                instructions[reph_id] = norm_reph
                            samples.append(
                                {
                                    "action_history_id": action_id,
                                    "agent_view_image_file": img_filename,
                                    "instruction_id": reph_id,
                                    "episode_id": episode_id,
                                    "timestep": int(frame_idx),
                                }
                            )
        pbar.close()

    total_unique_action_histories = len(action_histories)
    total_samples = len(samples)
    total_instructions = len(instructions)
    compression_ratio = (
        total_samples / total_unique_action_histories if total_unique_action_histories > 0 else 0.0
    )

    meta = {
        "action_dim": next(iter(action_histories.values())) and len(next(iter(action_histories.values()))[0])
        if action_histories
        else None,
        "window_before": window_before,
        "window_after": window_after,
        "total_window_size": history_length,
        "total_images": total_images,
        "images_folder": images_folder,
        "total_instructions": total_instructions,
        "total_samples": total_samples,
        "total_unique_action_histories": total_unique_action_histories,
        "compression_ratio": compression_ratio,
        "format_version": "droid_normalized_v1",
        "padding_value": PADDING_VALUE,
        "image_height": target_height,
        "image_width": target_width,
        "resize_mode": resize_mode,
    }

    out = {
        "action_histories": action_histories,
        "instructions": instructions,
        "samples": samples,
        "_metadata": meta,
    }

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved normalized dataset to {output_json}")
    print(f"Images saved under {images_folder}")
    print(f"Samples: {total_samples}, Instructions: {total_instructions}, Action histories: {total_unique_action_histories}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DROID (droid_1.0.1) into Bridge-style normalized JSON + JPGs")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of droid_1.0.1 dataset")
    parser.add_argument("--output_json", type=str, default="droid_normalized.json", help="Output normalized JSON path")
    parser.add_argument("--images_folder", type=str, default="droid_images", help="Folder to save concatenated JPGs")
    parser.add_argument(
        "--instruction_mapping", type=str, default=None, help="Bridge-format instruction_mapping.json (optional rephrases)"
    )
    parser.add_argument(
        "--chunk_filter",
        type=str,
        default=None,
        help="Comma-separated chunk indices to process (e.g., '0' or '0,1'). If omitted, all chunks are processed.",
    )
    parser.add_argument(
        "--ffmpeg_threads",
        type=int,
        default=0,
        help="Number of threads for ffmpeg video decoding (0 = ffmpeg default).",
    )
    parser.add_argument(
        "--past_actions",
        type=int,
        default=9,
        help="Number of past actions (before current) to include in each window",
    )
    parser.add_argument(
        "--future_actions",
        type=int,
        default=10,
        help="Number of actions starting at current step to include (includes current)",
    )
    parser.add_argument("--target_height", type=int, default=224, help="Optional resize+pad height before saving images")
    parser.add_argument("--target_width", type=int, default=224, help="Optional resize+pad width before saving images")
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="bilinear",
        help="Interpolation mode used when --target_height/--target_width are set",
    )

    args = parser.parse_args()

    preprocess(
        dataset_root=args.dataset_root,
        output_json=args.output_json,
        images_folder=args.images_folder,
        past_actions=args.past_actions,
        future_actions_including_current=args.future_actions,
        instruction_mapping_path=args.instruction_mapping,
        chunk_filter=args.chunk_filter,
        ffmpeg_threads=args.ffmpeg_threads,
        target_height=args.target_height,
        target_width=args.target_width,
        resize_mode=args.resize_mode,
    )

