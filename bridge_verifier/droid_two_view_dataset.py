import os
import glob
import json
import re
from typing import Optional, Callable, Dict, Any, List, Tuple, Set

import numpy as np
import pandas as pd
from PIL import Image
import imageio.v2 as iio
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


PADDING_VALUE = -5.0
CAMERA_EXT = "observation.images.exterior_1_left"
CAMERA_WRIST = "observation.images.wrist_left"


def normalize_instruction(instr: Optional[str]) -> Optional[str]:
    if not instr or not isinstance(instr, str):
        return None
    instr = instr.strip().lower()
    instr = instr.rstrip(".?!").strip()
    return instr or None


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
    if not chunk_filter:
        return None
    parts = [p.strip() for p in chunk_filter.split(",") if p.strip()]
    indices: Set[int] = set()
    for p in parts:
        if p.isdigit():
            indices.add(int(p))
    return indices or None


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

    resized_images = torch.nn.functional.interpolate(
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
    padded_images = torch.nn.functional.pad(
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




class DroidPreprocessedDataset(torch.utils.data.Dataset):
    """
    Bridge-style normalized loader for preprocessed DROID (JSON + JPGs).
    Expects:
      - JSON with action_histories, instructions, samples, _metadata
      - images_folder containing agent_view_image_file JPGs (already concatenated two views)
    """

    def __init__(self, json_path: str, images_folder: Optional[str], preprocess, expected_history: Optional[int] = None):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.action_histories = data["action_histories"]
        self.instructions = data["instructions"]
        self.samples = data["samples"]
        self.meta = data.get("_metadata", {})
        self.preprocess = preprocess

        meta_hist = self.meta.get("total_window_size")
        self.history_length = expected_history or meta_hist
        if self.history_length is None:
            raise ValueError("history_length not provided and not found in metadata")

        meta_images = self.meta.get("images_folder")
        self.images_folder = images_folder or meta_images
        if self.images_folder is None:
            raise ValueError("images_folder not provided and not found in metadata")

        meta_action_dim = self.meta.get("action_dim")
        if meta_action_dim is not None:
            self.action_dim = int(meta_action_dim)
        else:
            first_hist = next(iter(self.action_histories.values()))
            self.action_dim = len(first_hist[0])

        self.padding_value = float(self.meta.get("padding_value", PADDING_VALUE))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        action_id = sample["action_history_id"]
        instr_id = sample["instruction_id"]
        img_file = sample["agent_view_image_file"]

        action_hist = np.array(self.action_histories[action_id], dtype=np.float32)
        if action_hist.shape[0] != self.history_length:
            raise ValueError(f"History length mismatch: {action_hist.shape[0]} vs expected {self.history_length}")

        img_path = os.path.join(self.images_folder, img_file)
        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)

        caption = self.instructions[instr_id]
        return img, caption, torch.from_numpy(action_hist)


class DroidStreamingDataset(IterableDataset):
    """
    Lazy streaming loader mirroring preprocess_droid_to_bridge.py without building an index.
    Iterates episodes and yields samples on the fly.
    """

    def __init__(
        self,
        dataset_root: str,
        preprocess,
        instruction_mapping_path: Optional[str],
        past_actions: int = 9,
        future_actions: int = 10,
        chunk_filter: Optional[str] = None,
        ffmpeg_threads: int = 0,
        target_height: Optional[int] = 224,
        target_width: Optional[int] = 224,
        resize_mode: str = "bilinear",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_root = dataset_root
        self.preprocess = preprocess
        self.past_actions = past_actions
        self.future_actions = future_actions
        self.history_length = past_actions + future_actions
        self.window_before = past_actions
        self.window_after = max(0, future_actions - 1)
        self.padding_value = PADDING_VALUE
        self.ffmpeg_threads = ffmpeg_threads
        self.reader_kwargs = {}
        if ffmpeg_threads and ffmpeg_threads > 0:
            self.reader_kwargs["ffmpeg_params"] = ["-threads", str(ffmpeg_threads)]
        self.target_height = target_height
        self.target_width = target_width
        self.resize_mode = resize_mode
        self.rank = rank
        self.world_size = max(1, world_size)
        self.is_lazy = True
        self.action_dim: Optional[int] = None

        self.rephrases_map = load_instruction_mapping(instruction_mapping_path)
        self.chunk_filter_set = parse_chunk_filter(chunk_filter)

        info_path = os.path.join(dataset_root, "meta", "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            self.fps = int(info.get("fps", 15))
            self.chunk_size = int(info.get("chunks_size", 1000))
        else:
            self.fps = 15
            self.chunk_size = 1000

        self.data_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

        self.episode_meta_paths = sorted(
            glob.glob(os.path.join(dataset_root, "meta", "episodes", "chunk-*", "file-*.parquet"))
        )
        self.episodes_jsonl_path = os.path.join(dataset_root, "meta", "episodes.jsonl")
        self.use_jsonl = False
        if not self.episode_meta_paths:
            if os.path.exists(self.episodes_jsonl_path):
                self.use_jsonl = True
            else:
                raise FileNotFoundError("No episode metadata found under meta/episodes or meta/episodes.jsonl.")

    def _get_reader_episode(self, cam: str, chunk_idx: int, episode_id: int):
        path = os.path.join(
            self.dataset_root,
            "videos",
            f"chunk-{chunk_idx:03d}",
            cam,
            f"episode_{episode_id:06d}.mp4",
        )
        if not os.path.exists(path):
            return None
        try:
            reader = iio.get_reader(path, **self.reader_kwargs)
        except (OSError, RuntimeError) as e:
            print(f"Warning: could not open video {path}: {e}")
            return None
        return reader

    def _load_data_episode(self, chunk_idx: int, episode_id: int) -> Optional[pd.DataFrame]:
        key = (chunk_idx, episode_id)
        if key in self.data_cache:
            return self.data_cache[key]
        data_path = os.path.join(
            self.dataset_root, "data", f"chunk-{chunk_idx:03d}", f"episode_{episode_id:06d}.parquet"
        )
        if not os.path.exists(data_path):
            return None
        df = pd.read_parquet(data_path)
        self.data_cache[key] = df
        return df

    def _iter_episode(self, chunk_idx: int, episode_id: int, df: Optional[pd.DataFrame]):
        if df is None or df.empty:
            return
        if "is_episode_successful" in df.columns:
            success = bool(df["is_episode_successful"].iloc[0])
            if not success:
                return

        actions = np.array([np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0) for a in df["action"].to_numpy()])
        frame_indices = (
            df["frame_index"].to_numpy()
            if "frame_index" in df.columns
            else np.arange(len(actions))
        )

        instr_col1 = df["language_instruction"] if "language_instruction" in df else []
        instr_text = None
        for val in instr_col1:
            if isinstance(val, str) and val.strip():
                instr_text = val
                break
        if instr_text is None:
            return

        action_dim = actions[0].shape[0]
        if self.action_dim is None:
            self.action_dim = action_dim

        ext_reader = self._get_reader_episode(CAMERA_EXT, chunk_idx, episode_id)
        wrist_reader = self._get_reader_episode(CAMERA_WRIST, chunk_idx, episode_id)
        if ext_reader is None or wrist_reader is None:
            return

        for idx_in_episode, frame_idx in enumerate(frame_indices):
            sample_id = (episode_id * 10_000_000) + int(frame_idx)
            if sample_id % self.world_size != self.rank:
                continue

            start = idx_in_episode - self.past_actions
            end = idx_in_episode + self.future_actions
            pad_before = max(0, -start)
            pad_after = max(0, end - len(actions))
            actual_start = max(0, start)
            actual_end = min(len(actions), end)
            actual_actions = actions[actual_start:actual_end]

            parts = []
            if pad_before > 0:
                parts.append(np.full((pad_before, action_dim), self.padding_value, dtype=np.float32))
            if len(actual_actions) > 0:
                parts.append(np.stack(actual_actions, axis=0))
            if pad_after > 0:
                parts.append(np.full((pad_after, action_dim), self.padding_value, dtype=np.float32))
            action_hist = np.concatenate(parts, axis=0).astype(np.float32)
            if action_hist.shape[0] != self.history_length:
                continue

            base_text = normalize_instruction(instr_text)
            if base_text is None:
                continue

            texts = [base_text]
            if base_text in self.rephrases_map:
                texts.extend([normalize_instruction(r) for r in self.rephrases_map[base_text] if normalize_instruction(r)])

            try:
                ext_frame = ext_reader.get_data(int(frame_idx))
            except Exception:
                continue
            try:
                wrist_frame = wrist_reader.get_data(int(frame_idx))
            except Exception:
                continue

            if self.target_height is not None and self.target_width is not None:
                ext_frame = resize_with_pad_numpy(ext_frame, self.target_height, self.target_width, mode=self.resize_mode)
                wrist_frame = resize_with_pad_numpy(wrist_frame, self.target_height, self.target_width, mode=self.resize_mode)

            ext_image = Image.fromarray(ext_frame)
            wrist_image = Image.fromarray(wrist_frame)
            ext_image = self.preprocess(ext_image)
            wrist_image = self.preprocess(wrist_image)

            for text in texts:
                yield ext_image, wrist_image, text, torch.from_numpy(action_hist)

        if hasattr(ext_reader, "close"):
            ext_reader.close()
        if hasattr(wrist_reader, "close"):
            wrist_reader.close()

    def __iter__(self):
        if self.use_jsonl:
            max_chunk = max(self.chunk_filter_set) if self.chunk_filter_set else None
            with open(self.episodes_jsonl_path, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    if "episode_index" not in rec:
                        continue
                    episode_id = int(rec["episode_index"])
                    chunk_idx = episode_id // self.chunk_size
                    if self.chunk_filter_set and chunk_idx not in self.chunk_filter_set:
                        if max_chunk is not None and chunk_idx > max_chunk:
                            break
                        continue
                    df = self._load_data_episode(chunk_idx, episode_id)
                    yield from self._iter_episode(chunk_idx, episode_id, df)
        else:
            for ep_meta_path in self.episode_meta_paths:
                if self.chunk_filter_set:
                    match = re.search(r"chunk-(\d+)", ep_meta_path)
                    if not match or int(match.group(1)) not in self.chunk_filter_set:
                        continue
                ep_df = pd.read_parquet(ep_meta_path)
                for _, ep_row in ep_df.iterrows():
                    episode_id = int(ep_row["episode_index"])
                    data_chunk = int(ep_row["data/chunk_index"])
                    data_file = int(ep_row["data/file_index"])
                    data_from = int(ep_row["dataset_from_index"])
                    data_to = int(ep_row["dataset_to_index"])
                    if self.chunk_filter_set and data_chunk not in self.chunk_filter_set:
                        continue
                    key = (data_chunk, data_file)
                    df = self.data_cache.get(key)
                    if df is None:
                        path = os.path.join(
                            self.dataset_root, "data", f"chunk-{data_chunk:03d}", f"file-{data_file:03d}.parquet"
                        )
                        if not os.path.exists(path):
                            continue
                        df = pd.read_parquet(path)
                        self.data_cache[key] = df
                    ep_slice = df.iloc[data_from:data_to]
                    yield from self._iter_episode(data_chunk, episode_id, ep_slice)


