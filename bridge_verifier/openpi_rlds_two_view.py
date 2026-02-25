"""
PyTorch-friendly wrapper for OpenPI's RLDS DROID loader (`DroidRldsDataset`).

Why this exists:
- `openpi.training.droid_verifier_rlds_dataset.DroidRldsDataset` yields *batched* numpy dicts via tf.data.
- Verifier training code in this folder typically expects batched torch tensors:
    (ext_img, wrist_img, texts, action_hists)

This wrapper:
- Converts numpy -> torch
- Optionally applies a CLIP-style `preprocess` (expects PIL images)
- Optionally shards a global batch across DDP ranks by striding indices
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Iterator, Literal

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# from openpi.training.droid_verifier_rlds_dataset import DroidActionSpace, DroidRldsDataset
from droid_verifier_rlds_dataset import DroidActionSpace, DroidRldsDataset

# DROID action normalization stats (from PI0 training)
# These are used for quantile normalization to [-1, 1]
DROID_ACTION_Q01 = np.array([
    -0.45159999, -0.79799998, -0.4384, -0.90880001,
    -0.634, -0.63279998, -0.75160003, 0.0
], dtype=np.float32)

DROID_ACTION_Q99 = np.array([
    0.43880001, 0.76440001, 0.44319999, 0.78600001,
    0.63800001, 0.65679997, 0.72439998, 0.9982
], dtype=np.float32)


def _to_uint8_hwc(images: np.ndarray) -> np.ndarray:
    """
    Ensure images are uint8 in HWC (or NHWC for batches) before resizing.
    Mirrors the conversion in `openpi/policies/droid_policy.py`.
    """
    arr = np.asarray(images)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (255 * arr).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:  # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 4 and arr.shape[1] == 3:  # NCHW -> NHWC
        arr = np.transpose(arr, (0, 2, 3, 1))
    return arr


def _normalize_actions_quantile(
    actions: np.ndarray,
    q01: np.ndarray = DROID_ACTION_Q01,
    q99: np.ndarray = DROID_ACTION_Q99,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """
    Normalize actions for DROID:
    - Dims 0-6 (joints): quantile normalization to [-1, 1] using q01/q99
    - Dim 7 (gripper): binarized to 0 or 1 (threshold at 0.5)
    
    Args:
        actions: float array of shape [..., action_dim] where action_dim >= 8
        q01: 1st percentile per action dim
        q99: 99th percentile per action dim
        gripper_threshold: threshold for binarizing gripper (default 0.5)
    
    Returns:
        Normalized actions: joints in [-1, 1], gripper in {0, 1}
    """
    action_dim = actions.shape[-1]
    result = actions.copy()
    
    # Normalize joints (dims 0-6) with quantile normalization
    joint_dims = min(7, action_dim)
    q01_joints = q01[:joint_dims]
    q99_joints = q99[:joint_dims]
    result[..., :joint_dims] = (actions[..., :joint_dims] - q01_joints) / (q99_joints - q01_joints + 1e-6) * 2.0 - 1.0
    
    # Binarize gripper (dim 7) to 0 or 1
    if action_dim > 7:
        result[..., 7] = (actions[..., 7] >= gripper_threshold).astype(np.float32)
    
    return result


def _compute_action_deltas(
    actions: np.ndarray,
    state: np.ndarray,
    delta_mask: tuple[bool, ...] = (True, True, True, True, True, True, True, False),
) -> np.ndarray:
    """
    Convert absolute actions to delta actions (action - state), matching OpenPI's DeltaActions transform.
    
    Directly copied from openpi.transforms.DeltaActions:
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    
    Args:
        actions: [B, T, A] absolute action trajectories
        state: [B, A] current robot state (joint positions + gripper) - broadcasted across T
        delta_mask: Which dimensions to convert to deltas (default: joints=True, gripper=False)
    
    Returns:
        [B, T, A] delta actions where delta[t] = action[t] - state (broadcasted)
    """
    actions = np.array(actions, copy=True, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(delta_mask, dtype=bool)
    dims = mask.shape[-1]
    
    # OpenPI's exact implementation: broadcast state and subtract where mask is True
    actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
    
    return actions


def _normalize_instruction(instr: str | None) -> str | None:
    if not instr or not isinstance(instr, str):
        return None
    s = instr.strip().lower()
    s = s.rstrip(".?!").strip()
    return s or None


def _load_instruction_mapping(path: str | None) -> dict[str, list[str]]:
    """Loads Bridge-style instruction rephrase mapping JSON."""
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    mapping: dict[str, list[str]] = {}
    if isinstance(data, dict):
        for entry in data.values():
            if isinstance(entry, dict) and "original" in entry and "rephrases" in entry:
                orig = _normalize_instruction(entry.get("original"))
                if not orig:
                    continue
                rephs = [
                    r.strip()
                    for r in entry.get("rephrases", [])
                    if isinstance(r, str) and r.strip() and _normalize_instruction(r)
                ]
                if rephs:
                    mapping[orig] = rephs
    return mapping


def _resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    """
    Resize to target size with aspect ratio preserved by padding.
    Mirrors vla-clip/bridge_verifier/droid_two_view_dataset.py.
    """
    if images.shape[-1] <= 4:  # assume channels-last
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
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
    padded_images = F.pad(resized_images, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=constant_value)

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # NCHW -> NHWC
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)

    return padded_images


def _resize_with_pad_numpy_batch(images_uint8_hwc: np.ndarray, height: int, width: int, mode: str = "bilinear") -> np.ndarray:
    """Batch version of resize_with_pad_numpy from droid_two_view_dataset.py."""
    t = torch.from_numpy(images_uint8_hwc)
    resized = _resize_with_pad_torch(t, height, width, mode=mode)
    return resized.cpu().numpy()


def _to_str_list(x) -> list[str]:
    """Converts a batch of TF/np strings/bytes to a python list[str]."""
    # Common cases:
    # - numpy array of dtype=object with bytes
    # - numpy array of dtype='|S..' or '<U..'
    # - python list of bytes/str
    if isinstance(x, (list, tuple)):
        xs = x
    else:
        xs = np.asarray(x).tolist()

    out: list[str] = []
    for v in xs:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return out


def _preprocess_pil_batch(preprocess: Callable, images_uint8_hwc: np.ndarray) -> torch.Tensor:
    """Applies a CLIP/SigLIP `preprocess` to a batch of uint8 HWC images."""
    # `preprocess` from open_clip typically expects PIL images and returns CHW float tensors.
    imgs = []
    for im in images_uint8_hwc:
        imgs.append(preprocess(Image.fromarray(im)))
    return torch.stack(imgs, dim=0)


@dataclass(frozen=True)
class OpenPiRldsTwoViewConfig:
    data_dir: str
    per_rank_batch_size: int
    action_chunk_size: int
    expected_action_dim: int = 8
    action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION
    shuffle: bool = True
    # Default to PI0's idle-frame filter (filters out ~74% of frames that are idle/uninteresting)
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    # Optional instruction rephrasing (Bridge-style mapping json).
    instruction_mapping_path: str | None = None
    instruction_mode: Literal["original", "random_rephrase", "all_rephrases"] = "all_rephrases"
    max_rephrases: int = 8  # Maximum number of rephrases to use (including original)
    # Used when instruction_mode == "all_rephrases".
    # Keeps batch size fixed by shuffling rephrase-expanded samples in a local buffer.
    rephrase_shuffle_buffer_size: int = 4096

    # Pre-resize to target size with aspect ratio preservation and padding,
    # before applying the model's internal preprocess.
    target_height: int = 224
    target_width: int = 224
    resize_mode: str = "bilinear"

    # Delta actions. If True, converts actions to deltas (action[t] - state[t]).
    # Applied BEFORE normalization (matching OpenPI). Helps with training stability by reducing action magnitudes.
    # Normalization is always applied after delta conversion.
    use_delta_actions: bool = False

    # DDP sharding (optional). If world_size>1, we load a global batch and stride-select the local batch.
    rank: int = 0
    world_size: int = 1

    # Optional image preprocessing (e.g., open_clip preprocess). If None, returns uint8 NHWC tensors.
    preprocess: Callable | None = None


class OpenPiRldsTwoViewBatchedIterable:
    """
    Iterable that yields batches:
      (ext_img, wrist_img, texts, action_hists)

    - ext_img / wrist_img:
        - if preprocess provided: float tensor [B, C, H, W]
        - else: uint8 tensor [B, H, W, C]
    - texts: list[str] length B
    - action_hists: float tensor [B, T, A]
    """

    def __init__(self, cfg: OpenPiRldsTwoViewConfig):
        self.cfg = cfg
        self._rephrases_map = _load_instruction_mapping(cfg.instruction_mapping_path)
        # IMPORTANT:
        # Building `DroidRldsDataset` is expensive (loads filter JSON, builds a tf.lookup table,
        # and sets up a tf.data pipeline with a large shuffle buffer). If we build it inside
        # `__iter__`, every call to `iter(dataloader)` rebuilds everything and reprints logs like
        # "Creating idle filter hash table...".
        #
        # We therefore build it ONCE per process here and reuse it across `__iter__` calls.
        self._world_size = max(1, int(self.cfg.world_size))
        self._rank = int(self.cfg.rank)

        self._ds = DroidRldsDataset(
            data_dir=self.cfg.data_dir,
            batch_size=int(self.cfg.per_rank_batch_size),
            shuffle=self.cfg.shuffle,
            action_chunk_size=self.cfg.action_chunk_size,
            action_space=self.cfg.action_space,
            filter_dict_path=self.cfg.filter_dict_path,
            rank=self._rank,
            world_size=self._world_size,
            target_height=self.cfg.target_height,
            target_width=self.cfg.target_width,
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]]:
        # Special case: `all_rephrases` should use *all* variants over time, but NOT expand within a single batch.
        # We implement this by expanding samples into a shuffle buffer and drawing fixed-size batches from it.
        if self.cfg.instruction_mode == "all_rephrases":
            buffer_size = max(int(self.cfg.rephrase_shuffle_buffer_size), int(self.cfg.per_rank_batch_size))
            # Each entry: {'ext': img, 'wrist': img, 'actions': arr, 'variants': list[str]}
            # All text variants stored; one picked randomly at batch sampling time
            rephrase_buffer: list[dict] = []
            ds_iter = iter(self._ds)

            def _process_base_batch(nxt):
                ext_b = nxt["observation"]["image"]
                wrist_b = nxt["observation"]["wrist_image"]
                texts_b = _to_str_list(nxt["prompt"])
                actions_b = nxt["actions"]
                
                # Extract state for delta actions
                joint_pos_b = nxt["observation"]["joint_position"]
                gripper_pos_b = nxt["observation"]["gripper_position"]
                state_b = np.concatenate([joint_pos_b, gripper_pos_b], axis=-1)

                got_action_dim = int(np.asarray(actions_b).shape[-1])
                if got_action_dim != int(self.cfg.expected_action_dim):
                    raise AssertionError(
                        f"Expected action_dim={self.cfg.expected_action_dim} but got {got_action_dim}. "
                        "Make sure your RLDS loader outputs only the first 8 action dims (or update expected_action_dim)."
                    )
                # Convert to delta actions FIRST (on raw data), then always normalize
                if self.cfg.use_delta_actions:
                    actions_b = _compute_action_deltas(actions_b, state_b)
                
                # Always apply normalization
                actions_b = _normalize_actions_quantile(actions_b)

                # Images are already resized and uint8 HWC from the TF pipeline
                return ext_b, wrist_b, texts_b, actions_b

            while True:
                nxt = next(ds_iter)
                
                processed = _process_base_batch(nxt)
                if processed is not None:
                    ext, wrist, texts, actions = processed
                    norm_texts = [_normalize_instruction(t) for t in texts]
                    for i, (t, nt) in enumerate(zip(texts, norm_texts, strict=False)):
                        variants = [t]
                        if nt and nt in self._rephrases_map:
                            variants.extend(self._rephrases_map[nt])
                        variants = variants[: self.cfg.max_rephrases]
                        # Store ALL variants - pick one randomly at batch sampling time
                        rephrase_buffer.append({
                            'ext': ext[i].copy(),
                            'wrist': wrist[i].copy(),
                            'actions': np.asarray(actions[i]).copy(),
                            'variants': variants,  # All text variants
                        })

                # Cap buffer size to avoid unbounded growth (drop oldest)
                if len(rephrase_buffer) > buffer_size:
                    overflow = len(rephrase_buffer) - buffer_size
                    if overflow > 0:
                        del rephrase_buffer[:overflow]
                
                # Need at least batch_size samples to yield a batch
                if len(rephrase_buffer) < int(self.cfg.per_rank_batch_size):
                    del nxt, processed # Explicitly clear
                    continue

                b = int(self.cfg.per_rank_batch_size)
                idxs = np.random.choice(len(rephrase_buffer), size=b, replace=False)
                chosen = [rephrase_buffer[i] for i in idxs.tolist()]
                for i in sorted(idxs.tolist(), reverse=True):
                    del rephrase_buffer[i]

                ext = np.stack([c['ext'] for c in chosen], axis=0)
                wrist = np.stack([c['wrist'] for c in chosen], axis=0)
                actions = np.stack([c['actions'] for c in chosen], axis=0)
                # Pick one random variant per sample at batch time
                texts = [str(np.random.choice(c['variants'])) for c in chosen]
                
                # Clear chosen list and its dicts
                del chosen

                if self.cfg.preprocess is not None:
                    ext_t = _preprocess_pil_batch(self.cfg.preprocess, ext)
                    wrist_t = _preprocess_pil_batch(self.cfg.preprocess, wrist)
                else:
                    ext_t = torch.from_numpy(ext)
                    wrist_t = torch.from_numpy(wrist)

                actions_t = torch.from_numpy(np.asarray(actions, dtype=np.float32))
                
                # Explicitly clear large numpy arrays
                del ext, wrist, actions, nxt, processed
                
                yield ext_t, wrist_t, texts, actions_t

        # Default path: original / random_rephrase (no within-batch expansion)
        while True:
            data_iter = iter(self._ds)
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # exhausted → restart outer loop

                # batch is a dict of numpy arrays (from tf.data.as_numpy_iterator)
                ext = batch["observation"]["image"]  # uint8 [B,H,W,C]
                wrist = batch["observation"]["wrist_image"]  # uint8 [B,H,W,C]
                texts = _to_str_list(batch["prompt"])
                actions = batch["actions"]  # float [B,T,A]
                
                # Extract state (joint_position + gripper_position) for delta actions
                joint_pos = batch["observation"]["joint_position"]  # float [B,T,7]
                gripper_pos = batch["observation"]["gripper_position"]  # float [B,T,1]
                state = np.concatenate([joint_pos, gripper_pos], axis=-1)  # [B,T,8]

                # Enforce expected action dimensionality (verifier assumes 8-D actions).
                # Note: action windows (T) can be any length; only the final dim (A) must match.
                got_action_dim = int(np.asarray(actions).shape[-1])
                if got_action_dim != int(self.cfg.expected_action_dim):
                    raise AssertionError(
                        f"Expected action_dim={self.cfg.expected_action_dim} but got {got_action_dim}. "
                        "Make sure your RLDS loader outputs only the first 8 action dims (or update expected_action_dim)."
                    )

                # Convert to delta actions FIRST (on raw data, matching OpenPI), then always normalize
                if self.cfg.use_delta_actions:
                    actions = _compute_action_deltas(actions, state)
                
                # Always apply quantile normalization (q01/q99 -> [-1,1], gripper binarized)
                actions = _normalize_actions_quantile(actions)

                # Images are already resized and uint8 HWC from the TF pipeline
                
                # Instruction rephrasing.
                # - original: no change
                # - random_rephrase: choose one variant per sample, keep batch size fixed
                if self.cfg.instruction_mode != "original":
                    norm_texts = [_normalize_instruction(t) for t in texts]
                    if self.cfg.instruction_mode == "random_rephrase":
                        # Keep batch size fixed: choose randomly among original + rephrases (if any).
                        new_texts: list[str] = []
                        for t, nt in zip(texts, norm_texts, strict=False):
                            if nt and nt in self._rephrases_map:
                                choices = [t, *self._rephrases_map[nt]]
                                new_texts.append(str(np.random.choice(choices)))
                            else:
                                new_texts.append(t)
                        texts = new_texts
                    else:
                        raise ValueError(f"Unknown instruction_mode: {self.cfg.instruction_mode}")

                # Convert images
                if self.cfg.preprocess is not None:
                    ext_t = _preprocess_pil_batch(self.cfg.preprocess, ext)
                    wrist_t = _preprocess_pil_batch(self.cfg.preprocess, wrist)
                else:
                    ext_t = torch.from_numpy(ext)  # uint8 NHWC
                    wrist_t = torch.from_numpy(wrist)

                actions_t = torch.from_numpy(np.asarray(actions, dtype=np.float32))
                
                # Explicitly clear large numpy arrays and dictionaries
                del ext, wrist, actions, batch, state, joint_pos, gripper_pos
                
                yield ext_t, wrist_t, texts, actions_t


