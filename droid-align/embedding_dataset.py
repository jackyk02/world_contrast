"""
PyTorch-compatible datasets for loading precomputed SigLIP2 embedding TFRecords.

Two dataset modes:
  MultiViewDataset   – yields (ext1, ext2, wrist) triplets from the same timestep.
                       Used for multi-view camera alignment training.

  TemporalDataset    – yields (lang, ext1_t, ext1_{t+k}) tuples.
                       Used for temporal / subgoal alignment training.

Both classes are IterableDatasets backed by tf.data for efficient streaming.
They support DDP sharding: pass rank / world_size to shard the shard files.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


# ---------------------------------------------------------------------------
# TFRecord parsing constants (must match precompute_embeddings.py)
# ---------------------------------------------------------------------------
EMBED_DIM = 1024

FEATURE_SPEC = {
    "traj_id":        tf.io.FixedLenFeature([], tf.string),
    "num_steps":      tf.io.FixedLenFeature([], tf.int64),
    "embed_dim":      tf.io.FixedLenFeature([], tf.int64),
    "cameras_avail":  tf.io.FixedLenFeature([], tf.int64),
    "ext1_emb_bytes": tf.io.FixedLenFeature([], tf.string),
    "ext2_emb_bytes": tf.io.FixedLenFeature([], tf.string),
    "wrist_emb_bytes":tf.io.FixedLenFeature([], tf.string),
    "lang_emb_bytes": tf.io.FixedLenFeature([], tf.string),
}


def _decode_traj(serialized: bytes, embed_dim: int = EMBED_DIM) -> dict | None:
    """
    Parse one serialised trajectory proto into numpy arrays.

    Returns None if parsing fails or if trajectory has 0 steps.
    """
    try:
        ex = tf.io.parse_single_example(serialized, FEATURE_SPEC)
    except Exception:
        return None

    T = int(ex["num_steps"].numpy())
    D = int(ex["embed_dim"].numpy())
    if T == 0 or D == 0:
        return None

    def _mat(key):
        raw = ex[key].numpy()
        return np.frombuffer(raw, dtype=np.float16).reshape(T, D).copy()

    def _vec(key):
        raw = ex[key].numpy()
        return np.frombuffer(raw, dtype=np.float16).reshape(D).copy()

    return {
        "traj_id":       ex["traj_id"].numpy().decode("utf-8", errors="replace"),
        "T":             T,
        "D":             D,
        "cameras_avail": int(ex["cameras_avail"].numpy()),
        "ext1":          _mat("ext1_emb_bytes"),   # [T, D] float16
        "ext2":          _mat("ext2_emb_bytes"),   # [T, D] float16
        "wrist":         _mat("wrist_emb_bytes"),  # [T, D] float16
        "lang":          _vec("lang_emb_bytes"),   # [D]    float16
    }


def _to_float32_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert float16 numpy array to float32 torch tensor."""
    return torch.from_numpy(arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Multi-view dataset
# ---------------------------------------------------------------------------
class MultiViewDataset(IterableDataset):
    """
    Iterates over precomputed embedding shards and yields
    (ext1_emb, ext2_emb, wrist_emb) float32 tensors for the *same* timestep.

    Trajectories where a camera is absent (cameras_avail bitmask) are skipped
    by default (require_all_cameras=True).

    Args:
        embedding_dir:        Directory containing .tfrecord files.
        shuffle_shards:       Shuffle shard order each epoch.
        shuffle_buffer:       Number of samples in the in-memory shuffle buffer.
        require_all_cameras:  If True, skip trajectories missing ext1/ext2/wrist.
        rank / world_size:    DDP sharding of shard files.
    """

    def __init__(
        self,
        embedding_dir: str,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 4096,
        require_all_cameras: bool = True,
        rank: int = 0,
        world_size: int = 1,
        shard_start: int = 0,
        shard_end: Optional[int] = None,
    ):
        super().__init__()
        self.embedding_dir       = embedding_dir
        self.shuffle_shards      = shuffle_shards
        self.shuffle_buffer      = shuffle_buffer
        self.require_all_cameras = require_all_cameras
        self.rank                = rank
        self.world_size          = world_size

        all_shards = sorted(glob.glob(os.path.join(embedding_dir, "*.tfrecord")))
        if not all_shards:
            raise FileNotFoundError(f"No .tfrecord files in {embedding_dir}")

        all_shards = all_shards[shard_start:shard_end]
        if not all_shards:
            raise ValueError(f"No shards in range [{shard_start}:{shard_end}] in {embedding_dir}")

        # DDP: each rank gets a disjoint subset of shards
        self.shards = [s for i, s in enumerate(all_shards) if i % world_size == rank]
        if not self.shards:
            raise ValueError(f"No shards for rank={rank}, world_size={world_size}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        shards = list(self.shards)
        if self.shuffle_shards:
            random.shuffle(shards)

        buffer: list = []

        for shard_path in shards:
            try:
                ds = tf.data.TFRecordDataset(shard_path)
                for raw in ds:
                    traj = _decode_traj(raw.numpy())
                    if traj is None:
                        continue

                    # Check camera availability
                    if self.require_all_cameras and traj["cameras_avail"] != 0b111:
                        continue

                    T = traj["T"]
                    for t in range(T):
                        e1 = traj["ext1"][t]   # [D] float16
                        e2 = traj["ext2"][t]   # [D] float16
                        ew = traj["wrist"][t]  # [D] float16
                        buffer.append((e1, e2, ew))

                        if len(buffer) >= self.shuffle_buffer:
                            idx = random.randrange(len(buffer))
                            item = buffer[idx]
                            buffer[idx] = buffer[-1]
                            buffer.pop()
                            yield (
                                _to_float32_tensor(item[0]),
                                _to_float32_tensor(item[1]),
                                _to_float32_tensor(item[2]),
                            )
            except tf.errors.DataLossError:
                print(f"[WARNING] Skipping corrupted shard (DataLossError): {shard_path}", flush=True)
                continue
            except Exception as exc:
                # Convert non-picklable TF exceptions so PyTorch workers can re-raise them
                raise RuntimeError(f"Error reading shard {shard_path}: {exc}") from None

        # Drain remaining buffer
        random.shuffle(buffer)
        for item in buffer:
            yield (
                _to_float32_tensor(item[0]),
                _to_float32_tensor(item[1]),
                _to_float32_tensor(item[2]),
            )


# ---------------------------------------------------------------------------
# Temporal dataset
# ---------------------------------------------------------------------------
class TemporalDataset(IterableDataset):
    """
    Iterates over precomputed embedding shards and yields
    (lang_emb, ext1_t, ext1_{t+k}) float32 tensors.

    For each trajectory, all valid (t, t+k) pairs are generated.

    Args:
        embedding_dir:   Directory containing .tfrecord files.
        k:               Temporal offset (default 8 steps).
        camera:          Which image embedding to use for st and st+k.
                         One of "ext1", "ext2", "wrist" (default "ext1").
        shuffle_shards:  Shuffle shard order each epoch.
        shuffle_buffer:  In-memory shuffle buffer size.
        rank / world_size: DDP sharding.
    """

    def __init__(
        self,
        embedding_dir: str,
        k: int = 8,
        camera: str = "ext1",
        shuffle_shards: bool = True,
        shuffle_buffer: int = 4096,
        rank: int = 0,
        world_size: int = 1,
        shard_start: int = 0,
        shard_end: Optional[int] = None,
    ):
        super().__init__()
        assert camera in ("ext1", "ext2", "wrist"), \
            f"camera must be ext1/ext2/wrist, got {camera!r}"

        self.embedding_dir = embedding_dir
        self.k             = k
        self.camera        = camera
        self.shuffle_shards  = shuffle_shards
        self.shuffle_buffer  = shuffle_buffer
        self.rank          = rank
        self.world_size    = world_size

        all_shards = sorted(glob.glob(os.path.join(embedding_dir, "*.tfrecord")))
        if not all_shards:
            raise FileNotFoundError(f"No .tfrecord files in {embedding_dir}")

        all_shards = all_shards[shard_start:shard_end]
        if not all_shards:
            raise ValueError(f"No shards in range [{shard_start}:{shard_end}] in {embedding_dir}")

        self.shards = [s for i, s in enumerate(all_shards) if i % world_size == rank]
        if not self.shards:
            raise ValueError(f"No shards for rank={rank}, world_size={world_size}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        shards = list(self.shards)
        if self.shuffle_shards:
            random.shuffle(shards)

        buffer: list = []

        for shard_path in shards:
            try:
                ds = tf.data.TFRecordDataset(shard_path)
                for raw in ds:
                    traj = _decode_traj(raw.numpy())
                    if traj is None:
                        continue

                    T    = traj["T"]
                    lang = traj["lang"]                    # [D] float16
                    imgs = traj[self.camera]               # [T, D] float16

                    # Only trajectories long enough for at least one (t, t+k) pair
                    if T <= self.k:
                        continue

                    for t in range(T - self.k):
                        st   = imgs[t]          # [D] float16
                        stk  = imgs[t + self.k] # [D] float16
                        buffer.append((lang, st, stk))

                        if len(buffer) >= self.shuffle_buffer:
                            idx = random.randrange(len(buffer))
                            item = buffer[idx]
                            buffer[idx] = buffer[-1]
                            buffer.pop()
                            yield (
                                _to_float32_tensor(item[0]),
                                _to_float32_tensor(item[1]),
                                _to_float32_tensor(item[2]),
                            )
            except tf.errors.DataLossError:
                print(f"[WARNING] Skipping corrupted shard (DataLossError): {shard_path}", flush=True)
                continue
            except Exception as exc:
                raise RuntimeError(f"Error reading shard {shard_path}: {exc}") from None

        # Drain remaining buffer
        random.shuffle(buffer)
        for item in buffer:
            yield (
                _to_float32_tensor(item[0]),
                _to_float32_tensor(item[1]),
                _to_float32_tensor(item[2]),
            )
