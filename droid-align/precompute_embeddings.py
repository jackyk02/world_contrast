#!/usr/bin/env python3
"""
Precompute ViT-L/16-SigLIP2-384 embeddings for all DROID trajectories.

For each trajectory the script encodes:
  - exterior_image_1_left   (primary exterior camera)
  - exterior_image_2_left   (secondary exterior camera)
  - wrist_image_left        (wrist camera)
  - language_instruction    (text encoder, one per trajectory)

All embeddings are stored as float16.  Raw images and text strings are
discarded from the output.

Output TFRecord schema (one proto = one trajectory):
  traj_id           bytes   unique episode identifier
  num_steps         int64   number of steps T
  embed_dim         int64   embedding dimension D (1024 for ViT-L)
  cameras_avail     int64   bitmask: bit0=ext1, bit1=ext2, bit2=wrist
  ext1_emb_bytes    bytes   float16 raw bytes, shape [T, D]
  ext2_emb_bytes    bytes   float16 raw bytes, shape [T, D]
  wrist_emb_bytes   bytes   float16 raw bytes, shape [T, D]
  lang_emb_bytes    bytes   float16 raw bytes, shape [D]

Usage:
  conda activate CLIP-DROID
  python precompute_embeddings.py \\
      --rlds_data_dir /root/data \\
      --output_dir /root/data/droid_embeddings \\
      --num_shards 256 \\
      --img_batch_size 128 \\
      --device cuda
"""

import argparse
import os
# Must be set before importing numpy/PIL/torch to prevent OpenMP thread explosion
# when combined with ThreadPoolExecutor (192 cores × many libs = segfault).
os.environ.setdefault("OMP_NUM_THREADS",    "1")
os.environ.setdefault("MKL_NUM_THREADS",    "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")

import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Suppress TF logging before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# NOTE: do NOT set CUDA_VISIBLE_DEVICES="" here — that would hide the GPU from PyTorch too.
# TF is kept off the GPU solely via tf.config.set_visible_devices below.

import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl

tf.config.set_visible_devices([], "GPU")  # TF uses CPU; PyTorch keeps the GPU

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM      = 1024   # ViT-L/16-SigLIP2 embedding dimension
STAGE_A_HEIGHT = 224    # Stage A intermediate resolution (matches training pipeline)
STAGE_A_WIDTH  = 224
BACKBONE       = "hf-hub:timm/ViT-L-16-SigLIP2-384"


# ---------------------------------------------------------------------------
# Core GPU engine: Stage A + Stage B + encode for an arbitrary frame batch
# ---------------------------------------------------------------------------
import torch.nn.functional as _F
from collections import defaultdict


@torch.no_grad()
def encode_batch_gpu(
    model,
    frames: list,   # N × [H, W, 3] uint8 numpy arrays (mixed sizes allowed)
    device: str,
) -> torch.Tensor:
    """
    Encode any list of raw uint8 HWC frames in one GPU pass.

    Stage A – group by (H, W), bilinear resize+pad → [N, 3, 224, 224] float16
    Stage B – bicubic squash → [N, 3, 384, 384]  + SigLIP2 normalise → [-1, 1]
    Encode  – model.encode_image → [N, 1024] float16, L2-normalised

    All ops in float16.  Handles mixed camera resolutions via shape-grouping.

    Returns: float16 [N, 1024] GPU tensor.
    """
    N = len(frames)
    if N == 0:
        return torch.zeros(0, EMBED_DIM, dtype=torch.float16, device=device)

    # Group by (H, W) for a single batched Stage-A kernel per resolution
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, arr in enumerate(frames):
        groups[(arr.shape[0], arr.shape[1])].append(i)

    stage_a_out = [None] * N   # will hold [3, 224, 224] float16 tensors
    for (H, W), idxs in groups.items():
        batch = np.stack([frames[i] for i in idxs])         # [M, H, W, 3] uint8
        t = (torch.from_numpy(batch)
             .to(device).permute(0, 3, 1, 2)                # [M, 3, H, W]
             .to(torch.float16))                             # float16, [0-255]
        ratio = max(W / STAGE_A_WIDTH, H / STAGE_A_HEIGHT)
        nh, nw = int(H / ratio), int(W / ratio)
        t = _F.interpolate(t, (nh, nw), mode="bilinear", align_corners=False)
        ph0, ph_r = divmod(STAGE_A_HEIGHT - nh, 2)
        pw0, pw_r = divmod(STAGE_A_WIDTH  - nw, 2)
        t = _F.pad(t, (pw0, pw0 + pw_r, ph0, ph0 + ph_r), value=0.0)
        # t: [M, 3, 224, 224] float16
        for j, idx in enumerate(idxs):
            stage_a_out[idx] = t[j]

    # Single Stage B + encode pass for the whole buffer
    combined = torch.stack(stage_a_out)   # [N, 3, 224, 224] float16
    del stage_a_out
    combined = _F.interpolate(combined / 255.0, (384, 384),
                               mode="bicubic", align_corners=False)
    combined = (combined - 0.5) / 0.5    # [-1, 1] float16
    embs = model.encode_image(combined, normalize=True)   # [N, 1024] float16
    del combined
    return embs.to(torch.float16)         # [N, 1024] float16, still on GPU


# ---------------------------------------------------------------------------
# Cross-trajectory frame buffer
# ---------------------------------------------------------------------------
class CrossTrajectoryBuffer:
    """
    Accumulates raw uint8 frames from multiple trajectories and encodes them
    in large batches to maximise H200 utilisation.

    Workflow:
      1. add_trajectory()  — queues all 3×T frames + metadata; no GPU work yet
      2. is_full()         — True when buffered frame count ≥ max_batch_size
      3. flush()           — calls encode_batch_gpu on the entire buffer,
                             assigns embeddings back, returns completed results
      4. flush_all()       — drain remaining frames after the dataset is exhausted

    A trajectory result is returned only when ALL 3×T of its frames have been
    encoded (which may span multiple flush() calls for very long trajectories).
    """

    def __init__(self, max_batch_size: int, device: str):
        self.max_batch_size = max_batch_size
        self.device = device
        # Flat frame queue: (frame_np, traj_key, cam_id, step_idx)
        self._queue: list[tuple] = []
        # Per-trajectory accumulators keyed by a unique traj_key (int)
        self._partials: dict[int, dict] = {}
        self._next_key = 0

    def add_trajectory(
        self,
        traj_id: str,
        T: int,
        cameras_avail: int,
        lang_emb: torch.Tensor,       # float16 CPU [D]
        np_ext1: list,
        np_ext2: list,
        np_wrist: list,
    ) -> int:
        """Queue all 3×T frames. Returns the traj_key for tracking."""
        key = self._next_key
        self._next_key += 1
        self._partials[key] = {
            "traj_id":      traj_id,
            "T":            T,
            "cameras_avail": cameras_avail,
            "lang_emb":     lang_emb,
            "ext1_embs":    torch.zeros(T, EMBED_DIM, dtype=torch.float16),
            "ext2_embs":    torch.zeros(T, EMBED_DIM, dtype=torch.float16),
            "wrist_embs":   torch.zeros(T, EMBED_DIM, dtype=torch.float16),
            "remaining":    3 * T,
        }
        cam_arrays = [np_ext1, np_ext2, np_wrist]
        for cam_id, cam_list in enumerate(cam_arrays):
            for step_i, frame in enumerate(cam_list):
                self._queue.append((frame, key, cam_id, step_i))
        return key

    def __len__(self):
        return len(self._queue)

    def is_full(self) -> bool:
        return len(self._queue) >= self.max_batch_size

    def flush(self, model, n: int | None = None) -> list[dict]:
        """
        Encode up to n frames (default: max_batch_size).
        Returns a list of complete trajectory result dicts.
        """
        if not self._queue:
            return []
        n = n or self.max_batch_size
        batch = self._queue[:n]
        self._queue = self._queue[n:]

        frames = [item[0] for item in batch]
        metas  = [(item[1], item[2], item[3]) for item in batch]

        embs = encode_batch_gpu(model, frames, self.device)  # [B, D] float16 GPU
        embs_cpu = embs.cpu()                                 # move once, stay float16
        del embs

        cam_keys = ("ext1_embs", "ext2_embs", "wrist_embs")
        completed: list[dict] = []
        for i, (key, cam_id, step_i) in enumerate(metas):
            p = self._partials[key]
            p[cam_keys[cam_id]][step_i] = embs_cpu[i]
            p["remaining"] -= 1
            if p["remaining"] == 0:
                result = self._partials.pop(key)
                result["_key"] = key   # carry the key so main can look up shard_idx
                completed.append(result)

        return completed

    def flush_all(self, model) -> list[dict]:
        """Drain all remaining queued frames."""
        completed: list[dict] = []
        while self._queue:
            completed.extend(self.flush(model, n=self.max_batch_size))
        return completed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _patch_tokenizer(tokenizer):
    """
    Newer transformers versions removed `batch_encode_plus` from some tokenizers
    (e.g. GemmaTokenizer) even though `open_clip`'s HFTokenizer still calls it.
    Add it back as an alias for `__call__` if missing.
    """
    hf_tok = getattr(tokenizer, "tokenizer", None)
    if hf_tok is None:
        return
    if not hasattr(hf_tok, "batch_encode_plus"):
        hf_tok.batch_encode_plus = hf_tok.__call__
        logger.info("Patched tokenizer: added batch_encode_plus alias for __call__")


def load_siglip2(device: str):
    """Load ViT-L/16-SigLIP2-384 via open_clip and return (model, preprocess, tokenizer, context_length)."""
    from open_clip import create_model_from_pretrained, get_tokenizer

    logger.info(f"Loading SigLIP2 backbone: {BACKBONE}")
    model, preprocess = create_model_from_pretrained(BACKBONE)
    tokenizer = get_tokenizer(BACKBONE)

    # Fix: GemmaTokenizer in newer transformers may not have batch_encode_plus
    _patch_tokenizer(tokenizer)

    context_length = int(getattr(model, "context_length", 64))
    logger.info(f"context_length={context_length}")

    model = model.eval().to(device)
    model = model.to(torch.float16)
    logger.info(f"Model loaded on {device} (float16 inference)")
    return model, preprocess, tokenizer, context_length


# ---------------------------------------------------------------------------
# Text encoding helper  (image encoding is handled by fused_encode_all_cameras)
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_text(model, tokenizer, texts: list, device: str, context_length: int = 64) -> torch.Tensor:
    """
    Encode a list of text strings using SigLIP2 text encoder.

    Returns float16 CPU tensor of shape [N, D] (L2-normalised).
    Stays as a torch.Tensor — numpy conversion only at TFRecord serialisation.
    """
    tokens = tokenizer(texts, context_length=context_length).to(device)
    emb = model.encode_text(tokens, normalize=True)  # [N, D]
    return emb.cpu().to(torch.float16)


# ---------------------------------------------------------------------------
# TFRecord helpers
# ---------------------------------------------------------------------------
def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _to_fp16_bytes(t) -> bytes:
    """Convert a torch.float16 CPU tensor (or numpy array) to raw bytes."""
    if isinstance(t, torch.Tensor):
        return t.to(torch.float16).numpy().tobytes()
    return np.asarray(t, dtype=np.float16).tobytes()


def make_trajectory_example(
    traj_id: str,
    ext1_embs,          # torch.float16 [T, D]  or np.float16 [T, D]
    ext2_embs,
    wrist_embs,
    lang_emb,           # torch.float16 [D]     or np.float16 [D]
    cameras_avail: int,
) -> tf.train.Example:
    """
    Serialise one trajectory's embeddings into a tf.train.Example proto.

    Accepts either torch.float16 CPU tensors or numpy float16 arrays.
    The single numpy() + tobytes() call per field happens here — the only
    point where we leave the torch.float16 world.
    """
    if isinstance(ext1_embs, torch.Tensor):
        T, D = ext1_embs.shape
    else:
        T, D = np.asarray(ext1_embs).shape

    feature = {
        "traj_id":        _bytes_feature(traj_id.encode("utf-8")),
        "num_steps":      _int64_feature(T),
        "embed_dim":      _int64_feature(D),
        "cameras_avail":  _int64_feature(cameras_avail),
        "ext1_emb_bytes": _bytes_feature(_to_fp16_bytes(ext1_embs)),
        "ext2_emb_bytes": _bytes_feature(_to_fp16_bytes(ext2_embs)),
        "wrist_emb_bytes":_bytes_feature(_to_fp16_bytes(wrist_embs)),
        "lang_emb_bytes": _bytes_feature(_to_fp16_bytes(lang_emb)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def decode_image_to_numpy(img_data) -> np.ndarray | None:
    """
    Decode one image field to a uint8 [H, W, 3] numpy array.

    Handles two sources:
      - dlimp / raw RLDS:  encoded JPEG/PNG bytes (bytes or 0-d numpy scalar)
      - TFDS as_dataset(): already decoded uint8 [H, W, C] numpy arrays

    Returns None if decoding fails or the image has zero size.
    """
    if img_data is None:
        return None

    arr = np.asarray(img_data)

    # Already a decoded pixel array (H, W, C)
    if arr.ndim >= 2:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            return None
        return arr.astype(np.uint8)

    # Scalar containing encoded bytes
    raw: bytes
    if arr.ndim == 0:
        raw = arr.item()
    elif isinstance(img_data, bytes):
        raw = img_data
    else:
        return None

    if not isinstance(raw, bytes) or len(raw) == 0:
        return None

    try:
        decoded = tf.io.decode_image(raw, expand_animations=False, dtype=tf.uint8).numpy()
        return decoded.astype(np.uint8)
    except Exception:
        return None


def _extract_str(val) -> str | None:
    """Robustly extract a Python str from bytes/ndarray/str."""
    if isinstance(val, bytes):
        s = val.decode("utf-8", errors="replace").strip()
        return s if s else None
    if isinstance(val, str):
        return val.strip() or None
    arr = np.asarray(val)
    if arr.size == 0:
        return None
    item = arr.flat[0]
    return _extract_str(item)


# ---------------------------------------------------------------------------
# Trajectory iterator  (uses dlimp – same as the rest of the codebase)
# ---------------------------------------------------------------------------
def iter_droid_trajectories(
    data_dir: str,
    split: str = "train",
    num_parallel_reads: int = 16,
    prefetch: int = 32,
):
    """
    Yield numpy trajectory dicts using dlimp with parallel disk I/O.

    num_parallel_reads: concurrent TFRecord shard readers.
      DROID has 2048 shards; 16 readers is a good balance between I/O
      parallelism and memory pressure.  Increase if disk bandwidth allows.
    prefetch: number of trajectories to buffer ahead in the tf.data pipeline.
    """
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=False,
        num_parallel_reads=num_parallel_reads,
    )
    dataset = dataset.prefetch(prefetch)
    yield from dataset.as_numpy_iterator()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def prepare_trajectory(traj, model, tokenizer, device, context_length=64):
    """
    CPU-side trajectory prep: decode frames in parallel + encode language.

    GPU image encoding is intentionally deferred — the caller queues the
    returned numpy frames into a CrossTrajectoryBuffer so many trajectories
    can be batched together for a single large encode_batch_gpu call.

    Returns a dict with keys:
        traj_id, T, cameras_avail, lang_emb (float16 CPU tensor),
        np_ext1, np_ext2, np_wrist  (lists of [H,W,3] uint8 numpy arrays)
    or None if the trajectory has no usable images.
    """
    from concurrent.futures import ThreadPoolExecutor

    obs = traj.get("observation", {})

    def get_raw(key):
        arr = obs.get(key, None)
        return list(np.asarray(arr)) if arr is not None and len(arr) > 0 else []

    ext1_raw  = get_raw("exterior_image_1_left")
    ext2_raw  = get_raw("exterior_image_2_left")
    wrist_raw = get_raw("wrist_image_left")

    T = max(len(ext1_raw), len(ext2_raw), len(wrist_raw))
    if T == 0:
        return None

    ext1_raw  += [None] * (T - len(ext1_raw))
    ext2_raw  += [None] * (T - len(ext2_raw))
    wrist_raw += [None] * (T - len(wrist_raw))

    # Parallel JPEG decode for all 3T frames at once
    all_raw = ext1_raw + ext2_raw + wrist_raw
    with ThreadPoolExecutor(max_workers=min(32, len(all_raw))) as ex:
        all_np = list(ex.map(decode_image_to_numpy, all_raw))

    np_ext1  = all_np[:T]
    np_ext2  = all_np[T : 2 * T]
    np_wrist = all_np[2 * T :]

    cameras_avail = 0
    if any(a is not None for a in np_ext1):  cameras_avail |= 0b001
    if any(a is not None for a in np_ext2):  cameras_avail |= 0b010
    if any(a is not None for a in np_wrist): cameras_avail |= 0b100
    if cameras_avail == 0:
        return None

    # Language
    lang_text = None
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        arr = traj.get(key, None)
        if arr is None:
            continue
        for val in np.asarray(arr).flat:
            lang_text = _extract_str(val)
            if lang_text:
                break
        if lang_text:
            break
    if not lang_text:
        lang_text = "robot manipulation task"

    # Fill missing frames with zeros (shape from first valid frame)
    def fill_blanks(np_list):
        ref = next((a for a in np_list if a is not None), None)
        blank = np.zeros(ref.shape if ref is not None else (480, 640, 3), dtype=np.uint8)
        return [a if a is not None else blank for a in np_list]

    np_ext1  = fill_blanks(np_ext1)
    np_ext2  = fill_blanks(np_ext2)
    np_wrist = fill_blanks(np_wrist)

    # Language encoding (cheap GPU op, done per-trajectory)
    lang_emb = encode_text(model, tokenizer, [lang_text], device, context_length)[0]

    # traj_id
    try:
        meta    = traj.get("traj_metadata", {})
        ep_meta = meta.get("episode_metadata", {})
        traj_id = _extract_str(ep_meta.get("file_path", b"")) or f"traj_{hash(lang_text)}"
    except Exception:
        traj_id = "traj_unknown"

    return {
        "traj_id":      traj_id,
        "T":            T,
        "cameras_avail": cameras_avail,
        "lang_emb":     lang_emb,     # float16 CPU tensor [D]
        "np_ext1":      np_ext1,      # T × [H,W,3] uint8
        "np_ext2":      np_ext2,
        "np_wrist":     np_wrist,
    }


def write_result(result: dict, active_writers: list, shard_idx: int):
    """Serialise one completed trajectory result to its TFRecord shard."""
    writer = active_writers[shard_idx]
    if writer is None:
        return
    proto = make_trajectory_example(
        traj_id=result["traj_id"],
        ext1_embs=result["ext1_embs"],
        ext2_embs=result["ext2_embs"],
        wrist_embs=result["wrist_embs"],
        lang_emb=result["lang_emb"],
        cameras_avail=result["cameras_avail"],
    )
    writer.write(proto.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description="Precompute SigLIP2 embeddings for DROID")
    parser.add_argument("--rlds_data_dir",  required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--num_shards",     type=int, default=256)
    parser.add_argument("--batch_size",          type=int, default=4096,
                        help="Cross-trajectory GPU batch size (images). "
                             "H200: 4096 imgs ≈ 81 GB VRAM.")
    parser.add_argument("--device",              default="cuda")
    parser.add_argument("--start_shard",         type=int, default=0)
    parser.add_argument("--split",               default="train")
    parser.add_argument("--num_parallel_reads",  type=int, default=16,
                        help="Concurrent RLDS TFRecord shard readers (disk I/O parallelism).")
    parser.add_argument("--prefetch",            type=int, default=32,
                        help="tf.data prefetch buffer size (trajectories).")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    model, _, tokenizer, context_length = load_siglip2(device)

    # Open shard writers
    shard_paths = [
        os.path.join(args.output_dir,
                     f"droid_embeddings_{i:05d}-of-{args.num_shards:05d}.tfrecord")
        for i in range(args.num_shards)
    ]
    active_writers = [
        tf.io.TFRecordWriter(p) if i >= args.start_shard else None
        for i, p in enumerate(shard_paths)
    ]

    logger.info(
        f"Writing to {args.output_dir} | {args.num_shards} shards | "
        f"cross-traj batch_size={args.batch_size}"
    )

    # Buffer that accumulates frames across trajectories
    buf = CrossTrajectoryBuffer(max_batch_size=args.batch_size, device=device)
    # Maps traj_key → shard_idx so we know where to write after encoding
    key_to_shard: dict[int, int] = {}

    traj_count = 0
    skip_count = 0
    write_count = 0
    t_start = time.time()

    def _write_completed(completed: list[dict]):
        nonlocal write_count
        for res in completed:
            shard_idx = key_to_shard.pop(res.get("_key", -1), None)
            if shard_idx is not None:
                write_result(res, active_writers, shard_idx)
                write_count += 1

    ds_iter = iter_droid_trajectories(
        args.rlds_data_dir,
        split=args.split,
        num_parallel_reads=args.num_parallel_reads,
        prefetch=args.prefetch,
    )

    for episode in tqdm(ds_iter, desc="Trajectories", unit="traj"):
        shard_idx = traj_count % args.num_shards

        if shard_idx < args.start_shard:
            traj_count += 1
            continue

        try:
            prep = prepare_trajectory(episode, model, tokenizer, device, context_length)
        except Exception as e:
            logger.warning(f"Skipping trajectory {traj_count}: {e}")
            skip_count += 1
            traj_count += 1
            continue

        if prep is None:
            skip_count += 1
            traj_count += 1
            continue

        key = buf.add_trajectory(
            traj_id      = prep["traj_id"],
            T            = prep["T"],
            cameras_avail= prep["cameras_avail"],
            lang_emb     = prep["lang_emb"],
            np_ext1      = prep["np_ext1"],
            np_ext2      = prep["np_ext2"],
            np_wrist     = prep["np_wrist"],
        )
        key_to_shard[key] = shard_idx

        # Flush whenever the buffer is full
        while buf.is_full():
            _write_completed(buf.flush(model))

        traj_count += 1

        if traj_count % 200 == 0:
            elapsed = time.time() - t_start
            mem_gb = torch.cuda.memory_reserved() / 1e9 if device == "cuda" else 0
            logger.info(
                f"step={traj_count}  queued={len(buf)}  written={write_count}"
                f"  skipped={skip_count}  GPU_mem={mem_gb:.1f}GB"
                f"  {traj_count / max(elapsed,1):.1f} traj/s"
            )

    # Final drain — flush all remaining buffered frames
    logger.info("Draining buffer...")
    completed = buf.flush_all(model)
    _write_completed(completed)

    for w in active_writers:
        if w is not None:
            w.close()

    elapsed = time.time() - t_start
    logger.info(
        f"Done. {traj_count} trajectories in {elapsed/60:.1f} min "
        f"({skip_count} skipped, {write_count} written). "
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
