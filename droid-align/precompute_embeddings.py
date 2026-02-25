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
from PIL import Image
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
EMBED_DIM = 1024          # ViT-L/16-SigLIP2 embedding dimension
IMG_SIZE  = 384           # SigLIP2-384 input resolution
BACKBONE  = "hf-hub:timm/ViT-L-16-SigLIP2-384"


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
# Image / text encoding helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_images_batched(
    model,
    preprocess,
    pil_images: list,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Encode a list of PIL Images using SigLIP2 image encoder.

    Returns float16 numpy array of shape [N, D] (L2-normalised).
    PIL preprocessing runs on CPU in parallel via a ThreadPoolExecutor so the
    GPU stays saturated.
    """
    from concurrent.futures import ThreadPoolExecutor

    N = len(pil_images)
    if N == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float16)

    # Preprocess all images in parallel on CPU threads.
    # 32 workers is the sweet spot on 192-core machines with OMP_NUM_THREADS=1.
    n_workers = min(32, N)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        tensors_list = list(ex.map(preprocess, pil_images))

    all_embs = []
    for start in range(0, N, batch_size):
        batch_t = torch.stack(tensors_list[start : start + batch_size])
        batch_t = batch_t.to(device, dtype=torch.float16)
        emb = model.encode_image(batch_t, normalize=True)   # [B, D]
        all_embs.append(emb.cpu().to(torch.float16).numpy())

    return np.concatenate(all_embs, axis=0)  # [N, D]


@torch.no_grad()
def encode_all_cameras_batched(
    model,
    preprocess,
    pil_ext1: list,
    pil_ext2: list,
    pil_wrist: list,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all three camera views in a single GPU forward pass.

    Stacks ext1 + ext2 + wrist into one list of 3*T images, encodes them
    together, then splits the result back.  This keeps the H200 saturated
    rather than doing three separate (T-sized) passes.

    Returns: (ext1_embs, ext2_embs, wrist_embs)  each float16 [T, D].
    """
    T = len(pil_ext1)
    combined = pil_ext1 + pil_ext2 + pil_wrist   # 3*T images
    all_embs = encode_images_batched(model, preprocess, combined, batch_size, device)
    return all_embs[:T], all_embs[T : 2 * T], all_embs[2 * T :]


@torch.no_grad()
def encode_text(model, tokenizer, texts: list, device: str, context_length: int = 64) -> np.ndarray:
    """
    Encode a list of text strings using SigLIP2 text encoder.

    Returns float16 numpy array of shape [N, D] (L2-normalised).
    """
    tokens = tokenizer(texts, context_length=context_length).to(device)
    emb = model.encode_text(tokens, normalize=True)  # [N, D]
    return emb.cpu().to(torch.float16).numpy()


# ---------------------------------------------------------------------------
# TFRecord helpers
# ---------------------------------------------------------------------------
def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_trajectory_example(
    traj_id: str,
    ext1_embs: np.ndarray,
    ext2_embs: np.ndarray,
    wrist_embs: np.ndarray,
    lang_emb: np.ndarray,
    cameras_avail: int,
) -> tf.train.Example:
    """
    Serialise one trajectory's embeddings into a tf.train.Example proto.

    ext1/ext2/wrist_embs: float16 [T, D]
    lang_emb:            float16 [D]
    cameras_avail:       bitmask (bit0=ext1, bit1=ext2, bit2=wrist)
    """
    T, D = ext1_embs.shape
    feature = {
        "traj_id":        _bytes_feature(traj_id.encode("utf-8")),
        "num_steps":      _int64_feature(T),
        "embed_dim":      _int64_feature(D),
        "cameras_avail":  _int64_feature(cameras_avail),
        "ext1_emb_bytes": _bytes_feature(ext1_embs.astype(np.float16).tobytes()),
        "ext2_emb_bytes": _bytes_feature(ext2_embs.astype(np.float16).tobytes()),
        "wrist_emb_bytes":_bytes_feature(wrist_embs.astype(np.float16).tobytes()),
        "lang_emb_bytes": _bytes_feature(lang_emb.astype(np.float16).tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------
def decode_image_to_pil(img_data) -> Image.Image | None:
    """
    Decode one image to a PIL Image.

    Handles two sources:
      - dlimp / raw RLDS: encoded JPEG/PNG bytes (bytes or 0-d numpy bytes)
      - TFDS as_dataset():  already decoded uint8 [H, W, C] numpy arrays

    Returns None if the image cannot be decoded.
    """
    if img_data is None:
        return None

    arr = np.asarray(img_data)

    # Already a decoded pixel array (H, W, C) or (H, W)
    if arr.ndim >= 2:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            return None
        return Image.fromarray(arr.astype(np.uint8))

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
        return Image.fromarray(decoded)
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
def iter_droid_trajectories(data_dir: str, split: str = "train"):
    """
    Yield numpy trajectory dicts one at a time using dlimp.

    dlimp keeps image fields as encoded bytes ([T] dtype=object), matching
    the raw RLDS format and consistent with droid_verifier_rlds_dataset.py.
    No filtering, no shuffling – we want every trajectory.
    """
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=False)
    yield from dataset.as_numpy_iterator()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_trajectory(traj, model, preprocess, tokenizer, img_batch_size, device, context_length=64):
    """
    Encode all images and language instruction for one dlimp trajectory dict.

    With dlimp, image fields are [T] numpy arrays of encoded bytes.
    Language fields are [T] or [K] arrays of bytes.

    Returns a result dict or None if the trajectory has no usable images.
    """
    obs = traj.get("observation", {})

    def get_img_bytes(key) -> np.ndarray:
        arr = obs.get(key, None)
        if arr is None:
            return np.array([], dtype=object)
        return np.asarray(arr)

    ext1_raw  = get_img_bytes("exterior_image_1_left")
    ext2_raw  = get_img_bytes("exterior_image_2_left")
    wrist_raw = get_img_bytes("wrist_image_left")

    T = max(len(ext1_raw), len(ext2_raw), len(wrist_raw))
    if T == 0:
        return None

    # Pad shorter arrays to length T with None
    def pad_to_T(arr):
        out = list(arr) if len(arr) > 0 else []
        while len(out) < T:
            out.append(None)
        return out

    ext1_raw  = pad_to_T(ext1_raw)
    ext2_raw  = pad_to_T(ext2_raw)
    wrist_raw = pad_to_T(wrist_raw)

    # ---- decode images ----
    pil_ext1  = [decode_image_to_pil(b) for b in ext1_raw]
    pil_ext2  = [decode_image_to_pil(b) for b in ext2_raw]
    pil_wrist = [decode_image_to_pil(b) for b in wrist_raw]

    cameras_avail = 0
    if any(img is not None for img in pil_ext1):  cameras_avail |= 0b001
    if any(img is not None for img in pil_ext2):  cameras_avail |= 0b010
    if any(img is not None for img in pil_wrist): cameras_avail |= 0b100

    if cameras_avail == 0:
        return None

    # ---- language instruction ----
    lang_text = None
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        arr = traj.get(key, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        for val in arr.flat:
            lang_text = _extract_str(val)
            if lang_text:
                break
        if lang_text:
            break
    if not lang_text:
        lang_text = "robot manipulation task"

    # ---- fill missing frames with blank images ----
    blank = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    pil_ext1  = [img if img is not None else blank for img in pil_ext1]
    pil_ext2  = [img if img is not None else blank for img in pil_ext2]
    pil_wrist = [img if img is not None else blank for img in pil_wrist]

    # ---- encode images (all 3 cameras in one GPU pass) ----
    ext1_embs, ext2_embs, wrist_embs = encode_all_cameras_batched(
        model, preprocess, pil_ext1, pil_ext2, pil_wrist, img_batch_size, device
    )

    # ---- encode language ----
    lang_emb = encode_text(model, tokenizer, [lang_text], device, context_length)[0]  # [D]

    # ---- traj_id from metadata ----
    try:
        meta    = traj.get("traj_metadata", {})
        ep_meta = meta.get("episode_metadata", {})
        raw_fp  = ep_meta.get("file_path", b"")
        traj_id = _extract_str(raw_fp) or f"traj_{hash(lang_text)}"
    except Exception:
        traj_id = "traj_unknown"

    return {
        "traj_id":       traj_id,
        "ext1_embs":     ext1_embs,   # [T, D] float16
        "ext2_embs":     ext2_embs,   # [T, D] float16
        "wrist_embs":    wrist_embs,  # [T, D] float16
        "lang_emb":      lang_emb,    # [D] float16
        "cameras_avail": cameras_avail,
        "num_steps":     T,
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute SigLIP2 embeddings for DROID")
    parser.add_argument("--rlds_data_dir",  required=True, help="Parent dir of droid/ TFDS folder")
    parser.add_argument("--output_dir",     required=True, help="Where to write embedding TFRecords")
    parser.add_argument("--num_shards",     type=int, default=256, help="Number of output shard files")
    parser.add_argument("--img_batch_size", type=int, default=128, help="Images per GPU batch")
    parser.add_argument("--device",         default="cuda", help="PyTorch device")
    parser.add_argument("--start_shard",    type=int, default=0,   help="Resume from this shard index")
    parser.add_argument("--split",          default="train",        help="TFDS split")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, preprocess, tokenizer, context_length = load_siglip2(device)

    # Open shard writers
    writers = []
    shard_paths = []
    for i in range(args.num_shards):
        p = os.path.join(args.output_dir, f"droid_embeddings_{i:05d}-of-{args.num_shards:05d}.tfrecord")
        shard_paths.append(p)

    # Skip shards that already exist (resume support)
    existing = {i for i, p in enumerate(shard_paths) if os.path.exists(p)}
    if existing:
        logger.info(f"Found {len(existing)} existing shards – will skip already-assigned trajectories")

    # Open all writers
    active_writers = []
    for i, p in enumerate(shard_paths):
        if i >= args.start_shard:
            active_writers.append(tf.io.TFRecordWriter(p))
        else:
            active_writers.append(None)

    logger.info(f"Writing to {args.output_dir} in {args.num_shards} shards")

    # Process trajectories
    traj_count = 0
    skip_count = 0
    t_start = time.time()

    ds_iter = iter_droid_trajectories(args.rlds_data_dir, split=args.split)

    for episode in tqdm(ds_iter, desc="Trajectories", unit="traj"):
        shard_idx = traj_count % args.num_shards

        # Skip if this shard is before start_shard
        if shard_idx < args.start_shard:
            traj_count += 1
            continue

        try:
            result = process_trajectory(episode, model, preprocess, tokenizer, args.img_batch_size, device, context_length)
        except Exception as e:
            logger.warning(f"Skipping trajectory {traj_count}: {e}")
            skip_count += 1
            traj_count += 1
            continue

        if result is None:
            skip_count += 1
            traj_count += 1
            continue

        proto = make_trajectory_example(
            traj_id=result["traj_id"],
            ext1_embs=result["ext1_embs"],
            ext2_embs=result["ext2_embs"],
            wrist_embs=result["wrist_embs"],
            lang_emb=result["lang_emb"],
            cameras_avail=result["cameras_avail"],
        )
        writer = active_writers[shard_idx]
        if writer is not None:
            writer.write(proto.SerializeToString())

        traj_count += 1

        if traj_count % 500 == 0:
            elapsed = time.time() - t_start
            rate = traj_count / max(elapsed, 1.0)
            logger.info(
                f"Processed {traj_count} trajectories ({skip_count} skipped) "
                f"at {rate:.1f} traj/s"
            )

    # Close all writers
    for w in active_writers:
        if w is not None:
            w.close()

    elapsed = time.time() - t_start
    logger.info(
        f"Done. Processed {traj_count} trajectories in {elapsed/60:.1f} min "
        f"({skip_count} skipped). Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
