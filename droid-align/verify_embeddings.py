#!/usr/bin/env python3
"""
Verification script for precomputed SigLIP2 embedding TFRecords.

Checks:
  1. TFRecords are readable and parse correctly
  2. Embedding shapes match [T, D] and [D]
  3. Embeddings are float16 and L2-normalised (norm ≈ 1.0)
  4. Camera availability bitmasks are sensible
  5. Language embeddings are non-zero
  6. Temporal pair feasibility (enough steps for k=8 offset)

Usage:
  conda activate CLIP-DROID
  python verify_embeddings.py \\
      --embedding_dir /root/data/droid_embeddings \\
      --num_samples 100 \\
      --k 8
"""

import argparse
import os
import glob
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


# ---------------------------------------------------------------------------
# TFRecord parsing (must match precompute_embeddings.py schema)
# ---------------------------------------------------------------------------
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


def parse_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURE_SPEC)
    T = int(ex["num_steps"].numpy())
    D = int(ex["embed_dim"].numpy())

    def decode_matrix(raw_bytes, rows, cols):
        arr = np.frombuffer(raw_bytes.numpy(), dtype=np.float16)
        return arr.reshape(rows, cols)

    def decode_vector(raw_bytes, dim):
        arr = np.frombuffer(raw_bytes.numpy(), dtype=np.float16)
        return arr.reshape(dim)

    ext1  = decode_matrix(ex["ext1_emb_bytes"],  T, D)
    ext2  = decode_matrix(ex["ext2_emb_bytes"],  T, D)
    wrist = decode_matrix(ex["wrist_emb_bytes"], T, D)
    lang  = decode_vector(ex["lang_emb_bytes"],  D)

    return {
        "traj_id":       ex["traj_id"].numpy().decode("utf-8", errors="replace"),
        "num_steps":     T,
        "embed_dim":     D,
        "cameras_avail": int(ex["cameras_avail"].numpy()),
        "ext1":          ext1,
        "ext2":          ext2,
        "wrist":         wrist,
        "lang":          lang,
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------
def check_norms(arr: np.ndarray, name: str, tol: float = 0.05) -> bool:
    """Check that each row is approximately unit-normalised."""
    arr_f32 = arr.astype(np.float32)
    norms = np.linalg.norm(arr_f32, axis=-1)
    mean_norm = norms.mean()
    max_dev = np.abs(norms - 1.0).max()
    ok = max_dev < tol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: norm mean={mean_norm:.4f}, max_dev={max_dev:.4f}")
    return ok


def camera_bits_str(bits: int) -> str:
    parts = []
    if bits & 0b001: parts.append("ext1")
    if bits & 0b010: parts.append("ext2")
    if bits & 0b100: parts.append("wrist")
    return "+".join(parts) if parts else "NONE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Verify precomputed DROID embedding TFRecords")
    parser.add_argument("--embedding_dir", required=True, help="Directory containing .tfrecord shards")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of trajectories to inspect")
    parser.add_argument("--k", type=int, default=8,   help="Temporal offset k for pair feasibility check")
    args = parser.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.embedding_dir, "*.tfrecord")))
    if not shard_files:
        raise FileNotFoundError(f"No .tfrecord files found in {args.embedding_dir}")
    print(f"Found {len(shard_files)} TFRecord shard(s) in {args.embedding_dir}")

    # Aggregate statistics
    stats = {
        "total_trajs": 0,
        "total_steps": 0,
        "parse_errors": 0,
        "norm_failures": 0,
        "temporal_feasible": 0,
        "cameras_counter": {},
        "embed_dims": set(),
        "step_lengths": [],
    }

    ds = tf.data.TFRecordDataset(shard_files)
    sample_count = 0

    for raw in ds:
        if sample_count >= args.num_samples:
            break
        try:
            ex = parse_example(raw)
        except Exception as e:
            print(f"  [ERROR] Parse error on sample {sample_count}: {e}")
            stats["parse_errors"] += 1
            sample_count += 1
            continue

        T   = ex["num_steps"]
        D   = ex["embed_dim"]
        cam = ex["cameras_avail"]

        stats["total_trajs"] += 1
        stats["total_steps"] += T
        stats["embed_dims"].add(D)
        stats["step_lengths"].append(T)

        cam_str = camera_bits_str(cam)
        stats["cameras_counter"][cam_str] = stats["cameras_counter"].get(cam_str, 0) + 1

        print(f"\nTrajectory {sample_count}: id={ex['traj_id'][:60]}")
        print(f"  num_steps={T}, embed_dim={D}, cameras={cam_str} ({bin(cam)})")

        # Shape checks
        assert ex["ext1"].shape  == (T, D), f"ext1 shape {ex['ext1'].shape} != ({T},{D})"
        assert ex["ext2"].shape  == (T, D), f"ext2 shape {ex['ext2'].shape} != ({T},{D})"
        assert ex["wrist"].shape == (T, D), f"wrist shape {ex['wrist'].shape} != ({T},{D})"
        assert ex["lang"].shape  == (D,),   f"lang shape {ex['lang'].shape} != ({D},)"
        print(f"  [OK] All shapes correct")

        # Dtype checks
        for name, arr in [("ext1", ex["ext1"]), ("ext2", ex["ext2"]),
                          ("wrist", ex["wrist"]), ("lang", ex["lang"][np.newaxis])]:
            assert arr.dtype == np.float16, f"{name} dtype={arr.dtype}, expected float16"
        print(f"  [OK] All dtypes float16")

        # Norm checks
        ok1 = check_norms(ex["ext1"],        "ext1_embs")
        ok2 = check_norms(ex["ext2"],        "ext2_embs")
        ok3 = check_norms(ex["wrist"],       "wrist_embs")
        ok4 = check_norms(ex["lang"][np.newaxis], "lang_emb")
        if not all([ok1, ok2, ok3, ok4]):
            stats["norm_failures"] += 1

        # Temporal feasibility
        if T > args.k:
            stats["temporal_feasible"] += 1
            print(f"  [OK] Temporal pairs: {T - args.k} valid (t, t+{args.k}) pairs")
        else:
            print(f"  [WARN] Trajectory too short for k={args.k} (T={T})")

        # Sample cosine similarities (diagonal should be ~1 for normalised vectors)
        sample_idx = min(0, T - 1)
        sim_ext1_ext2 = float(np.dot(
            ex["ext1"][sample_idx].astype(np.float32),
            ex["ext2"][sample_idx].astype(np.float32),
        ))
        sim_ext1_wrist = float(np.dot(
            ex["ext1"][sample_idx].astype(np.float32),
            ex["wrist"][sample_idx].astype(np.float32),
        ))
        sim_ext1_lang = float(np.dot(
            ex["ext1"][sample_idx].astype(np.float32),
            ex["lang"].astype(np.float32),
        ))
        print(f"  Sample cosine sims at t=0: ext1⋅ext2={sim_ext1_ext2:.3f}, "
              f"ext1⋅wrist={sim_ext1_wrist:.3f}, ext1⋅lang={sim_ext1_lang:.3f}")

        sample_count += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Shards found:           {len(shard_files)}")
    print(f"  Trajectories inspected: {stats['total_trajs']}")
    print(f"  Parse errors:           {stats['parse_errors']}")
    print(f"  Norm failures:          {stats['norm_failures']}")
    print(f"  Total steps inspected:  {stats['total_steps']}")
    if stats["step_lengths"]:
        lens = np.array(stats["step_lengths"])
        print(f"  Step length stats:      min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}")
    print(f"  Embed dims seen:        {stats['embed_dims']}")
    print(f"  Temporal feasible (k={args.k}): {stats['temporal_feasible']}/{stats['total_trajs']}")
    print(f"  Camera availability breakdown:")
    for cam_str, count in sorted(stats["cameras_counter"].items()):
        print(f"    {cam_str:30s}: {count}")

    all_ok = stats["parse_errors"] == 0 and stats["norm_failures"] == 0
    print(f"\n{'[PASS] All checks passed!' if all_ok else '[WARN] Some checks failed – see above.'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
