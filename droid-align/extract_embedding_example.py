#!/usr/bin/env python3
"""
Extract one or more example trajectories from the first N embedding TFRecord shards.

Saves parsed data (traj_id, num_steps, embed_dim, cameras_avail, ext1/ext2/wrist/lang
embeddings as float16 arrays) to a .npz file for inspection or unit tests.

Usage:
  conda activate CLIP-DROID
  python extract_embedding_example.py \\
      --embedding_dir /root/data/droid_embeddings \\
      --num_shards 100 \\
      --num_examples 1 \\
      --output example_embedding_data.npz
"""

import argparse
import glob
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Must match precompute_embeddings.py schema
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


def parse_example(serialized: bytes) -> dict:
    ex = tf.io.parse_single_example(serialized, FEATURE_SPEC)
    T = int(ex["num_steps"].numpy())
    D = int(ex["embed_dim"].numpy())

    def decode_matrix(raw_bytes, rows, cols):
        arr = np.frombuffer(raw_bytes.numpy(), dtype=np.float16)
        return arr.reshape(rows, cols).copy()

    def decode_vector(raw_bytes, dim):
        arr = np.frombuffer(raw_bytes.numpy(), dtype=np.float16)
        return arr.reshape(dim).copy()

    return {
        "traj_id":       ex["traj_id"].numpy().decode("utf-8", errors="replace"),
        "num_steps":     T,
        "embed_dim":     D,
        "cameras_avail": int(ex["cameras_avail"].numpy()),
        "ext1":          decode_matrix(ex["ext1_emb_bytes"], T, D),
        "ext2":          decode_matrix(ex["ext2_emb_bytes"], T, D),
        "wrist":         decode_matrix(ex["wrist_emb_bytes"], T, D),
        "lang":          decode_vector(ex["lang_emb_bytes"], D),
    }


def main():
    p = argparse.ArgumentParser(description="Extract example data from embedding TFRecords")
    p.add_argument("--embedding_dir", default="/root/data/droid_embeddings", help="Directory of .tfrecord shards")
    p.add_argument("--num_shards", type=int, default=100, help="Use first N shard files")
    p.add_argument("--num_examples", type=int, default=1, help="Number of trajectories to extract")
    p.add_argument("--output", default="example_embedding_data.npz", help="Output .npz path")
    args = p.parse_args()

    pattern = os.path.join(args.embedding_dir, "*.tfrecord")
    shards = sorted(glob.glob(pattern))[: args.num_shards]
    if not shards:
        raise SystemExit(f"No TFRecord files in {args.embedding_dir}")

    print(f"Reading from first {len(shards)} shards in {args.embedding_dir}")
    examples = []

    for path in shards:
        for raw in tf.data.TFRecordDataset([path]):
            ex = parse_example(raw.numpy())
            examples.append(ex)
            if len(examples) >= args.num_examples:
                break
        if len(examples) >= args.num_examples:
            break

    if not examples:
        raise SystemExit("No examples found in the given shards")

    # Save first example as .npz (load with np.load(path, allow_pickle=True) for traj_id)
    out = examples[0]
    np.savez(
        args.output,
        traj_id=np.array([out["traj_id"]], dtype=object),
        num_steps=np.int64(out["num_steps"]),
        embed_dim=np.int64(out["embed_dim"]),
        cameras_avail=np.int64(out["cameras_avail"]),
        ext1=out["ext1"],
        ext2=out["ext2"],
        wrist=out["wrist"],
        lang=out["lang"],
    )
    print(f"Saved 1 example to {args.output}")
    tid = out["traj_id"]
    print(f"  traj_id={tid[:80]}..." if len(tid) > 80 else f"  traj_id={tid}")
    print(f"  num_steps={out['num_steps']}, embed_dim={out['embed_dim']}, cameras_avail={out['cameras_avail']}")
    print(f"  ext1 shape {out['ext1'].shape}, lang shape {out['lang'].shape}")

    if len(examples) > 1:
        extra = os.path.splitext(args.output)[0] + "_extra.npz"
        np.savez(
            extra,
            traj_ids=np.array([e["traj_id"] for e in examples], dtype=object),
            num_steps=np.array([e["num_steps"] for e in examples]),
            embed_dims=np.array([e["embed_dim"] for e in examples]),
        )
        print(f"Saved {len(examples)} traj_ids/num_steps/embed_dims to {extra}")


if __name__ == "__main__":
    main()
