#!/usr/bin/env python3
"""
Append per-step delta actions to existing droid_embeddings TFRecords.

For each trajectory already stored in the embedding shards:
  1. Retrieve DROID 1.0.1 actions via RLDS (action_dict/joint_position +
     action_dict/gripper_position → 8-D, matching DroidRldsDataset).
  2. Retrieve per-step state (observation/joint_position + observation/gripper_position).
  3. Compute per-step delta actions:  delta[t] = action[t] - state[t]  for
     joint dims (0-6); gripper dim (7) is left as-is.  This matches the
     _compute_action_deltas() function in openpi_rlds_two_view.py.
  4. Apply quantile normalisation (q01/q99 → [-1,1] for joints; binarise
     gripper at 0.5), matching _normalize_actions_quantile() in the same file.
  5. Also store normalised *absolute* actions (no delta) for use-cases where
     use_delta_actions=False.

New fields added per trajectory:
  action_dim           int64   8
  actions_bytes        bytes   float32 raw bytes, shape [T, 8]
                               – normalised absolute actions
  delta_actions_bytes  bytes   float32 raw bytes, shape [T, 8]
                               – normalised delta actions (action[t] - state[t])
  is_successful        int64   1 if episode_metadata/file_path contains "success",
                               else 0  (mirrors the filter in droid_verifier_rlds_dataset.py)
  is_demonstration     int64   1 if the last step has is_terminal=True, else 0
                               (mirrors the --demonstrations_only filter in
                               precompute_embeddings.py; ~76k of ~95.6k train episodes)

Existing fields (embeddings, traj_id, etc.) are preserved unchanged.

If --output_dir is omitted the shards are rewritten in-place (via atomic
rename of a temp file in the same directory so the originals are safe until
the write completes).

Usage:
  conda activate CLIP-DROID
  python append_delta_actions.py \\
      --embeddings_dir /root/data/droid_embeddings \\
      --rlds_data_dir  /root/data \\
      [--output_dir    /root/data/droid_embeddings_with_actions]
"""

import argparse
import glob
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Must precede any numpy/torch import to prevent thread over-subscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl

tf.config.set_visible_devices([], "GPU")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action normalisation constants (copied from openpi_rlds_two_view.py)
# ---------------------------------------------------------------------------
DROID_ACTION_Q01 = np.array(
    [-0.45159999, -0.79799998, -0.4384, -0.90880001,
     -0.634,      -0.63279998, -0.75160003, 0.0],
    dtype=np.float32,
)
DROID_ACTION_Q99 = np.array(
    [0.43880001, 0.76440001, 0.44319999, 0.78600001,
     0.63800001, 0.65679997, 0.72439998, 0.9982],
    dtype=np.float32,
)

DELTA_MASK = np.array([True, True, True, True, True, True, True, False], dtype=bool)


def _normalize_actions_quantile(
    actions: np.ndarray,
    q01: np.ndarray = DROID_ACTION_Q01,
    q99: np.ndarray = DROID_ACTION_Q99,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """
    Quantile-normalise DROID 8-D actions, matching openpi_rlds_two_view.py.
    Dims 0-6 (joints): mapped to [-1, 1] via q01/q99.
    Dim 7  (gripper):  binarised to 0 or 1.
    """
    result = np.array(actions, copy=True, dtype=np.float32)
    joint_dims = min(7, actions.shape[-1])
    result[..., :joint_dims] = (
        (actions[..., :joint_dims] - q01[:joint_dims])
        / (q99[:joint_dims] - q01[:joint_dims] + 1e-6)
        * 2.0
        - 1.0
    )
    if actions.shape[-1] > 7:
        result[..., 7] = (actions[..., 7] >= gripper_threshold).astype(np.float32)
    return result


def _compute_per_step_delta(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """
    Compute per-step delta actions: delta[t] = action[t] - state[t] for
    joint dims (DELTA_MASK=True); gripper (dim 7) is unchanged.

    actions : float32 [T, 8]
    state   : float32 [T, 8]
    returns : float32 [T, 8]

    This is the element-wise version of _compute_action_deltas() in
    openpi_rlds_two_view.py (which broadcasts a single centre state across
    a whole action window; here each timestep has its own reference state).
    """
    delta = np.array(actions, copy=True, dtype=np.float32)
    delta[:, DELTA_MASK] -= state[:, DELTA_MASK]
    return delta


# ---------------------------------------------------------------------------
# TFRecord helpers
# ---------------------------------------------------------------------------

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _to_fp32_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes()


def augment_proto(
    raw_bytes: bytes,
    actions_norm: np.ndarray,        # float32 [T, 8]  normalised absolute
    delta_actions_norm: np.ndarray,  # float32 [T, 8]  normalised delta
    is_successful: int,              # 1 if file_path contains "success"
    is_demonstration: int,           # 1 if last step is_terminal=True
) -> bytes:
    """
    Parse an existing tf.train.Example proto, add action + success fields,
    serialise.  Existing fields are preserved exactly.
    """
    example = tf.train.Example()
    example.ParseFromString(raw_bytes)
    f = example.features.feature
    f["action_dim"].CopyFrom(_int64_feature(actions_norm.shape[-1]))
    f["actions_bytes"].CopyFrom(_bytes_feature(_to_fp32_bytes(actions_norm)))
    f["delta_actions_bytes"].CopyFrom(_bytes_feature(_to_fp32_bytes(delta_actions_norm)))
    f["is_successful"].CopyFrom(_int64_feature(is_successful))
    f["is_demonstration"].CopyFrom(_int64_feature(is_demonstration))
    return example.SerializeToString()


# ---------------------------------------------------------------------------
# TFRecord reading
# ---------------------------------------------------------------------------

def read_shard(shard_path: str) -> list[tuple[str, bytes]]:
    """Return [(traj_id, raw_proto_bytes), ...] for every record in the shard."""
    records = []
    for raw in tf.data.TFRecordDataset([shard_path]):
        rb = raw.numpy()
        ex = tf.train.Example()
        ex.ParseFromString(rb)
        traj_id_bytes = ex.features.feature["traj_id"].bytes_list.value[0]
        traj_id = traj_id_bytes.decode("utf-8", errors="replace")
        records.append((traj_id, rb))
    return records


# ---------------------------------------------------------------------------
# DROID 1.0.1 iterator (identical to precompute_embeddings.py)
# ---------------------------------------------------------------------------

def iter_droid_trajectories(
    data_dir: str,
    split: str = "train",
    num_parallel_reads: int = 16,
    prefetch: int = 32,
):
    builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=False,
        num_parallel_reads=num_parallel_reads,
    )
    dataset = dataset.prefetch(prefetch)
    yield from dataset.as_numpy_iterator()


def _extract_str(val) -> str | None:
    if isinstance(val, bytes):
        s = val.decode("utf-8", errors="replace").strip()
        return s if s else None
    if isinstance(val, str):
        return val.strip() or None
    arr = np.asarray(val)
    if arr.size == 0:
        return None
    return _extract_str(arr.flat[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append per-step delta actions to droid_embeddings TFRecords"
    )
    parser.add_argument(
        "--embeddings_dir", required=True,
        help="Directory containing droid_embeddings_*.tfrecord shards",
    )
    parser.add_argument(
        "--rlds_data_dir", required=True,
        help="RLDS data root (parent of droid/1.0.1/), e.g. /root/data",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory for augmented shards. "
             "If omitted, shards are rewritten in-place (atomic rename).",
    )
    parser.add_argument(
        "--split", default="train",
        help="TFDS split to stream (default: train)",
    )
    parser.add_argument(
        "--num_parallel_reads", type=int, default=16,
        help="Concurrent RLDS TFRecord shard readers",
    )
    parser.add_argument(
        "--prefetch", type=int, default=32,
        help="tf.data prefetch buffer size (trajectories)",
    )
    parser.add_argument(
        "--demonstrations_only", action="store_true", default=True,
        help="Skip non-demonstration episodes (is_terminal False on last step)",
    )
    parser.add_argument(
        "--no_demonstrations_only", action="store_false",
        dest="demonstrations_only",
    )
    args = parser.parse_args()

    in_place = args.output_dir is None
    output_dir = args.embeddings_dir if in_place else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – read all existing embedding shards
    # ------------------------------------------------------------------
    shard_paths = sorted(
        glob.glob(os.path.join(args.embeddings_dir, "*.tfrecord"))
    )
    if not shard_paths:
        logger.error(f"No *.tfrecord files found in {args.embeddings_dir}")
        return

    logger.info(f"Found {len(shard_paths)} shard(s) in {args.embeddings_dir}")

    # shard_data: { shard_path -> [(traj_id, raw_bytes), ...] }
    shard_data: dict[str, list[tuple[str, bytes]]] = {}
    # traj_id -> shard_path  (for quick lookup)
    traj_id_to_shard: dict[str, str] = {}

    for sp in shard_paths:
        logger.info(f"  Reading {os.path.basename(sp)} …")
        records = read_shard(sp)
        shard_data[sp] = records
        for tid, _ in records:
            traj_id_to_shard[tid] = sp

    total_trajs = len(traj_id_to_shard)
    logger.info(f"Total unique trajectories in embedding shards: {total_trajs}")

    # ------------------------------------------------------------------
    # Step 2 – stream DROID 1.0.1, collect action data for matching trajs
    # ------------------------------------------------------------------
    # action_store: { traj_id -> (actions_norm [T,8], delta_actions_norm [T,8],
    #                              is_successful, is_demonstration) }
    action_store: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}

    logger.info("Streaming DROID 1.0.1 to extract actions …")
    t_start = time.time()
    episode_count = 0
    matched = 0

    for episode in tqdm(
        iter_droid_trajectories(
            args.rlds_data_dir,
            split=args.split,
            num_parallel_reads=args.num_parallel_reads,
            prefetch=args.prefetch,
        ),
        desc="DROID episodes",
        unit="traj",
    ):
        episode_count += 1

        # Optionally skip non-demonstration trajectories
        if args.demonstrations_only:
            term = episode.get("is_terminal")
            if term is None or not bool(np.asarray(term).flat[-1]):
                continue

        # Extract traj_id
        try:
            ep_meta = episode.get("traj_metadata", {}).get("episode_metadata", {})
            traj_id = _extract_str(ep_meta.get("file_path", b""))
        except Exception:
            traj_id = None

        if not traj_id or traj_id not in traj_id_to_shard:
            continue

        if traj_id in action_store:
            continue  # already processed (shouldn't happen, but be safe)

        # ---- success flags -----------------------------------------------
        # is_successful: file_path contains "success"
        #   mirrors the regex filter in droid_verifier_rlds_dataset.py
        is_successful = 1 if traj_id and "success" in traj_id else 0

        # is_demonstration: last step has is_terminal=True
        #   mirrors --demonstrations_only in precompute_embeddings.py
        term = episode.get("is_terminal")
        is_demonstration = 1 if (term is not None and bool(np.asarray(term).flat[-1])) else 0

        # ---- actions: action_dict/joint_position + action_dict/gripper_position
        try:
            joint_pos = np.asarray(
                episode["action_dict"]["joint_position"], dtype=np.float32
            )   # [T, 7]
            gripper_pos = np.asarray(
                episode["action_dict"]["gripper_position"], dtype=np.float32
            )  # [T, 1]
        except KeyError as e:
            logger.warning(f"traj {traj_id}: missing action key {e}, skipping")
            continue

        # ---- state: observation/joint_position + observation/gripper_position
        try:
            obs_joint = np.asarray(
                episode["observation"]["joint_position"], dtype=np.float32
            )   # [T, 7]
            obs_gripper = np.asarray(
                episode["observation"]["gripper_position"], dtype=np.float32
            )  # [T, 1]
        except KeyError as e:
            logger.warning(f"traj {traj_id}: missing obs state key {e}, skipping")
            continue

        T = joint_pos.shape[0]
        if T == 0:
            logger.warning(f"traj {traj_id}: zero-length trajectory, skipping")
            continue

        # Concatenate to 8-D
        actions = np.concatenate([joint_pos, gripper_pos], axis=-1)   # [T, 8]
        state   = np.concatenate([obs_joint, obs_gripper], axis=-1)   # [T, 8]

        # Per-step delta (action[t] - state[t] for joints; gripper unchanged)
        delta_actions = _compute_per_step_delta(actions, state)

        # Normalise (always applied after delta conversion, matching training pipeline)
        actions_norm       = _normalize_actions_quantile(actions)
        delta_actions_norm = _normalize_actions_quantile(delta_actions)

        action_store[traj_id] = (actions_norm, delta_actions_norm, is_successful, is_demonstration)
        matched += 1

        if matched % 200 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"  Matched {matched}/{total_trajs} trajectories "
                f"({episode_count} episodes scanned, "
                f"{elapsed:.0f}s elapsed)"
            )

        if matched >= total_trajs:
            logger.info("All embedding trajectories matched – stopping DROID scan early.")
            break

    elapsed = time.time() - t_start
    logger.info(
        f"DROID scan complete: {episode_count} episodes, "
        f"{matched}/{total_trajs} matched in {elapsed:.0f}s"
    )

    missing = total_trajs - matched
    if missing > 0:
        logger.warning(
            f"{missing} trajectory/ies in embeddings could NOT be matched in DROID "
            f"(they will be written without action data, existing fields unchanged)."
        )

    # ------------------------------------------------------------------
    # Step 3 – rewrite each shard with the new action fields
    # ------------------------------------------------------------------
    logger.info("Rewriting shards …")
    total_written  = 0
    total_augmented = 0
    total_skipped   = 0

    for shard_path, records in shard_data.items():
        shard_name = os.path.basename(shard_path)
        output_path = os.path.join(output_dir, shard_name)

        if in_place:
            # Write to a temp file in the same directory, then atomically rename
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=output_dir, suffix=".tmp.tfrecord"
            )
            os.close(tmp_fd)
            write_path = tmp_path
        else:
            write_path = output_path

        augmented = 0
        skipped   = 0

        with tf.io.TFRecordWriter(write_path) as writer:
            for traj_id, raw_bytes in records:
                if traj_id in action_store:
                    actions_norm, delta_actions_norm, is_succ, is_demo = action_store[traj_id]
                    new_bytes = augment_proto(
                        raw_bytes, actions_norm, delta_actions_norm,
                        is_successful=is_succ, is_demonstration=is_demo,
                    )
                    writer.write(new_bytes)
                    augmented += 1
                else:
                    # No action data found – write original record unchanged
                    writer.write(raw_bytes)
                    skipped += 1

        if in_place:
            shutil.move(tmp_path, shard_path)
            logger.info(
                f"  {shard_name}: {augmented} augmented, "
                f"{skipped} written without actions (in-place)"
            )
        else:
            logger.info(
                f"  {shard_name}: {augmented} augmented, "
                f"{skipped} written without actions → {output_path}"
            )

        total_written   += augmented + skipped
        total_augmented += augmented
        total_skipped   += skipped

    elapsed_total = time.time() - t_start
    logger.info(
        f"Done. {total_written} records written total "
        f"({total_augmented} with action data, {total_skipped} without). "
        f"Total elapsed: {elapsed_total / 60:.1f} min."
    )
    logger.info(
        "New TFRecord fields added:\n"
        "  action_dim          int64   8\n"
        "  actions_bytes       bytes   float32 [T, 8]  normalised absolute actions\n"
        "  delta_actions_bytes bytes   float32 [T, 8]  normalised delta actions\n"
        "  is_successful       int64   1 if file_path contains 'success', else 0\n"
        "  is_demonstration    int64   1 if last step is_terminal=True, else 0"
    )


if __name__ == "__main__":
    main()
