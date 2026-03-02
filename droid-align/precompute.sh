#!/usr/bin/env bash
# =============================================================================
# Steps 1 & 2: Precompute SigLIP2 embeddings + append delta actions for DROID
#
# Step 1 – precompute_embeddings.py
#   Pipeline per trajectory:
#     1. Parallel JPEG decode (32 CPU threads, all 3×T frames at once)
#     2. Frames queued into a CrossTrajectoryBuffer (accumulates across trajs)
#     3. On flush (buffer full): Stage A (float16 bilinear resize+pad → 224×224)
#                              + Stage B (float16 bicubic squash → 384×384 + norm)
#                              + encode_image — all float16 on GPU, one batch
#     4. Write completed trajectories to sharded TFRecords (float16 bytes)
#
#   Output fields per trajectory:
#     traj_id, num_steps, embed_dim, cameras_avail,
#     ext1_emb_bytes, ext2_emb_bytes, wrist_emb_bytes, lang_emb_bytes
#
# Step 2 – append_delta_actions.py
#   Streams DROID 1.0.1 to retrieve per-step actions and state, then rewrites
#   each shard in-place (atomic rename) with three new fields:
#     action_dim           int64
#     actions_bytes        float32 [T, 8]  normalised absolute actions
#     delta_actions_bytes  float32 [T, 8]  normalised delta actions
#                                          (action[t] - state[t], joints only)
#     is_successful        int64   1 if file_path contains "success"
#     is_demonstration     int64   1 if last step is_terminal=True
#
# H200 benchmark (ViT-L/16-SigLIP2-384):
#   cross-traj batch_size=4096 → ~81 GB / 144 GB VRAM  (~1200 img/s)
#   OMP_NUM_THREADS=1 mandatory on 192-core machines (prevents segfaults)
#
# Resume step 1: set START_SHARD > 0 to skip already-written shards.
# Resume step 2: safe to re-run — in-place atomic rewrite of existing shards.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# Activate conda (works in non-interactive shells)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CLIP-DROID

# Prevent OpenMP from spawning its own thread pool — conflicts with our
# ThreadPoolExecutor on 192-core machines and causes segfaults.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Paths  (edit these)
# ---------------------------------------------------------------------------
RLDS_DATA_DIR="/root/data"               # parent dir of droid/ TFDS folder
OUTPUT_DIR="/root/data/droid_embeddings" # where to write embedding TFRecords

# ---------------------------------------------------------------------------
# Step 1 config
# ---------------------------------------------------------------------------
NUM_SHARDS=256    # number of output .tfrecord shard files

# GPU batch size: frames accumulated across trajectories before one encode_image call.
# H200: 4096 → ~81 GB VRAM.  Reduce to 2048 if OOM.
BATCH_SIZE=3066

# RLDS I/O parallelism — the dominant bottleneck is disk reads.
# DROID has 2048 TFRecord shards; 16 readers balance throughput vs memory.
# Increase to 32+ on fast NVMe arrays.
NUM_PARALLEL_READS=32

# tf.data pipeline prefetch (trajectories buffered ahead of the main loop)
PREFETCH=32

DEVICE="cuda"
START_SHARD=0   # resume: set > 0 to skip already-written shards

# ---------------------------------------------------------------------------
# Step 2 config
# ---------------------------------------------------------------------------
# Set SKIP_APPEND=1 to run step 1 only (e.g. when appending separately).
SKIP_APPEND=${SKIP_APPEND:-0}

# ---------------------------------------------------------------------------
# Step 1: Precompute image + language embeddings
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Step 1: precompute_embeddings.py"
echo "  RLDS_DATA_DIR : $RLDS_DATA_DIR"
echo "  OUTPUT_DIR    : $OUTPUT_DIR"
echo "  num_shards    : $NUM_SHARDS  |  batch_size: $BATCH_SIZE"
echo "  start_shard   : $START_SHARD"
echo "============================================================"

python precompute_embeddings.py \
    --rlds_data_dir      "$RLDS_DATA_DIR" \
    --output_dir         "$OUTPUT_DIR" \
    --num_shards         "$NUM_SHARDS" \
    --batch_size         "$BATCH_SIZE" \
    --num_parallel_reads "$NUM_PARALLEL_READS" \
    --prefetch           "$PREFETCH" \
    --device             "$DEVICE" \
    --start_shard        "$START_SHARD"

echo ""
echo "Step 1 complete. Embeddings written to: $OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 2: Append per-step delta actions + success flags (in-place)
# ---------------------------------------------------------------------------
if [[ "$SKIP_APPEND" == "1" ]]; then
    echo ""
    echo "SKIP_APPEND=1 — skipping step 2 (append_delta_actions.py)."
    exit 0
fi

echo ""
echo "============================================================"
echo "Step 2: append_delta_actions.py"
echo "  embeddings_dir : $OUTPUT_DIR"
echo "  rlds_data_dir  : $RLDS_DATA_DIR"
echo "  (in-place rewrite via atomic rename)"
echo "============================================================"

python append_delta_actions.py \
    --embeddings_dir     "$OUTPUT_DIR" \
    --rlds_data_dir      "$RLDS_DATA_DIR" \
    --num_parallel_reads "$NUM_PARALLEL_READS" \
    --prefetch           "$PREFETCH"

echo ""
echo "Step 2 complete. Delta actions appended to: $OUTPUT_DIR"
echo ""
echo "New fields added per trajectory:"
echo "  action_dim           int64"
echo "  actions_bytes        float32 [T, 8]  normalised absolute actions"
echo "  delta_actions_bytes  float32 [T, 8]  normalised delta actions (a[t] - state[t])"
echo "  is_successful        int64   1 if file_path contains 'success'"
echo "  is_demonstration     int64   1 if last step is_terminal=True"
