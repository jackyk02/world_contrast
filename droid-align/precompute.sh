#!/usr/bin/env bash
# =============================================================================
# Step 1: Precompute ViT-L/16-SigLIP2-384 embeddings for DROID
#
# Pipeline per trajectory:
#   1. Parallel JPEG decode (32 CPU threads, all 3×T frames at once)
#   2. Frames queued into a CrossTrajectoryBuffer (accumulates across trajs)
#   3. On flush (buffer full): Stage A (float16 bilinear resize+pad → 224×224)
#                            + Stage B (float16 bicubic squash → 384×384 + norm)
#                            + encode_image — all float16 on GPU, one batch
#   4. Write completed trajectories to sharded TFRecords (float16 bytes)
#
# H200 benchmark (ViT-L/16-SigLIP2-384):
#   cross-traj batch_size=4096 → ~81 GB / 144 GB VRAM  (~1200 img/s)
#   OMP_NUM_THREADS=1 mandatory on 192-core machines (prevents segfaults)
#
# Resume: set START_SHARD > 0 to skip already-written shards.
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
# Config
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
START_SHARD=100   # resume: shards 0-99 are complete; shards 100-255 will be rewritten

mkdir -p "$OUTPUT_DIR"

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
echo "Preprocessing complete. Embeddings written to: $OUTPUT_DIR"
