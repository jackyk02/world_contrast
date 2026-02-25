#!/usr/bin/env bash
# =============================================================================
# Step 1: Precompute ViT-L/16-SigLIP2-384 embeddings for DROID
#
# Reads raw DROID RLDS trajectories and writes per-trajectory TFRecords
# containing float16 embeddings for ext1, ext2, wrist, and language.
#
# Run time: ~12-24 hours for the full DROID dataset on a single A100.
# Can resume: already-completed shards are skipped (--start_shard).
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# Activate conda environment (works in non-interactive shells)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CLIP-DROID

# Prevent OpenMP from spawning its own thread pool (conflicts with ThreadPoolExecutor
# on 192-core machines and causes segfaults / resource exhaustion).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Paths  (edit these)
# ---------------------------------------------------------------------------
RLDS_DATA_DIR="/root/data"                     # parent dir of droid/ TFDS folder
OUTPUT_DIR="/root/data/droid_embeddings"       # where to write embedding TFRecords

# ---------------------------------------------------------------------------
# Preprocessing config
# ---------------------------------------------------------------------------
NUM_SHARDS=256          # output shard count (more shards → smaller files)
# H200 benchmark results (ViT-L/16-SigLIP2-384):
#   GPU max batch  : 4096 imgs (81 GB / 144 GB VRAM)
#   Optimal gpu_bs : 512  (~280-300 img/s end-to-end incl. CPU JPEG decode)
#   CPU workers    : 32   (OMP_NUM_THREADS=1 mandatory on 192-core machines)
IMG_BATCH_SIZE=512
DEVICE="cuda"
START_SHARD=0           # set > 0 to resume from a given shard

python precompute_embeddings.py \
    --rlds_data_dir  "$RLDS_DATA_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --num_shards     "$NUM_SHARDS" \
    --img_batch_size "$IMG_BATCH_SIZE" \
    --device         "$DEVICE" \
    --start_shard    "$START_SHARD"

echo ""
echo "Preprocessing complete. Embeddings written to: $OUTPUT_DIR"
