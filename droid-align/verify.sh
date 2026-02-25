#!/usr/bin/env bash
# =============================================================================
# Step 1 verification: check precomputed embedding TFRecords
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CLIP-DROID

EMBEDDING_DIR="/root/data/droid_embeddings"
NUM_SAMPLES=200   # trajectories to inspect
K=8               # temporal offset to check feasibility for

python verify_embeddings.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --num_samples   "$NUM_SAMPLES" \
    --k             "$K"
