#!/usr/bin/env bash
# =============================================================================
# Multi-view Camera Alignment with FiLM Conditioning
#
# For each precomputed embedding (ext1, ext2, wrist):
#   1. L2 normalize v and t (language)
#   2. Apply FiLM conditioning: gamma, beta = FiLM(t); v_film = gamma*v + beta
#   3. Re-normalize v_film
#   4. ViewAdapter(v_film) → aligned embedding
#
# Uses MultiViewWithLangDataset (ext1, ext2, wrist, lang) and symmetric
# Triangle loss. Same validation and checkpointing as train_multiview.sh.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CLIP-DROID

# ---------------------------------------------------------------------------
# Paths  (edit these)
# ---------------------------------------------------------------------------
EMBEDDING_DIR="/root/data/droid_embeddings"    # precomputed TFRecord dir
CHECKPOINT_DIR="/root/vla-clip/droid-align/multiview_film_ckpts"

# ---------------------------------------------------------------------------
# Training config (same as train_multiview.sh; only model + dataset differ)
# ---------------------------------------------------------------------------
NUM_GPUS=1
BATCH_SIZE=1024       # per-rank batch size; 1/3 of full dataset (~256 shards)
PROJ_DIM=1024         # adapter output dimension
LR=6e-3               # linear scaling: 3e-4 * (1024/512)
WARMUP_STEPS=500
NUM_TRAIN_STEPS=10000 # scaled for ~1/3 of full dataset
SAVE_INTERVAL=2500
LOG_FREQ=50
VAL_INTERVAL=500      # validate every N steps (0 = disabled)
VAL_BATCHES=50       # mini-batches per validation run
VAL_SHARD_START=100  # validation uses single shard 100
VAL_SHARD_END=101
GRAD_CLIP=5.0        # 1.0 can over-clip; 5.0 often helps break plateau
LABEL_SMOOTHING=0.05 # 0.1 softens targets; 0.05 or 0 can lower loss further
PORT=12356

# Optional: resume from checkpoint
RESUME=""
# RESUME="$CHECKPOINT_DIR/droid_multiview_film_step_10000.pt"

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    train_multiview_film.py \
    --embedding_dir    "$EMBEDDING_DIR" \
    --checkpoint_dir   "$CHECKPOINT_DIR" \
    --save_name        "droid_multiview_film" \
    --proj_dim         "$PROJ_DIM" \
    --batch_size       "$BATCH_SIZE" \
    --lr               "$LR" \
    --warmup_steps     "$WARMUP_STEPS" \
    --num_train_steps  "$NUM_TRAIN_STEPS" \
    --save_interval    "$SAVE_INTERVAL" \
    --log_freq         "$LOG_FREQ" \
    --val_interval     "$VAL_INTERVAL" \
    --val_batches      "$VAL_BATCHES" \
    --val_shard_start  "$VAL_SHARD_START" \
    --val_shard_end    "$VAL_SHARD_END" \
    --shuffle_buffer   20000 \
    --num_workers      4 \
    --max_checkpoints  20 \
    --label_smoothing  "$LABEL_SMOOTHING" \
    --grad_clip        "$GRAD_CLIP" \
    --shard_start      101 \
    --shard_end        256 \
    --port             "$PORT" \
    --use_wandb \
    ${RESUME:+--resume "$RESUME"}
