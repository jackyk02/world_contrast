#!/usr/bin/env bash
# =============================================================================
# Step 2: Multi-view Camera Alignment Training
#
# Trains ViewAdapter heads (ext1, ext2, wrist) using the symmetric Triangle
# loss so that all three camera views of the same scene are aligned in a
# shared embedding space.
#
# Uses torchrun for multi-GPU DDP.  Adjust --nproc_per_node to your GPU count.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CLIP-DROID

# ---------------------------------------------------------------------------
# Paths  (edit these)
# ---------------------------------------------------------------------------
EMBEDDING_DIR="/root/data/droid_embeddings"    # precomputed TFRecord dir
CHECKPOINT_DIR="/root/vla-clip/droid-align/multiview_ckpts"

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
NUM_GPUS=4
BATCH_SIZE=512        # per-rank batch size; effective = NUM_GPUS * BATCH_SIZE
PROJ_DIM=512          # adapter output dimension
LR=3e-4
WARMUP_STEPS=1000
NUM_TRAIN_STEPS=200000
SAVE_INTERVAL=5000
LOG_FREQ=50
PORT=12356

# Optional: resume from checkpoint
RESUME=""
# RESUME="$CHECKPOINT_DIR/droid_multiview_align_step_10000.pt"

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    train_multiview_align.py \
    --embedding_dir    "$EMBEDDING_DIR" \
    --checkpoint_dir   "$CHECKPOINT_DIR" \
    --save_name        "droid_multiview_align" \
    --proj_dim         "$PROJ_DIM" \
    --batch_size       "$BATCH_SIZE" \
    --lr               "$LR" \
    --warmup_steps     "$WARMUP_STEPS" \
    --num_train_steps  "$NUM_TRAIN_STEPS" \
    --save_interval    "$SAVE_INTERVAL" \
    --log_freq         "$LOG_FREQ" \
    --shuffle_buffer   8192 \
    --num_workers      4 \
    --max_checkpoints  20 \
    --label_smoothing  0.1 \
    --grad_clip        1.0 \
    --port             "$PORT" \
    ${RESUME:+--resume "$RESUME"} \
    --use_wandb

# =============================================================================
# CLI Reference (all available flags):
# =============================================================================
#
# Data:
#   --embedding_dir PATH    Precomputed embedding TFRecord directory (required)
#   --embed_dim N           SigLIP2 embedding dimension (default: 1024)
#   --shuffle_buffer N      In-memory shuffle buffer size (default: 8192)
#   --num_workers N         DataLoader worker processes (default: 4)
#
# Model:
#   --proj_dim N            Adapter MLP output dimension (default: 512)
#
# Training:
#   --num_train_steps N     Total training steps (default: 200000)
#   --batch_size N          Per-GPU batch size (default: 512)
#   --lr FLOAT              Learning rate (default: 3e-4)
#   --weight_decay FLOAT    AdamW weight decay (default: 1e-4)
#   --warmup_steps N        Linear LR warmup steps (default: 1000)
#   --grad_clip FLOAT       Gradient clipping norm (default: 1.0)
#   --label_smoothing FLOAT Label smoothing for cross-entropy (default: 0.1)
#
# Logging:
#   --log_freq N            Log every N steps (default: 50)
#   --use_wandb             Enable Weights & Biases logging
#
# Checkpointing:
#   --checkpoint_dir PATH   Checkpoint output directory
#   --save_name NAME        Checkpoint filename prefix
#   --save_interval N       Save every N steps (default: 5000)
#   --max_checkpoints N     Keep only N most recent checkpoints (default: 10)
#   --resume PATH           Resume from checkpoint file
#
# DDP:
#   --world_size N          Number of GPUs (auto-detected from torchrun)
#   --port N                DDP communication port (default: 12356)
#
# =============================================================================
