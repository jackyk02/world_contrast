#!/usr/bin/env bash
# =============================================================================
# Step 3: Temporal / Subgoal Alignment Training
#
# Trains TemporalAligner adapters (language, image) using the Triangle loss
# to align language instructions with current-frame embeddings (s_t) and
# predicted subgoal embeddings (s_{t+k}, k=8 steps ahead).
#
# Triangle loss: language is the anchor.
# Triplet:  (lang_i,  s_t_i,  s_{t+k}_i)  ← matched (small area)
# Negative: (lang_i,  s_t_j,  s_{t+k}_j)  ← mismatched (large area)
#
# Uses torchrun for multi-GPU DDP.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate CLIP-DROID

# ---------------------------------------------------------------------------
# Paths  (edit these)
# ---------------------------------------------------------------------------
EMBEDDING_DIR="/root/data/droid_embeddings"    # precomputed TFRecord dir
CHECKPOINT_DIR="/root/vla-clip/droid-align/temporal_ckpts"

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
NUM_GPUS=1
BATCH_SIZE=1024       # per-rank batch size; 1/3 of full dataset (~256 shards)
PROJ_DIM=512          # adapter output dimension
K=8                   # temporal offset: use s_{t+8} as subgoal
CAMERA="ext1"         # camera for s_t and s_{t+k}: ext1 | ext2 | wrist
LR=6e-4               # linear scaling: 3e-4 * (1024/512)
WARMUP_STEPS=500
NUM_TRAIN_STEPS=67000  # scaled for ~1/3 of full dataset
SAVE_INTERVAL=5000
LOG_FREQ=50
AUX_LOSS_WEIGHT=0.5   # weight for the auxiliary (st-anchored) triangle loss
PORT=12357

# Optional: resume from checkpoint
RESUME=""
# RESUME="$CHECKPOINT_DIR/droid_temporal_align_step_10000.pt"

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    train_temporal_align.py \
    --embedding_dir    "$EMBEDDING_DIR" \
    --checkpoint_dir   "$CHECKPOINT_DIR" \
    --save_name        "droid_temporal_align" \
    --k                "$K" \
    --camera           "$CAMERA" \
    --proj_dim         "$PROJ_DIM" \
    --batch_size       "$BATCH_SIZE" \
    --lr               "$LR" \
    --warmup_steps     "$WARMUP_STEPS" \
    --num_train_steps  "$NUM_TRAIN_STEPS" \
    --save_interval    "$SAVE_INTERVAL" \
    --log_freq         "$LOG_FREQ" \
    --aux_loss_weight  "$AUX_LOSS_WEIGHT" \
    --shuffle_buffer   8192 \
    --num_workers      4 \
    --max_checkpoints  20 \
    --label_smoothing  0.1 \
    --grad_clip        1.0 \
    --shard_start      100 \
    --shard_end        256 \
    --port             "$PORT" \
    ${RESUME:+--resume "$RESUME"}

# =============================================================================
# CLI Reference (all available flags):
# =============================================================================
#
# Data:
#   --embedding_dir PATH    Precomputed embedding TFRecord directory (required)
#   --embed_dim N           SigLIP2 embedding dimension (default: 1024)
#   --k N                   Temporal offset in steps (default: 8)
#   --camera NAME           Camera for s_t/s_{t+k}: ext1|ext2|wrist (default: ext1)
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
#   --aux_loss_weight FLOAT Weight for auxiliary (st-anchored) loss (default: 0.5)
#                           Set to 0.0 to use only the language-anchored loss.
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
#   --port N                DDP communication port (default: 12357)
#
# =============================================================================
