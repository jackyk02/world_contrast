#!/usr/bin/env bash
# =============================================================================
# Step 3: Temporal / Subgoal Alignment Training
#
# Trains ActionAlignmentModel adapters (image, action) using the Triangle loss
# to align current-frame embeddings (s_t), delta action chunks (a_{t:t+k-1}),
# and predicted subgoal embeddings (s_{t+k}, k=8 steps ahead).
#
# Triangle loss: s_t is the primary anchor.
# Triplet:  (s_t_i,  a_{t:t+k-1}_i,  s_{t+k}_i)  ← matched (small area)
# Negative: (s_t_i,  a_{t:t+k-1}_j,  s_{t+k}_j)  ← mismatched (large area)
#
# Optional auxiliary loss uses action chunk as anchor:
# Triplet:  (a_{t:t+k-1}_i,  s_t_i,  s_{t+k}_i)  ← matched
#
# Requires: delta_actions_bytes in TFRecords (run append_delta_actions.py first).
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
PROJ_DIM=512          # shared projection dim (img adapter + transformer d_model)
LR=3e-4
K=8                   # temporal offset: use s_{t+8} as subgoal; action chunk length
CAMERA="ext1"         # camera for s_t and s_{t+k}: ext1 | ext2 | wrist

# Action Transformer encoder (matches finetune_droid_two_view_ddp.py)
NHEAD=8               # attention heads  (proj_dim must be divisible by nhead)
NUM_ENCODER_LAYERS=4  # TransformerEncoder depth
DROPOUT=0.1           # dropout in TransformerEncoderLayer

WARMUP_STEPS=2000
NUM_TRAIN_STEPS=670000  # scaled for ~1/3 of full dataset
SAVE_INTERVAL=5000
LOG_FREQ=50
VAL_INTERVAL=500   # validate every N steps (0 = disabled)
VAL_BATCHES=50      # mini-batches per validation run
VAL_SHARD_START=100 # validation uses single shard 100
VAL_SHARD_END=101
LABEL_SMOOTHING=0.05
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
    --save_name        "droid_action_temporal_align" \
    --k                "$K" \
    --camera           "$CAMERA" \
    --proj_dim         "$PROJ_DIM" \
    --nhead            "$NHEAD" \
    --num_encoder_layers "$NUM_ENCODER_LAYERS" \
    --dropout          "$DROPOUT" \
    --batch_size       "$BATCH_SIZE" \
    --lr               "$LR" \
    --warmup_steps     "$WARMUP_STEPS" \
    --num_train_steps  "$NUM_TRAIN_STEPS" \
    --save_interval    "$SAVE_INTERVAL" \
    --log_freq         "$LOG_FREQ" \
    --aux_loss_weight  "$AUX_LOSS_WEIGHT" \
    --val_interval     "$VAL_INTERVAL" \
    --val_batches      "$VAL_BATCHES" \
    --val_shard_start  "$VAL_SHARD_START" \
    --val_shard_end    "$VAL_SHARD_END" \
    --shuffle_buffer   40000 \
    --num_workers      8 \
    --max_checkpoints  20 \
    --label_smoothing  "$LABEL_SMOOTHING" \
    --grad_clip        5.0 \
    --shard_start      101 \
    --shard_end        256 \
    --port             "$PORT" \
    --use_wandb \
    ${RESUME:+--resume "$RESUME"}

# =============================================================================
# CLI Reference (all available flags):
# =============================================================================
#
# Data:
#   --embedding_dir PATH    Precomputed embedding TFRecord directory (required)
#                           Must have delta_actions_bytes (run append_delta_actions.py)
#   --embed_dim N           SigLIP2 embedding dimension (default: 1024)
#   --action_dim N          Action dim per step (default: 8)
#   --k N                   Temporal offset = action chunk length (default: 8)
#   --camera NAME           Camera for s_t/s_{t+k}: ext1|ext2|wrist (default: ext1)
#   --shuffle_buffer N      In-memory shuffle buffer size (default: 8192)
#   --num_workers N         DataLoader worker processes (default: 4)
#   --successful_only       Only train on is_successful=1 trajectories
#
# Model:
#   --proj_dim N            Shared projection dim = Transformer d_model (default: 512)
#                           Must be divisible by --nhead
#   --nhead N               Attention heads in TransformerEncoder (default: 8)
#   --num_encoder_layers N  TransformerEncoder depth (default: 4)
#   --dropout FLOAT         Dropout in TransformerEncoderLayer (default: 0.1)
#
# Training:
#   --num_train_steps N     Total training steps (default: 200000)
#   --batch_size N          Per-GPU batch size (default: 512)
#   --lr FLOAT              Learning rate (default: 3e-4)
#   --weight_decay FLOAT    AdamW weight decay (default: 1e-4)
#   --warmup_steps N        Linear LR warmup steps (default: 1000)
#   --grad_clip FLOAT       Gradient clipping norm (default: 1.0)
#   --label_smoothing FLOAT Label smoothing for cross-entropy (default: 0.1)
#   --aux_loss_weight FLOAT Weight for action-anchored auxiliary loss (default: 0.5)
#                           Set to 0.0 to use only the s_t-anchored loss.
#
# Logging:
#   --log_freq N            Log every N steps (default: 50)
#   --use_wandb             Enable Weights & Biases logging
#   --val_interval N        Validate every N steps; 0 = disabled (default: 1000)
#   --val_batches N         Mini-batches per validation run (default: 50)
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
