#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# =============================================================================
# DROID Two-View Verifier Training Script
# Uses OpenPI RLDS loader with idle frame filtering and action normalization
# Step-based training (like PI0) - no epochs
# =============================================================================

# Data paths
RLDS_DATA_DIR="/root/data"                                    # Parent directory of droid/ TFDS folder
INSTRUCTION_MAPPING="filtered_droid_rephrases_16_with_alternates.json"        # Optional: instruction rephrase mapping

# Training config (PI0-style step-based)
python finetune_droid_two_view_ddp.py \
  --openpi_rlds_data_dir "$RLDS_DATA_DIR" \
  --openpi_action_chunk_size 16 \
  --filter_dict_path "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json" \
  --instruction_mapping "$INSTRUCTION_MAPPING" \
  --instruction_mode all_rephrases \
  --resume /root/vla-clip/bridge_verifier/droid_ckpts/droid_two_view_rlds_step_372000.pt \
  --override_lr \
  --max_rephrases 32 \
  --use_delta_actions \
  --use_transformer \
  --num_train_steps 20000000 \
  --batch_size 300 \
  --lr 1.0e-5 \
  --warmup_steps 0 \
  --train_log_freq 5 \
  --eval_log_freq 0 \
  --eval_steps 5000 \
  --eval_instruction_mode original \
  --save_interval 1000 \
  --checkpoint_dir droid_ckpts \
  --save_name droid_two_view_rlds \
  --max_checkpoints 100 \
  --save_examples_count 10 \
  --use_wandb \
  --world_size 8

# =============================================================================
# CLI Reference (all available options):
# =============================================================================
#
# Data Loading:
#   --openpi_rlds_data_dir PATH      Parent directory of droid/ TFDS folder (required)
#   --openpi_action_chunk_size N     Action chunk size, creates (2*N-1) window (default: 16 → 31 actions)
#   --filter_dict_path PATH          Idle frame filter JSON (default: gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json)
#                                    Set to "" to disable filtering
#
# Instruction Rephrasing:
#   --instruction_mapping PATH       Bridge-style rephrase mapping JSON (optional)
#   --instruction_mode MODE          original | random_rephrase | all_rephrases (default: all_rephrases)
#   --max_rephrases N                Max rephrases per sample including original (default: 8)
#
# Image Preprocessing:
#   --target_height N                Resize height with aspect ratio + padding (default: 224)
#   --target_width N                 Resize width with aspect ratio + padding (default: 224)
#   --resize_mode MODE               Interpolation: bilinear | nearest | bicubic (default: bilinear)
#
# Action Normalization:
#   --normalize_actions              Enable quantile normalization (default: enabled)
#   --no_normalize_actions           Disable normalization (use raw actions)
#
# Model:
#   --backbone MODEL                 SigLIP2 model (default: hf-hub:timm/ViT-L-16-SigLIP2-384)
#   --use_transformer                Use transformer encoder for actions (default: MLP)
#   --history_length N               Override action history length (default: infer from data)
#
# Training (step-based like PI0):
#   --num_train_steps N              Total training steps (default: 100000)
#   --batch_size N                   Per-GPU batch size (default: 64)
#   --lr FLOAT                       Learning rate (default: 1e-6)
#   --warmup_steps N                 Linear warmup steps (default: 1000)
#
# Logging & Evaluation:
#   --train_log_freq N               Log training metrics every N steps (default: 100)
#   --eval_log_freq N                Run eval every N steps, 0 to disable (default: 500)
#   --eval_steps N                   Number of eval batches per eval interval (default: 10)
#   --eval_instruction_mode MODE     Instruction mode for eval stream (default: original)
#   --use_wandb                      Enable Weights & Biases logging
#
# Checkpointing:
#   --save_interval N                Save checkpoint every N steps (default: 5000)
#   --checkpoint_dir PATH            Directory to save checkpoints (default: droid_trajectory_checkpoints_ddp)
#   --save_name NAME                 Name for checkpoints and wandb run
#   --resume PATH                    Resume from checkpoint
#   --override_lr                    Override learning rate from checkpoint with --lr value
#   --max_checkpoints N              Keep only N most recent checkpoints (default: 5)
#
# Distributed:
#   --world_size N                   Number of GPUs (default: all available)
#   --port N                         DDP communication port (default: 12355)
#
# Debugging:
#   --save_examples_dir PATH         Save sample visualizations before training
#   --save_examples_count N          Number of samples to save (default: 0)
#
# =============================================================================
