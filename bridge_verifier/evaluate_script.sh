#!/bin/bash
set -e

cd /root/vla-clip/bridge_verifier
mkdir -p logs

# Common parameters
DATASET="bridge_dataset_with_rephrases.json"
IMAGES_FOLDER="bridge_dataset_with_rephrases_images"
HISTORY_LENGTH=10
NUM_SAMPLES=50
ACTION_POOL_SIZE=20
USE_TRANSFORMER="--use_transformer"

# Model-to-backbone mapping
declare -A BACKBONES
BACKBONES["500m"]="hf-hub:timm/ViT-B-16-SigLIP2-384"
BACKBONES["1b"]="hf-hub:timm/ViT-L-16-SigLIP2-384"
BACKBONES["2b"]="hf-hub:timm/ViT-gopt-16-SigLIP2-384"

# Number of GPUs to use
NUM_GPUS=4
GPU_ID=0

# Function to launch a single evaluation
launch_eval() {
    local ckpt=$1
    local gpu_id=$2
    local backbone=$3
    local log_file="logs/eval_${ckpt%.pt}.log"

    echo "🚀 [GPU $gpu_id] Evaluating $ckpt with $backbone"
    echo "📝 Logging to $log_file"

    CUDA_VISIBLE_DEVICES=$gpu_id python3 vla_siglip_inference_bridge.py \
        --model_path "$ckpt" \
        --bridge_dataset "$DATASET" \
        --images_folder "$IMAGES_FOLDER" \
        --history_length "$HISTORY_LENGTH" \
        --num_samples "$NUM_SAMPLES" \
        --action_pool_size "$ACTION_POOL_SIZE" \
        $USE_TRANSFORMER \
        --backbone "$backbone" \
        > "$log_file" 2>&1 &
}

# Iterate through checkpoints
for ckpt in *.pt; do
    prefix=$(echo $ckpt | cut -d'_' -f1)
    backbone=${BACKBONES[$prefix]}

    if [ -z "$backbone" ]; then
        echo "⚠️  Skipping $ckpt (unknown prefix: $prefix)"
        continue
    fi

    # Assign GPU in round-robin fashion
    GPU_ID=$((GPU_ID % NUM_GPUS))

    launch_eval "$ckpt" "$GPU_ID" "$backbone"

    GPU_ID=$((GPU_ID + 1))
done

# Additional model (vla_clip variant)
echo "🚀 [GPU 0] Evaluating bridge_rephrases_epoch_20.pt with vla_clip_inference_bridge.py"
LOG_FILE="logs/eval_bridge_rephrases_epoch_20.log"
CUDA_VISIBLE_DEVICES=0 python3 vla_clip_inference_bridge.py \
    --model_path bridge_rephrases_epoch_20.pt \
    --bridge_dataset "$DATASET" \
    --images_folder "$IMAGES_FOLDER" \
    --history_length "$HISTORY_LENGTH" \
    --num_samples "$NUM_SAMPLES" \
    --action_pool_size "$ACTION_POOL_SIZE" \
    $USE_TRANSFORMER \
    > "$LOG_FILE" 2>&1 &

wait
echo "✅ All evaluations completed. Check logs/ for outputs."
