#!/usr/bin/env python3
"""
DDP training for VLA-SigLIP2 on DROID two-view dataset with cross-rank contrastive loss.
"""

# === CRITICAL: Set thread limits BEFORE importing numerical libraries ===
# This prevents thread oversubscription in multi-GPU training
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
# TensorFlow (used by RLDS dataset)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"

import argparse
import gc
import time
import json
import glob
from typing import Optional, Dict, Any, List, Tuple, Set

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from open_clip import create_model_from_pretrained, get_tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from PIL import Image

from model import AttentionPooling, ModelConfig, TextAwareVisualExtraction, sincos_position_embedding
from openpi_rlds_two_view import OpenPiRldsTwoViewBatchedIterable, OpenPiRldsTwoViewConfig

PADDING_VALUE = -5.0

# PyTorch threading (matches environment variable settings above)
torch.set_num_threads(8)

class VLA_SigLIP2_Droid(nn.Module):
    """
    Same architecture as Bridge variant; only data differs (two-view images).
    """

    def __init__(self, model_config, use_transformer: bool = False):
        super().__init__()
        self.model = model_config.clip_model
        for param in self.model.parameters():
            param.requires_grad = False
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        # Keep the frozen SigLIP2 backbone in eval mode (disables dropout, etc.).
        # Note: calling .train() on the outer module (e.g., DDP) would otherwise
        # recursively set this submodule back to train mode.
        self.model.eval()

        text_pooling_output_dim = model_config.text_pooling_output_dim
        pooling_heads = model_config.pooling_heads
        pooling_layers = model_config.pooling_layers
        self.num_readouts = model_config.num_readouts

        text_dim = self.model.text.output_dim
        vision_dim = self.model.visual.trunk.num_features
        vision_pooling_output_dim = model_config.vision_pooling_output_dim

        self.visual_patch_size = self.model.visual.trunk.patch_embed.proj.kernel_size[0]
        image_size = self.model.visual.image_size[0] if hasattr(self.model.visual, "image_size") else 224
        self.num_img_patches = (image_size // self.visual_patch_size) ** 2

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.text_aware_visual_extraction = TextAwareVisualExtraction(
            num_img_patches=self.num_img_patches,
            vision_dim=vision_dim,
        )
        self.text_pooling = AttentionPooling(
            text_dim,
            text_pooling_output_dim,
            pooling_heads,
            pooling_layers,
            num_readouts=self.num_readouts,
        )
        self.vision_poolings = AttentionPooling(
            vision_dim,
            vision_pooling_output_dim,
            pooling_heads,
            pooling_layers,
            num_readouts=self.num_readouts,
        )

        self.f_t_dim = text_pooling_output_dim + (2 * vision_pooling_output_dim)
        self.input_projection = nn.Linear(self.f_t_dim, vision_pooling_output_dim)

        self.action_dim = model_config.action_dim
        self.history_length = model_config.history_length
        self.use_transformer = use_transformer

        if self.use_transformer:
            self.single_step_action_encoder = nn.Linear(self.action_dim, vision_pooling_output_dim)
            self.register_buffer(
                "action_pos_embedding",
                sincos_position_embedding(self.history_length, vision_pooling_output_dim).unsqueeze(0),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=vision_pooling_output_dim,
                nhead=8,
                dim_feedforward=vision_pooling_output_dim * 2,
                dropout=0.1,
            )
            self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        else:
            self.complex_action_encoder = nn.Sequential(
                nn.Linear(self.history_length * self.action_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, vision_pooling_output_dim),
            )

        self.activation = {}

        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output

            return hook

        self.hooks = [
            self.model.visual.trunk.blocks[-1].attn.register_forward_hook(
                get_activation("image_patches")
            ),
            self.model.text.transformer.register_forward_hook(
                get_activation("text_features")
            ),
        ]

        self.action_padding_value = PADDING_VALUE

    def train(self, mode: bool = True):
        """
        Put trainable heads in train/eval, but always keep the frozen SigLIP2 backbone in eval mode.
        This avoids stochastic layers (e.g., dropout) corrupting the cached-hook features.
        """
        super().train(mode)
        self.model.eval()
        return self

    def set_trainable_dtype(self, dtype=torch.float32):
        for name, module in self.named_modules():
            if "model." in name:
                continue
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.requires_grad:
                    module.weight.data = module.weight.data.to(dtype)
            if hasattr(module, "bias") and module.bias is not None:
                if module.bias is not None and module.bias.requires_grad:
                    module.bias.data = module.bias.data.to(dtype)
        return self

    def extract_features(self, ext_images, wrist_images, text):
        ext_images = ext_images.to(torch.bfloat16)
        wrist_images = wrist_images.to(torch.bfloat16)
        stacked_images = torch.cat([ext_images, wrist_images], dim=0)
        with torch.no_grad():
            _ = self.model.encode_text(text, normalize=False)
            _ = self.model.encode_image(stacked_images, normalize=False)

        text_features = self.activation["text_features"]
        text_features = self.model.text.ln_final(text_features)
        if hasattr(self.model.text, "text_projection") and self.model.text.text_projection is not None:
            batch_size, seq_len, hidden_dim = text_features.shape
            text_features = self.model.text.text_projection(text_features.reshape(-1, hidden_dim))
            text_features = text_features.reshape(batch_size, seq_len, -1)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        patch_features = self.activation["image_patches"]
        b2 = patch_features.shape[0]
        b = b2 // 2
        ext_patch_features = patch_features[:b]
        wrist_patch_features = patch_features[b:]
        if patch_features.shape[1] == self.num_img_patches + 1:
            ext_patch_features = ext_patch_features[:, 1:, :]
            wrist_patch_features = wrist_patch_features[:, 1:, :]
        elif patch_features.shape[1] == self.num_img_patches:
            pass
        else:
            ext_patch_features = ext_patch_features[:, 1:, :]
            wrist_patch_features = wrist_patch_features[:, 1:, :]

        ext_patch_features = ext_patch_features.float()
        wrist_patch_features = wrist_patch_features.float()
        ext_patch_features = ext_patch_features / ext_patch_features.norm(dim=-1, keepdim=True)
        wrist_patch_features = wrist_patch_features / wrist_patch_features.norm(dim=-1, keepdim=True)
        
        # Clear activation cache to prevent memory accumulation
        self.activation.clear()
        
        return ext_patch_features, wrist_patch_features, text_features

    def forward_features(self, ext_image, wrist_image, text, action_histories):
        ext_patch_features, wrist_patch_features, text_features = self.extract_features(ext_image, wrist_image, text)

        ext_tav_features = self.text_aware_visual_extraction(ext_patch_features, text_features)
        wrist_tav_features = self.text_aware_visual_extraction(wrist_patch_features, text_features)
        ext_vision_token = self.vision_poolings(ext_tav_features)
        wrist_vision_token = self.vision_poolings(wrist_tav_features)
        vision_token = torch.cat([ext_vision_token, wrist_vision_token], dim=-1)

        text_token = self.text_pooling(text_features)
        combined_features = torch.cat([text_token, vision_token], dim=-1)
        combined_features = self.input_projection(combined_features)
        combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)

        action_histories = action_histories.float().to(ext_image.device)

        if self.use_transformer:
            padding_mask = (action_histories[:, :, 0] == self.action_padding_value)
            encoded_steps = self.single_step_action_encoder(action_histories)
            # Add positional embedding
            encoded_steps = encoded_steps + self.action_pos_embedding
            encoded_steps_permuted = encoded_steps.permute(1, 0, 2)
            transformer_output_permuted = self.trajectory_encoder(
                encoded_steps_permuted, src_key_padding_mask=padding_mask
            )
            transformer_output = transformer_output_permuted.permute(1, 0, 2)
            mask_expanded = (~padding_mask).unsqueeze(-1).float()
            summed_features = (transformer_output * mask_expanded).sum(dim=1)
            num_non_padded = mask_expanded.sum(dim=1)
            num_non_padded = torch.clamp(num_non_padded, min=1e-9)
            projected_trajectory = summed_features / num_non_padded
        else:
            batch_size = action_histories.shape[0]
            flat_actions = action_histories.reshape(batch_size, -1)
            projected_trajectory = self.complex_action_encoder(flat_actions)

        projected_trajectory = projected_trajectory / projected_trajectory.norm(dim=-1, keepdim=True)
        
        return combined_features, projected_trajectory

    def forward(self, ext_image, wrist_image, text, action_histories):
        """
        Full forward pass computing logits from features.
        For backward compatibility and single-GPU training.
        """
        combined_features, projected_trajectory = self.forward_features(ext_image, wrist_image, text, action_histories)
        
        logits_scale = self.logit_scale.exp()
        image_logits = logits_scale * torch.matmul(combined_features, projected_trajectory.T)
        action_logits = logits_scale * torch.matmul(projected_trajectory, combined_features.T)

        return image_logits, action_logits


def setup_distributed(rank, world_size, port=12355):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}"),
    )


def cleanup_distributed():
    dist.destroy_process_group()


def calculate_accuracy_metrics(logits_per_image, logits_per_action, labels, device, k_values=[1, 5]):
    """
    Calculate top-k accuracy for contrastive learning.
    
    Args:
        logits_per_image: [local_batch_size, global_batch_size] similarity scores
        logits_per_action: [local_batch_size, global_batch_size] similarity scores
        labels: [local_batch_size] ground truth indices in global batch
        device: torch device
        k_values: list of k values for top-k accuracy
    """
    metrics = {}
    global_batch_size = logits_per_image.shape[1]

    for k in k_values:
        if k <= global_batch_size:
            _, topk_indices = torch.topk(logits_per_image, k, dim=1)
            # Check if true label is in top-k predictions for each sample
            correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
            accuracy = correct.any(dim=1).float().mean().item()
            metrics[f"img2act_top{k}_acc"] = accuracy

    for k in k_values:
        if k <= global_batch_size:
            _, topk_indices = torch.topk(logits_per_action, k, dim=1)
            correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
            accuracy = correct.any(dim=1).float().mean().item()
            metrics[f"act2img_top{k}_acc"] = accuracy

    return metrics


def get_gpu_metrics(device):
    if torch.cuda.is_available() and device.type == "cuda":
        gpu_id = device.index
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
        return {
            "gpu_memory_allocated_gb": allocated,
            "gpu_memory_reserved_gb": reserved,
            "gpu_memory_max_allocated_gb": max_allocated,
            "gpu_id": gpu_id,
        }
    return {}


def calculate_gradient_metrics(model):
    total_norm = 0
    param_count = 0
    grad_count = 0

    for p in model.parameters():
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += p.numel()
            grad_count += 1

    total_norm = total_norm ** 0.5
    return {
        "grad_norm": total_norm,
        "grad_param_count": param_count,
        "grad_layer_count": grad_count,
    }


def gather_features_with_gradient(features, rank, world_size):
    """
    Gather features from all ranks while preserving gradient flow.
    
    For contrastive learning, we need all negatives from all GPUs, but gradients
    should only flow back to the local features.
    
    Args:
        features: Local features tensor [B, D]
        rank: Current process rank
        world_size: Total number of processes
    
    Returns:
        Gathered features [B*world_size, D] with proper gradient flow
    """
    if world_size == 1 or not dist.is_initialized():
        return features
    
    # Gather features from all ranks
    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_features, features)
    
    # Replace the gathered feature at current rank with the local one
    # This ensures gradients flow back correctly to local features
    gathered_features[rank] = features
    
    # Concatenate all features
    all_features = torch.cat(gathered_features, dim=0)
    
    return all_features


def manage_checkpoints(checkpoint_dir, save_name, max_checkpoints):
    import glob
    import re

    # Match step-based checkpoint names (e.g., save_name_step_5000.pt)
    pattern = os.path.join(checkpoint_dir, f"{save_name}_step_*.pt")
    checkpoints = glob.glob(pattern)
    
    # Exclude final checkpoints from deletion
    checkpoints = [c for c in checkpoints if "_final.pt" not in c]
    
    if len(checkpoints) <= max_checkpoints:
        return

    checkpoint_info = []
    for ckpt_path in checkpoints:
        match = re.search(r"_step_(\d+)\.pt$", ckpt_path)
        if match:
            step_num = int(match.group(1))
            checkpoint_info.append((step_num, ckpt_path))

    checkpoint_info.sort(key=lambda x: x[0])
    num_to_delete = len(checkpoint_info) - max_checkpoints
    for i in range(num_to_delete):
        _, ckpt_path = checkpoint_info[i]
        try:
            os.remove(ckpt_path)
            print(f"Deleted old checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Warning: Could not delete checkpoint {ckpt_path}: {e}")


def _extract_preprocess_norm(preprocess):
    default_mean = [0.48145466, 0.4578275, 0.40821073]
    default_std = [0.26862954, 0.26130258, 0.27577711]
    try:
        transforms = getattr(preprocess, "transforms", None)
        if transforms:
            for t in transforms:
                if hasattr(t, "mean") and hasattr(t, "std"):
                    mean = [float(m) for m in t.mean]
                    std = [float(s) for s in t.std]
                    return mean, std
    except Exception:
        pass
    return default_mean, default_std


def _tensor_to_pil(image_tensor, mean, std):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    tensor = image_tensor.detach().cpu().float()
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    tensor = (tensor * std_t) + mean_t
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def save_dataset_examples(
    dataset,
    preprocess,
    output_dir: str,
    num_examples: int,
    padding_value: float,
    history_length: Optional[int],
    rank: int,
):
    """
    Save examples from the processed dataloader (as they appear in training).
    The dataloader yields batches, so we iterate through batch items.
    """
    os.makedirs(output_dir, exist_ok=True)
    mean, std = _extract_preprocess_norm(preprocess)
    saved = 0
    iterator = iter(dataset)

    while saved < num_examples:
        try:
            batch = next(iterator)
        except StopIteration:
            if saved == 0:
                print(f"Rank {rank}: No samples available to save.")
            break
        except Exception as e:
            print(f"Rank {rank}: Error while reading batch: {e}")
            break

        if batch is None:
            continue

        # Unpack batch (dataset yields batches, not individual samples)
        if len(batch) == 4:
            ext_imgs, wrist_imgs, captions, actions_batch = batch
        elif len(batch) == 3:
            ext_imgs, captions, actions_batch = batch
            wrist_imgs = None
        else:
            print(f"Rank {rank}: Unexpected batch format (len={len(batch)}), skipping.")
            break

        # Iterate through batch items
        batch_size = ext_imgs.shape[0] if torch.is_tensor(ext_imgs) else len(captions)
        for i in range(batch_size):
            if saved >= num_examples:
                break

            # Extract single item from batch
            ext_img = ext_imgs[i]
            wrist_img = wrist_imgs[i] if wrist_imgs is not None else None
            caption = captions[i] if isinstance(captions, list) else captions
            actions = actions_batch[i]

            actions_np = actions.detach().cpu().numpy()
            non_padding_mask = ~(np.isclose(actions_np, padding_value).all(axis=-1))
            metadata = {
                "caption": caption,  # Now a single string!
                "actions_shape": list(actions_np.shape),
                "non_padding_steps": int(non_padding_mask.sum()),
                "padding_value": padding_value,
                "history_length": history_length,
                "actions": actions_np.tolist(),
            }

            try:
                ext_pil = _tensor_to_pil(ext_img, mean, std)
                ext_path = os.path.join(output_dir, f"example_{saved}_ext.png")
                ext_pil.save(ext_path)
                metadata["ext_image"] = ext_path

                if wrist_img is not None:
                    wrist_pil = _tensor_to_pil(wrist_img, mean, std)
                    wrist_path = os.path.join(output_dir, f"example_{saved}_wrist.png")
                    wrist_pil.save(wrist_path)
                    metadata["wrist_image"] = wrist_path

                meta_path = os.path.join(output_dir, f"example_{saved}.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                saved += 1
            except Exception as e:
                print(f"Rank {rank}: Failed to save example {saved}: {e}")
                continue

        if saved >= num_examples:
            break

    if saved > 0:
        print(f"Rank {rank}: Saved {saved} dataset example(s) to {output_dir}")


def train_droid_ddp(
    rank: int,
    world_size: int,
    openpi_rlds_data_dir: Optional[str],
    openpi_action_chunk_size: int,
    filter_dict_path: Optional[str],
    instruction_mapping: Optional[str],
    instruction_mode: str,
    max_rephrases: int,
    target_height: int,
    target_width: int,
    resize_mode: str,
    use_delta_actions: bool,
    history_length: Optional[int],
    backbone: str,
    num_train_steps: int,
    batch_size: int,
    learning_rate: float,
    save_name=None,
    checkpoint_dir="checkpoints",
    use_wandb=False,
    resume_checkpoint=None,
    use_transformer=False,
    port=12355,
    warmup_steps=1000,
    train_log_freq=50,
    eval_log_freq=500,
    eval_steps: int = 10,
    eval_instruction_mode: str = "original",
    save_interval: int = 5000,
    max_checkpoints=5,
    example_save_dir: Optional[str] = None,
    example_save_count: int = 0,
    override_lr: bool = False,
):
    setup_distributed(rank, world_size, port)
    device = torch.device(f"cuda:{rank}")

    # NOTE: We intentionally avoid barriers in shutdown paths; on Ctrl-C ranks can exit at different times.
    try:
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        dist.barrier(device_ids=[rank])

        torch.manual_seed(42 + rank)
        np.random.seed(42 + rank)

        if rank == 0:
            print(f"Loading SigLIP2 model: {backbone}")
        siglip2_model, preprocess = create_model_from_pretrained(backbone)
        siglip2_model = siglip2_model.to(device)
        tokenizer = get_tokenizer(backbone)

        if not openpi_rlds_data_dir:
            if rank == 0:
                print("Error: --openpi_rlds_data_dir must be provided (other data loaders were removed).")
            return None

        # Train stream (shuffle=True)
        dataset = OpenPiRldsTwoViewBatchedIterable(
            OpenPiRldsTwoViewConfig(
                data_dir=openpi_rlds_data_dir,
                per_rank_batch_size=batch_size,
                action_chunk_size=openpi_action_chunk_size,
                expected_action_dim=8,
                shuffle=True,
                filter_dict_path=filter_dict_path,
                preprocess=preprocess,
                instruction_mapping_path=instruction_mapping,
                instruction_mode=instruction_mode,
                max_rephrases=max_rephrases,
                target_height=target_height,
                target_width=target_width,
                resize_mode=resize_mode,
                use_delta_actions=use_delta_actions,
                rank=rank,
                world_size=world_size,
            )
        )

        # Probe first batch to infer dimensions
        _, _, _, first_actions = next(iter(dataset))
        action_dim = int(first_actions.shape[-1])
        assert action_dim == 8, f"Expected 8-D actions but got action_dim={action_dim}"
        history_length = int(first_actions.shape[-2])
        padding_value = PADDING_VALUE

        if rank == 0:
            print(f"OpenPI RLDS: action_dim={action_dim}, history_length={history_length}, per_rank_batch_size={batch_size}")
            if filter_dict_path:
                print(f"Idle frame filtering: ENABLED (using {filter_dict_path})")
            else:
                print("Idle frame filtering: DISABLED (training on all frames including idle)")
            print("Action normalization: ALWAYS ENABLED (joints: q01/q99 -> [-1,1], gripper: binarized to 0/1)")
            if use_delta_actions:
                print("Delta actions: ENABLED (action - state for joints, computed BEFORE normalization, matching OpenPI/PI0)")
            else:
                print("Delta actions: DISABLED (using absolute actions, then normalized)")

        # Optional sample dump
        if rank == 0 and example_save_dir and example_save_count > 0:
            save_dataset_examples(
                dataset=dataset,
                preprocess=preprocess,
                output_dir=example_save_dir,
                num_examples=example_save_count,
                padding_value=padding_value,
                history_length=history_length,
                rank=rank,
            )

        if save_name is None:
            save_name = f"droid_two_view_h{history_length}_{'transformer' if use_transformer else 'mlp'}_ddp"

        if use_wandb and rank == 0:
            import wandb

            wandb.init(project="VLA-SigLIP2-Droid-DDP", name=save_name)
            wandb.config.update(
                {
                    "backbone": backbone,
                    "learning_rate": learning_rate,
                    "num_train_steps": num_train_steps,
                    "batch_size": batch_size,
                    "world_size": world_size,
                    "device": f"cuda:{rank}",
                    "history_length": history_length,
                    "use_transformer": use_transformer,
                    "warmup_steps": warmup_steps,
                    "save_interval": save_interval,
                    "train_log_freq": train_log_freq,
                    "eval_log_freq": eval_log_freq,
                }
            )

        if rank == 0:
            print(f"Streaming mode: training for {num_train_steps} total steps.")

        if eval_log_freq > 0:
            # Eval stream (shuffle=False)
            eval_dataloader = OpenPiRldsTwoViewBatchedIterable(
                OpenPiRldsTwoViewConfig(
                    data_dir=openpi_rlds_data_dir,
                    per_rank_batch_size=batch_size,
                    action_chunk_size=openpi_action_chunk_size,
                    expected_action_dim=8,
                    shuffle=False,
                    filter_dict_path=filter_dict_path,
                    preprocess=preprocess,
                    instruction_mapping_path=instruction_mapping,
                    instruction_mode=eval_instruction_mode,
                    max_rephrases=max_rephrases,
                    target_height=target_height,
                    target_width=target_width,
                    resize_mode=resize_mode,
                    use_delta_actions=use_delta_actions,
                    rank=rank,
                    world_size=world_size,
                )
            )

        model_config = ModelConfig(clip_model=siglip2_model, history_length=history_length, action_dim=action_dim)
        model = VLA_SigLIP2_Droid(model_config, use_transformer=use_transformer).to(device)
        model.action_padding_value = padding_value
        model.set_trainable_dtype(torch.float32)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

        global_step = 0
        optimizer_state = None
        scheduler_state = None

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Rank {rank}: Loading checkpoint from {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            target_model = model.module if isinstance(model, DDP) else model

            # Use strict=False to handle old checkpoints missing new buffers (e.g., action_pos_embedding).
            # Fixed buffers like sinusoidal positional embeddings will use freshly initialized values.
            missing_keys, unexpected_keys = target_model.load_state_dict(state_dict, strict=False)
            if missing_keys and rank == 0:
                print(f"Rank {rank}: Checkpoint missing keys (using initialized values): {missing_keys}")
            if unexpected_keys and rank == 0:
                print(f"Rank {rank}: Checkpoint has unexpected keys (ignored): {unexpected_keys}")
            global_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0
            optimizer_state = checkpoint.get("optimizer_state_dict", None) if isinstance(checkpoint, dict) else None
            scheduler_state = checkpoint.get("scheduler_state_dict", None) if isinstance(checkpoint, dict) else None
            print(f"Rank {rank}: Loaded checkpoint from step {global_step}")

        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
                print(f"Rank {rank}: Successfully loaded optimizer state.")
            except Exception as e:
                print(f"Rank {rank}: Warning - Could not load optimizer state: {e}")

        warmup_steps_resolved = max(1, int(warmup_steps))

        def warmup_lambda(step: int):
            return float(step) / float(warmup_steps_resolved) if step < warmup_steps_resolved else 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        if scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
                print(f"Rank {rank}: Successfully loaded scheduler state.")
            except Exception as e:
                print(f"Rank {rank}: Warning - Could not load scheduler state: {e}")

        # Override scheduler base_lrs as well (load_state_dict restores saved base_lrs)
        if override_lr:
            scheduler.base_lrs = [learning_rate for _ in scheduler.base_lrs]
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            if rank == 0:
                print(f"Overriding scheduler base_lrs to {learning_rate}")

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Warmup steps: {warmup_steps_resolved}, Total training steps: {num_train_steps}")
            print(f"Training log frequency: every {train_log_freq} steps")
            print(f"Checkpoint save interval: every {save_interval} steps")

        model.train()
        train_iter = iter(dataset)
        interval_train_loss = 0.0
        interval_image_loss = 0.0
        interval_action_loss = 0.0
        interval_grad_norm = 0.0
        interval_img2act_top1 = 0.0
        interval_act2img_top1 = 0.0
        interval_batch_count = 0
        interval_start_time = time.time()

        if rank == 0:
            step_pbar = tqdm(range(global_step, num_train_steps), desc="Training", initial=global_step, total=num_train_steps)
        else:
            step_pbar = range(global_step, num_train_steps)

        try:
            for _ in step_pbar:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(dataset)
                    batch = next(train_iter)

                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    ext_img, wrist_img, texts, action_hists = batch
                    ext_img = ext_img.to(device)
                    wrist_img = wrist_img.to(device)
                else:
                    img, texts, action_hists = batch
                    ext_img = img.to(device)
                    wrist_img = img.to(device)
                
                del batch # Free batch container

                input_texts = tokenizer(texts, context_length=siglip2_model.context_length).to(device)
                input_actions = action_hists.to(device, dtype=torch.float32)
                current_batch_size = ext_img.shape[0]

                optimizer.zero_grad()
                
                # Extract features and gather from all ranks for global contrastive loss
                combined_features, projected_trajectory = model.module.forward_features(
                    ext_img, wrist_img, input_texts, input_actions
                )
                
                # Gather features from all GPUs to use all samples as negatives
                combined_features_all = gather_features_with_gradient(combined_features, rank, world_size)
                projected_trajectory_all = gather_features_with_gradient(projected_trajectory, rank, world_size)
                
                # Compute logits with global negatives
                logit_scale = model.module.logit_scale.exp()
                logits_per_image = logit_scale * torch.matmul(combined_features, projected_trajectory_all.T)
                logits_per_action = logit_scale * torch.matmul(projected_trajectory, combined_features_all.T)
                
                # Labels account for rank offset in global batch
                positive_labels = torch.arange(current_batch_size, device=device) + rank * current_batch_size
                
                image_loss = F.cross_entropy(logits_per_image, positive_labels)
                action_loss = F.cross_entropy(logits_per_action, positive_labels)
                loss = (image_loss + action_loss) / 2.0
                loss.backward()

                grad_metrics = calculate_gradient_metrics(model)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                # Calculate accuracy before deleting logits
                acc_metrics = calculate_accuracy_metrics(logits_per_image, logits_per_action, positive_labels, device)

                # Keep numeric values before deleting tensors
                loss_val = float(loss.item())
                img_loss_val = float(image_loss.item())
                act_loss_val = float(action_loss.item())

                # Free large tensors
                del combined_features_all, projected_trajectory_all, logits_per_image, logits_per_action
                del ext_img, wrist_img, input_texts, input_actions, loss, image_loss, action_loss

                interval_train_loss += loss_val
                interval_image_loss += img_loss_val
                interval_action_loss += act_loss_val
                interval_grad_norm += float(grad_metrics["grad_norm"])
                interval_img2act_top1 += float(acc_metrics.get("img2act_top1_acc", 0.0))
                interval_act2img_top1 += float(acc_metrics.get("act2img_top1_acc", 0.0))
                interval_batch_count += 1

                if rank == 0 and global_step % train_log_freq == 0 and interval_batch_count > 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logit_scale = model.module.logit_scale.exp().item()
                    gpu_metrics = get_gpu_metrics(device)

                    avg_loss = interval_train_loss / interval_batch_count
                    avg_img_loss = interval_image_loss / interval_batch_count
                    avg_act_loss = interval_action_loss / interval_batch_count
                    avg_grad = interval_grad_norm / interval_batch_count
                    avg_img2act = interval_img2act_top1 / interval_batch_count
                    avg_act2img = interval_act2img_top1 / interval_batch_count
                    interval_time = time.time() - interval_start_time
                    samples_per_sec = (interval_batch_count * batch_size * world_size) / max(interval_time, 1e-6)

                    log_dict = {
                        "step": global_step,
                        "learning_rate": current_lr,
                        "train/loss": avg_loss,
                        "train/image_loss": avg_img_loss,
                        "train/action_loss": avg_act_loss,
                        "train/grad_norm": avg_grad,
                        "train/img2act_top1_acc": avg_img2act,
                        "train/act2img_top1_acc": avg_act2img,
                        "model/logit_scale": logit_scale,
                        "timing/samples_per_sec": samples_per_sec,
                    }
                    for key, value in gpu_metrics.items():
                        log_dict[f"gpu/{key}"] = value
                    if use_wandb:
                        import wandb

                        wandb.log(log_dict)
                    print(
                        f"Step {global_step}/{num_train_steps}: Loss={avg_loss:.4f}, LR={current_lr:.2e}, "
                        f"ImgAcc={avg_img2act:.3f}, ActAcc={avg_act2img:.3f}, {samples_per_sec:.1f} samples/s"
                    )

                    interval_train_loss = 0.0
                    interval_image_loss = 0.0
                    interval_action_loss = 0.0
                    interval_grad_norm = 0.0
                    interval_img2act_top1 = 0.0
                    interval_act2img_top1 = 0.0
                    interval_batch_count = 0
                    interval_start_time = time.time()

                if rank == 0:
                    step_pbar.set_postfix(
                        {
                            "loss": f"{loss_val:.4f}",
                            "img_acc": f"{acc_metrics.get('img2act_top1_acc', 0):.3f}",
                            "act_acc": f"{acc_metrics.get('act2img_top1_acc', 0):.3f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        }
                    )

                # Periodic garbage collection and cache cleanup
                if global_step % train_log_freq == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                if eval_log_freq > 0 and (global_step % eval_log_freq == 0):
                    # Release training batch from GPU memory before eval to avoid OOM
                    del ext_img, wrist_img, input_texts, input_actions
                    torch.cuda.empty_cache()
                    
                    model.eval()
                    eval_loss_sum = 0.0
                    eval_img2act_top1_sum = 0.0
                    eval_act2img_top1_sum = 0.0
                    eval_batches = 0
                    with torch.no_grad():
                        eval_iter = iter(eval_dataloader)
                        for _ in range(max(0, int(eval_steps))):
                            ext_img_e, wrist_img_e, texts_e, action_hists_e = next(eval_iter)
                            ext_img_e = ext_img_e.to(device)
                            wrist_img_e = wrist_img_e.to(device)
                            texts_tok_e = tokenizer(texts_e, context_length=siglip2_model.context_length).to(device)
                            action_hists_e = action_hists_e.to(device, dtype=torch.float32)

                            bs_e = ext_img_e.shape[0]
                            
                            # Use global negatives for eval too
                            combined_features_e, projected_trajectory_e = model.module.forward_features(
                                ext_img_e, wrist_img_e, texts_tok_e, action_hists_e
                            )
                            combined_features_all_e = gather_features_with_gradient(combined_features_e, rank, world_size)
                            projected_trajectory_all_e = gather_features_with_gradient(projected_trajectory_e, rank, world_size)
                            
                            logit_scale_e = model.module.logit_scale.exp()
                            logits_img_e = logit_scale_e * torch.matmul(combined_features_e, projected_trajectory_all_e.T)
                            logits_act_e = logit_scale_e * torch.matmul(projected_trajectory_e, combined_features_all_e.T)
                            
                            labels_e = torch.arange(bs_e, device=device) + rank * bs_e
                            image_loss_e = F.cross_entropy(logits_img_e, labels_e)
                            action_loss_e = F.cross_entropy(logits_act_e, labels_e)
                            loss_e = (image_loss_e + action_loss_e) / 2.0
                            
                            acc_e = calculate_accuracy_metrics(logits_img_e, logits_act_e, labels_e, device, k_values=[1])
                            eval_loss_sum += float(loss_e.item())
                            eval_img2act_top1_sum += float(acc_e.get("img2act_top1_acc", 0.0))
                            eval_act2img_top1_sum += float(acc_e.get("act2img_top1_acc", 0.0))
                            eval_batches += 1
                            

                    metrics = torch.tensor(
                        [eval_loss_sum, eval_img2act_top1_sum, eval_act2img_top1_sum, float(eval_batches)],
                        device=device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    loss_sum_all, img2act_sum_all, act2img_sum_all, batches_all = metrics.tolist()
                    batches_all = max(1.0, batches_all)
                    if rank == 0:
                        eval_payload = {
                            "eval/loss": loss_sum_all / batches_all,
                            "eval/img2act_top1_acc": img2act_sum_all / batches_all,
                            "eval/act2img_top1_acc": act2img_sum_all / batches_all,
                        }
                        if use_wandb:
                            import wandb

                            wandb.log(eval_payload, step=global_step)
                        print(
                            f"Eval step {global_step}: loss={eval_payload['eval/loss']:.4f} "
                            f"img2act@1={eval_payload['eval/img2act_top1_acc']:.3f} "
                            f"act2img@1={eval_payload['eval/act2img_top1_acc']:.3f}"
                        )
                    
                    # Clean up eval iterator and tensors to release memory
                    del eval_iter
                    torch.cuda.empty_cache()
                    model.train()

                if save_interval > 0 and global_step % save_interval == 0:
                    if rank == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_step_{global_step}.pt")
                        torch.save(
                            {
                                "model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "global_step": global_step,
                            },
                            checkpoint_path,
                        )
                        print(f"Checkpoint saved at {checkpoint_path}")
                        manage_checkpoints(checkpoint_dir, save_name, max_checkpoints)
                    dist.barrier(device_ids=[rank])

        except KeyboardInterrupt:
            # Best-effort: save an interrupted checkpoint from rank0.
            if rank == 0:
                try:
                    ckpt_path = os.path.join(checkpoint_dir, f"{save_name}_step_{global_step}_interrupted.pt")
                    torch.save(
                        {
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "global_step": global_step,
                            "interrupted": True,
                        },
                        ckpt_path,
                    )
                    print(f"\nInterrupted: saved checkpoint at {ckpt_path}")
                except Exception as e:
                    print(f"\nInterrupted: failed to save checkpoint: {e}")
            return None

        # Final save
        if rank == 0:
            final_checkpoint_path = os.path.join(checkpoint_dir, f"{save_name}_step_{global_step}_final.pt")
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                },
                final_checkpoint_path,
            )
            print(f"Final checkpoint saved at {final_checkpoint_path}")

        if use_wandb and rank == 0:
            import wandb

            wandb.finish()

        return model.module if rank == 0 else None
    finally:
        # Always tear down the process group to avoid NCCL resource leak warnings on exit.
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def ddp_main(rank, world_size, args):
    # Allow disabling filter by passing empty string
    filter_dict_path = args.filter_dict_path if args.filter_dict_path else None
    model = train_droid_ddp(
        rank=rank,
        world_size=world_size,
        openpi_rlds_data_dir=args.openpi_rlds_data_dir,
        openpi_action_chunk_size=args.openpi_action_chunk_size,
        filter_dict_path=filter_dict_path,
        instruction_mapping=args.instruction_mapping,
        instruction_mode=args.instruction_mode,
        max_rephrases=args.max_rephrases,
        target_height=args.target_height,
        target_width=args.target_width,
        resize_mode=args.resize_mode,
        use_delta_actions=args.use_delta_actions,
        history_length=args.history_length,
        backbone=args.backbone,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_name=args.save_name,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
        resume_checkpoint=args.resume,
        use_transformer=args.use_transformer,
        port=args.port,
        warmup_steps=args.warmup_steps,
        train_log_freq=args.train_log_freq,
        eval_log_freq=args.eval_log_freq,
        eval_steps=args.eval_steps,
        eval_instruction_mode=args.eval_instruction_mode,
        save_interval=args.save_interval,
        max_checkpoints=args.max_checkpoints,
        example_save_dir=args.save_examples_dir,
        example_save_count=args.save_examples_count,
        override_lr=args.override_lr,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VLA-SigLIP2 model for DROID two-view dataset with action trajectories and contrastive loss using DDP"
    )
    parser.add_argument("--num_train_steps", type=int, default=100000, help="Total number of training steps (like PI0)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (per GPU)")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")

    parser.add_argument(
        "--backbone",
        type=str,
        default="hf-hub:timm/ViT-L-16-SigLIP2-384",
        help="SigLIP2 model backbone (e.g., hf-hub:timm/ViT-B-16-SigLIP2-256, hf-hub:timm/ViT-L-16-SigLIP2-384)",
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=None,
        help="Action history length (if omitted, inferred from dataset metadata)",
    )
    parser.add_argument("--use_transformer", action="store_true", help="Use transformer for action history encoding instead of MLP")

    parser.add_argument(
        "--openpi_rlds_data_dir",
        type=str,
        default=None,
        help="If set, use OpenPI RLDS DROID loader (openpi/training/droid_verifier_rlds_dataset.py). "
        "This should be the parent directory of the `droid/` TFDS folder.",
    )
    parser.add_argument(
        "--openpi_action_chunk_size",
        type=int,
        default=16,
        help="Action chunk size for OpenPI RLDS loader. With history+future chunking, total window is (2*chunk-1).",
    )
    parser.add_argument(
        "--filter_dict_path",
        type=str,
        default="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        help="Path to JSON file mapping episodes to non-idle frame ranges. Set to empty string to disable idle filtering.",
    )
    parser.add_argument(
        "--instruction_mapping",
        type=str,
        default=None,
        help="Optional Bridge-style instruction rephrase mapping JSON. If set, can expand/replace prompts per sample.",
    )
    parser.add_argument(
        "--instruction_mode",
        type=str,
        default="all_rephrases",
        choices=["original", "random_rephrase", "all_rephrases"],
        help="How to apply instruction_mapping: original (no change), random_rephrase (keep batch size), "
        "all_rephrases (use all rephrases but shuffle them across batches while keeping batch size fixed).",
    )
    parser.add_argument(
        "--max_rephrases",
        type=int,
        default=8,
        help="Maximum number of rephrases to use per sample (including original). Only applies to all_rephrases mode.",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=224,
        help="Resize images to this height (with aspect ratio preserved and padding) BEFORE model preprocess.",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=224,
        help="Resize images to this width (with aspect ratio preserved and padding) BEFORE model preprocess.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="bilinear",
        help="Interpolation mode for resize+pad step (before model preprocess).",
    )
    parser.add_argument(
        "--use_delta_actions",
        action="store_true",
        default=False,
        help="Convert actions to deltas (action - state). Applied BEFORE normalization (matching OpenPI/PI0). Helps reduce action magnitudes.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="droid_trajectory_checkpoints_ddp", help="Directory to save checkpoints")
    parser.add_argument("--save_name", type=str, default=None, help="Name for saved model and wandb run (generated if None)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint state_dict to resume training from")
    parser.add_argument("--override_lr", action="store_true", help="Override learning rate from checkpoint with --lr value")

    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=12355, help="Port for distributed communication")

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of linear warmup steps")
    parser.add_argument("--train_log_freq", type=int, default=100, help="Log training metrics every N steps")
    parser.add_argument("--eval_log_freq", type=int, default=500, help="Run evaluation every N steps (0 to disable)")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of eval batches to run each eval interval")
    parser.add_argument(
        "--eval_instruction_mode",
        type=str,
        default="original",
        choices=["original", "random_rephrase", "all_rephrases"],
        help="Instruction mode used for the eval stream (default: original).",
    )
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument(
        "--max_checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep (older ones are deleted)"
    )
    parser.add_argument(
        "--save_examples_dir",
        type=str,
        default="examples",
        help="If set, dumps a few dataset samples (rank0) before training.",
    )
    parser.add_argument(
        "--save_examples_count",
        type=int,
        default=0,
        help="How many samples to dump before training starts (rank0 only).",
    )

    args = parser.parse_args()

    if args.world_size > torch.cuda.device_count():
        print(f"Error: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available.")
        exit(1)

    if args.world_size < 1:
        print("Error: world_size must be at least 1.")
        exit(1)

    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed. Running without wandb logging.")
            args.use_wandb = False

    history_length = args.history_length
    history_print = history_length if history_length is not None else "infer-from-dataset/stream"
    print("Starting DDP training with SigLIP2 (DROID two-view)...")
    print(f"Backbone: {args.backbone}")
    print(
        f"Config: History={history_print}, ActionEncoder={'Transformer' if args.use_transformer else 'MLP'}, LR={args.lr}, BS={args.batch_size}"
    )
    print(f"Training: {args.num_train_steps} steps, warmup={args.warmup_steps} steps, save_interval={args.save_interval} steps")
    print(f"DDP Config: World Size={args.world_size}, Port={args.port}")
    print(f"Logging: train_log_freq={args.train_log_freq}, eval_log_freq={args.eval_log_freq}")
    print(f"Using wandb: {args.use_wandb}")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.world_size == 1:
        print("Running on single GPU (no multiprocessing)")
        finetuned_model = ddp_main(0, 1, args)
    else:
        print(f"Spawning {args.world_size} processes for DDP training")
        try:
            mp.spawn(
                ddp_main,
                args=(args.world_size, args),
                nprocs=args.world_size,
                join=True,
            )
            finetuned_model = None
        except Exception as e:
            print(f"Error during multiprocessing spawn: {e}")
            print("Try reducing batch size or world size.")
            raise

    if finetuned_model is not None:
        FINAL_SAVE_PATH = os.path.join(args.checkpoint_dir, f"{args.save_name or 'droid_two_view'}_final_best.pt")
        print(f"Saving final model (best validation weights) to {FINAL_SAVE_PATH}...")
        torch.save(finetuned_model.state_dict(), FINAL_SAVE_PATH)
        print("Done!")
    else:
        print("DDP training completed. Check checkpoint directory for saved models.")
