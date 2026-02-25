#!/usr/bin/env python3
"""
Multi-view Camera Alignment Training (DROID)
=============================================
Trains lightweight MLP adapter heads on top of frozen precomputed SigLIP2
embeddings using the Triangle loss to align three camera views:
  - exterior_image_1  (ext1)
  - exterior_image_2  (ext2)
  - wrist_image       (wrist)

The symmetric Triangle loss minimises the area of the triangle formed by a
matched (ext1_i, ext2_i, wrist_i) triplet while pushing mismatched triplets
to form larger triangles.  All three anchor permutations are averaged for a
fully symmetric objective.

Architecture:
  Three separate ViewAdapter(1024 → proj_dim) MLPs, one per camera.
  Each adapter: Linear → GELU → LayerNorm → Linear → L2-normalise.

DDP is used for multi-GPU training (torch.distributed.launch / torchrun).

Usage (see train_multiview.sh):
  torchrun --nproc_per_node 4 train_multiview_align.py \\
      --embedding_dir /root/data/droid_embeddings \\
      --checkpoint_dir multiview_ckpts \\
      ...
"""

import os
os.environ.setdefault("OMP_NUM_THREADS",   "8")
os.environ.setdefault("MKL_NUM_THREADS",   "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from triangle_loss import symmetric_triangle_loss, triangle_loss
from embedding_dataset import MultiViewDataset

torch.set_num_threads(8)


# ---------------------------------------------------------------------------
# Model: per-view adapter heads
# ---------------------------------------------------------------------------
class ViewAdapter(nn.Module):
    """
    Two-layer MLP adapter that maps a frozen SigLIP2 embedding to a learned
    alignment space.

    Input:  [B, input_dim]  (float32, L2-normalised SigLIP2 embeddings)
    Output: [B, output_dim] (float32, L2-normalised projected embeddings)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
        )
        # Initialise final layer near identity to stabilise early training
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MultiViewAligner(nn.Module):
    """Three ViewAdapters (ext1, ext2, wrist) + learnable temperature."""

    def __init__(self, input_dim: int = 1024, proj_dim: int = 512):
        super().__init__()
        self.ext1_adapter  = ViewAdapter(input_dim, proj_dim)
        self.ext2_adapter  = ViewAdapter(input_dim, proj_dim)
        self.wrist_adapter = ViewAdapter(input_dim, proj_dim)
        # Initialise log-temperature to log(1/0.07) ≈ 2.66
        self.log_temp = nn.Parameter(torch.tensor(2.6592))

    @property
    def temperature(self) -> float:
        return float(self.log_temp.exp().clamp(0.01, 100.0))

    def forward(
        self,
        ext1_emb:  torch.Tensor,
        ext2_emb:  torch.Tensor,
        wrist_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.ext1_adapter(ext1_emb)
        z2 = self.ext2_adapter(ext2_emb)
        zw = self.wrist_adapter(wrist_emb)
        return z1, z2, zw


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_ddp(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def gather_embeddings(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather embeddings from all ranks to form a larger negative batch."""
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def manage_checkpoints(ckpt_dir: str, save_name: str, max_keep: int):
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, f"{save_name}_step_*.pt")))
    while len(ckpts) > max_keep:
        oldest = ckpts.pop(0)
        os.remove(oldest)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(rank: int, world_size: int, args: argparse.Namespace):
    is_ddp = world_size > 1
    if is_ddp:
        setup_ddp(rank, world_size, args.port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    dataset = MultiViewDataset(
        embedding_dir=args.embedding_dir,
        shuffle_shards=True,
        shuffle_buffer=args.shuffle_buffer,
        require_all_cameras=True,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- Model ----
    model = MultiViewAligner(input_dim=args.embed_dim, proj_dim=args.proj_dim).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[rank])

    # ---- Optimiser & scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(1, args.warmup_steps)
        # Cosine decay to 10% of peak
        progress = (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Resume ----
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if is_ddp else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        if rank == 0:
            print(f"Resumed from {args.resume} at step {global_step}")

    # ---- Wandb ----
    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project="droid-multiview-align", name=args.save_name, config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Training loop ----
    model.train()
    loss_accum = 0.0
    t_start = time.time()

    for ext1_emb, ext2_emb, wrist_emb in loader:
        if global_step >= args.num_train_steps:
            break

        ext1_emb  = ext1_emb.to(device)
        ext2_emb  = ext2_emb.to(device)
        wrist_emb = wrist_emb.to(device)

        # Forward
        raw_model = model.module if is_ddp else model
        z1, z2, zw = model(ext1_emb, ext2_emb, wrist_emb)

        # Cross-rank negatives for larger effective batch
        if is_ddp and world_size > 1:
            z1_all = gather_embeddings(z1, world_size)
            z2_all = gather_embeddings(z2, world_size)
            zw_all = gather_embeddings(zw, world_size)
        else:
            z1_all, z2_all, zw_all = z1, z2, zw

        temp = raw_model.temperature

        loss = symmetric_triangle_loss(
            z1_all, z2_all, zw_all,
            temperature=temp,
            label_smoothing=args.label_smoothing,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_accum += float(loss.item())
        global_step += 1

        # ---- Logging ----
        if global_step % args.log_freq == 0 and rank == 0:
            avg_loss = loss_accum / args.log_freq
            loss_accum = 0.0
            elapsed = time.time() - t_start
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step:8d} | loss={avg_loss:.4f} | "
                f"temp={temp:.4f} | lr={lr_now:.2e} | "
                f"elapsed={elapsed/60:.1f}m"
            )
            if args.use_wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/temp": temp,
                           "train/lr": lr_now}, step=global_step)

        # ---- Checkpointing ----
        if global_step % args.save_interval == 0 and rank == 0:
            raw_model = model.module if is_ddp else model
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"{args.save_name}_step_{global_step}.pt"
            )
            torch.save(
                {
                    "model_state_dict":     raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step":          global_step,
                    "args":                 vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
            manage_checkpoints(args.checkpoint_dir, args.save_name, args.max_checkpoints)

        if global_step % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if is_ddp:
        dist.destroy_process_group()

    if rank == 0:
        elapsed = time.time() - t_start
        print(f"Training complete. {global_step} steps in {elapsed/60:.1f} min.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-view camera alignment with Triangle loss")

    # Data
    parser.add_argument("--embedding_dir",  required=True, help="Precomputed embedding TFRecord dir")
    parser.add_argument("--embed_dim",       type=int, default=1024, help="SigLIP2 embedding dim")
    parser.add_argument("--shuffle_buffer",  type=int, default=8192, help="Shuffle buffer size")
    parser.add_argument("--num_workers",     type=int, default=4,    help="DataLoader workers")

    # Model
    parser.add_argument("--proj_dim",        type=int, default=512,  help="Adapter projection dim")

    # Training
    parser.add_argument("--num_train_steps", type=int,   default=200_000)
    parser.add_argument("--batch_size",      type=int,   default=512,   help="Per-rank batch size")
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--warmup_steps",    type=int,   default=1000)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    # Logging
    parser.add_argument("--log_freq",        type=int, default=50)
    parser.add_argument("--use_wandb",       action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint_dir",  default="multiview_ckpts")
    parser.add_argument("--save_name",       default="droid_multiview_align")
    parser.add_argument("--save_interval",   type=int, default=5000)
    parser.add_argument("--max_checkpoints", type=int, default=10)
    parser.add_argument("--resume",          default=None)

    # DDP
    parser.add_argument("--world_size",      type=int, default=1)
    parser.add_argument("--port",            type=int, default=12356)

    args = parser.parse_args()

    # torchrun sets LOCAL_RANK automatically
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if world_size > 1:
        train(local_rank, world_size, args)
    else:
        train(0, 1, args)


if __name__ == "__main__":
    main()
