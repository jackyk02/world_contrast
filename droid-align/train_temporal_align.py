#!/usr/bin/env python3
"""
Temporal / Subgoal Alignment Training (DROID)
=============================================
Trains lightweight MLP adapter heads on top of frozen precomputed SigLIP2
embeddings using the Triangle loss to align:

  - language_instruction  (text embedding, anchor)
  - s_t                   (current frame image embedding)
  - s_{t+k}               (predicted subgoal frame, k steps ahead; default k=8)

The Triangle loss with language as the anchor minimises the area of the
triangle formed by matched (lang_i, st_i, stk_i) while maximising areas for
mismatched triplets.  This teaches the model to align:
  1. The instruction with the current state  (what am I doing now?)
  2. The instruction with the subgoal state  (what will I achieve?)
  3. Implicitly, the current state and subgoal state via the shared anchor.

Architecture:
  LangAdapter:  1024 → proj_dim  (language)
  ImgAdapter:   1024 → proj_dim  (shared for both s_t and s_{t+k})
  Learnable temperature.

DDP is used for multi-GPU training.

Usage (see train_temporal.sh):
  torchrun --nproc_per_node 4 train_temporal_align.py \\
      --embedding_dir /root/data/droid_embeddings \\
      --checkpoint_dir temporal_ckpts \\
      --k 8 \\
      ...
"""

import os
os.environ.setdefault("OMP_NUM_THREADS",   "8")
os.environ.setdefault("MKL_NUM_THREADS",   "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from triangle_loss import triangle_loss
from embedding_dataset import TemporalDataset

torch.set_num_threads(8)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModalityAdapter(nn.Module):
    """
    Two-layer MLP projection head.

    Input:  [B, input_dim]   (float32, L2-normalised SigLIP2 embeddings)
    Output: [B, output_dim]  (float32, L2-normalised projected embeddings)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TemporalAligner(nn.Module):
    """
    Language adapter + shared image adapter (for both s_t and s_{t+k}).
    Learnable temperature.
    """

    def __init__(self, input_dim: int = 1024, proj_dim: int = 512):
        super().__init__()
        self.lang_adapter = ModalityAdapter(input_dim, proj_dim)
        # s_t and s_{t+k} share weights: they are the same camera at different times
        self.img_adapter  = ModalityAdapter(input_dim, proj_dim)
        self.log_temp     = nn.Parameter(torch.tensor(2.6592))

    @property
    def temperature(self) -> torch.Tensor:
        # Return as tensor so gradients flow back into log_temp
        return self.log_temp.exp().clamp(0.01, 100.0)

    def forward(
        self,
        lang_emb: torch.Tensor,
        st_emb:   torch.Tensor,
        stk_emb:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_lang = self.lang_adapter(lang_emb)
        z_st   = self.img_adapter(st_emb)
        z_stk  = self.img_adapter(stk_emb)
        return z_lang, z_st, z_stk


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_ddp(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def gather_embeddings(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-gather embeddings across ranks."""
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
    dataset = TemporalDataset(
        embedding_dir=args.embedding_dir,
        k=args.k,
        camera=args.camera,
        shuffle_shards=True,
        shuffle_buffer=args.shuffle_buffer,
        rank=rank,
        world_size=world_size,
        shard_start=args.shard_start,
        shard_end=args.shard_end if args.shard_end >= 0 else None,
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
    model = TemporalAligner(input_dim=args.embed_dim, proj_dim=args.proj_dim).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[rank])

    # ---- Optimiser & scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(1, args.warmup_steps)
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
        wandb.init(project="droid-temporal-align", name=args.save_name, config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Training loop ----
    model.train()
    loss_accum      = 0.0
    lang_st_accum   = 0.0
    lang_stk_accum  = 0.0
    t_start         = time.time()

    for lang_emb, st_emb, stk_emb in loader:
        if global_step >= args.num_train_steps:
            break

        lang_emb = lang_emb.to(device)
        st_emb   = st_emb.to(device)
        stk_emb  = stk_emb.to(device)

        # Forward
        raw_model = model.module if is_ddp else model
        z_lang, z_st, z_stk = model(lang_emb, st_emb, stk_emb)

        # Cross-rank negatives
        if is_ddp and world_size > 1:
            z_lang_all = gather_embeddings(z_lang, world_size)
            z_st_all   = gather_embeddings(z_st,   world_size)
            z_stk_all  = gather_embeddings(z_stk,  world_size)
        else:
            z_lang_all, z_st_all, z_stk_all = z_lang, z_st, z_stk

        temp = raw_model.temperature

        # Primary: language-anchored triangle loss (lang, s_t, s_{t+k})
        # This aligns the instruction with both the current and subgoal states.
        loss_main = triangle_loss(
            z_lang_all, z_st_all, z_stk_all,
            temperature=temp,
            label_smoothing=args.label_smoothing,
        )

        # Optional auxiliary: temporal ordering loss (s_t, lang, s_{t+k})
        # Helps align s_t and s_{t+k} with each other via language.
        if args.aux_loss_weight > 0.0:
            loss_aux = triangle_loss(
                z_st_all, z_lang_all, z_stk_all,
                temperature=temp,
                label_smoothing=args.label_smoothing,
            )
            loss = loss_main + args.aux_loss_weight * loss_aux
        else:
            loss = loss_main

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_accum += float(loss.item())

        # Track pairwise cosine similarity (diagnostics)
        with torch.no_grad():
            lang_st_sim  = (z_lang * z_st).sum(dim=-1).mean().item()
            lang_stk_sim = (z_lang * z_stk).sum(dim=-1).mean().item()
        lang_st_accum  += lang_st_sim
        lang_stk_accum += lang_stk_sim

        global_step += 1

        # ---- Logging ----
        if global_step % args.log_freq == 0 and rank == 0:
            n = args.log_freq
            avg_loss      = loss_accum / n
            avg_lang_st   = lang_st_accum / n
            avg_lang_stk  = lang_stk_accum / n
            loss_accum    = lang_st_accum = lang_stk_accum = 0.0
            elapsed       = time.time() - t_start
            lr_now        = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step:8d} | loss={avg_loss:.4f} | "
                f"temp={temp.item():.4f} | lr={lr_now:.2e} | "
                f"sim(lang,st)={avg_lang_st:.3f} | sim(lang,stk)={avg_lang_stk:.3f} | "
                f"elapsed={elapsed/60:.1f}m"
            )
            if args.use_wandb:
                import wandb
                wandb.log(
                    {
                        "train/loss":        avg_loss,
                        "train/temp":        temp.item(),
                        "train/lr":          lr_now,
                        "train/sim_lang_st": avg_lang_st,
                        "train/sim_lang_stk":avg_lang_stk,
                    },
                    step=global_step,
                )

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
    parser = argparse.ArgumentParser(description="Temporal alignment with Triangle loss")

    # Data
    parser.add_argument("--embedding_dir",   required=True, help="Precomputed embedding TFRecord dir")
    parser.add_argument("--embed_dim",        type=int, default=1024, help="SigLIP2 embedding dim")
    parser.add_argument("--k",                type=int, default=8,    help="Temporal offset steps")
    parser.add_argument("--camera",           default="ext1",
                        choices=["ext1", "ext2", "wrist"],
                        help="Which camera embedding to use for s_t and s_{t+k}")
    parser.add_argument("--shuffle_buffer",   type=int, default=8192)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--shard_start",      type=int, default=0,    help="First shard index (inclusive)")
    parser.add_argument("--shard_end",        type=int, default=-1,   help="Last shard index (exclusive); -1 = all")

    # Model
    parser.add_argument("--proj_dim",         type=int, default=512)

    # Training
    parser.add_argument("--num_train_steps",  type=int,   default=200_000)
    parser.add_argument("--batch_size",       type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--warmup_steps",     type=int,   default=1000)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--label_smoothing",  type=float, default=0.1)
    parser.add_argument("--aux_loss_weight",  type=float, default=0.5,
                        help="Weight for the auxiliary (st-anchored) triangle loss; 0 to disable")

    # Logging
    parser.add_argument("--log_freq",         type=int, default=50)
    parser.add_argument("--use_wandb",        action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint_dir",   default="temporal_ckpts")
    parser.add_argument("--save_name",        default="droid_temporal_align")
    parser.add_argument("--save_interval",    type=int, default=5000)
    parser.add_argument("--max_checkpoints",  type=int, default=10)
    parser.add_argument("--resume",           default=None)

    # DDP
    parser.add_argument("--world_size",       type=int, default=1)
    parser.add_argument("--port",             type=int, default=12357)

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if world_size > 1:
        train(local_rank, world_size, args)
    else:
        train(0, 1, args)


if __name__ == "__main__":
    main()
