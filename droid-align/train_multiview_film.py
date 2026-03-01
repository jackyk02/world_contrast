#!/usr/bin/env python3
"""
Multi-view Camera Alignment with FiLM Conditioning (DROID)
==========================================================
Same as multi-view alignment but conditions each precomputed embedding on
the trajectory language embedding via FiLM before the view adapters.

Pipeline for each (visual v, language t):
  1. L2 normalize v and t
  2. Apply FiLM: gamma, beta = FiLM(t_norm); v_film = gamma * v_norm + beta
  3. Re-normalize v_film
  4. ViewAdapter(v_film) → aligned embedding

Uses MultiViewWithLangDataset (ext1, ext2, wrist, lang) and the same
symmetric Triangle loss on the three adapter outputs.

DDP is used for multi-GPU training.
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

from triangle_loss import symmetric_triangle_loss
from embedding_dataset import MultiViewWithLangDataset

torch.set_num_threads(8)


# ---------------------------------------------------------------------------
# FiLM: scale and shift from language
# ---------------------------------------------------------------------------
class FiLMLayer(nn.Module):
    """
    From L2-normalised language embedding t [B, D], produce scale gamma and
    shift beta (each [B, D]) and apply: out = L2(gamma * v + beta).
    v and t are assumed to be L2-normalised by the caller before apply().
    """

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim * 2),
        )
        # Init so initially gamma ≈ 1, beta ≈ 0 (identity)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].weight)
        self.mlp[-1].bias.data[:dim] = 1.0  # gamma init to 1

    def forward(self, v_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        """
        v_norm: [B, D] L2-normalised visual
        t_norm: [B, D] L2-normalised language
        Returns: [B, D] L2-normalised FiLM-conditioned visual
        """
        gb = self.mlp(t_norm)  # [B, 2*D]
        gamma, beta = gb.chunk(2, dim=-1)
        v_film = gamma * v_norm + beta
        return F.normalize(v_film, dim=-1)


# ---------------------------------------------------------------------------
# View adapter (same as train_multiview_align)
# ---------------------------------------------------------------------------
class ViewAdapter(nn.Module):
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


# ---------------------------------------------------------------------------
# Multi-view aligner with FiLM
# ---------------------------------------------------------------------------
class MultiViewFiLMAligner(nn.Module):
    """
    For each view: L2(v) and L2(t) → FiLM(v_norm, t_norm) → L2(v_film) → ViewAdapter → z.
    Shared FiLM and shared language t for all three views; per-view adapters.
    """

    def __init__(self, input_dim: int = 1024, proj_dim: int = 512):
        super().__init__()
        self.film = FiLMLayer(input_dim)
        self.ext1_adapter  = ViewAdapter(input_dim, proj_dim)
        self.ext2_adapter  = ViewAdapter(input_dim, proj_dim)
        self.wrist_adapter = ViewAdapter(input_dim, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(2.6592))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 100.0)

    def _condition(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v_norm = F.normalize(v, dim=-1)
        t_norm = F.normalize(t, dim=-1)
        return self.film(v_norm, t_norm)

    def forward(
        self,
        ext1_emb:  torch.Tensor,
        ext2_emb:  torch.Tensor,
        wrist_emb: torch.Tensor,
        lang_emb:  torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) L2 norm v and t, 2) FiLM, 3) re-normalize (inside FiLM)
        ext1_film  = self._condition(ext1_emb,  lang_emb)
        ext2_film  = self._condition(ext2_emb,  lang_emb)
        wrist_film = self._condition(wrist_emb, lang_emb)
        z1 = self.ext1_adapter(ext1_film)
        z2 = self.ext2_adapter(ext2_film)
        zw = self.wrist_adapter(wrist_film)
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
# Validation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _retrieval_acc(
    anchor1: torch.Tensor,
    anchor2: torch.Tensor,
    candidates: torch.Tensor,
) -> float:
    N = anchor1.shape[0]
    u = anchor1 - anchor2
    u_norm_sq = (u * u).sum(dim=1, keepdim=True)
    v = anchor1.unsqueeze(1) - candidates.unsqueeze(0)
    v_norm_sq = (v * v).sum(dim=2)
    uv_dot    = (u.unsqueeze(1) * v).sum(dim=2)
    area = u_norm_sq * v_norm_sq - uv_dot ** 2
    preds  = area.argmin(dim=1)
    labels = torch.arange(N, device=anchor1.device)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def compute_val_metrics(
    model: nn.Module,
    val_loader_iter,
    val_batches: int,
    device: torch.device,
    label_smoothing: float,
) -> dict:
    raw_model = model.module if hasattr(model, "module") else model
    model.eval()

    val_loss_sum  = 0.0
    acc_01to2_sum = 0.0
    acc_02to1_sum = 0.0
    acc_12to0_sum = 0.0
    n_batches     = 0

    for _ in range(val_batches):
        try:
            ext1_emb, ext2_emb, wrist_emb, lang_emb = next(val_loader_iter)
        except StopIteration:
            break

        ext1_emb  = ext1_emb.to(device)
        ext2_emb  = ext2_emb.to(device)
        wrist_emb = wrist_emb.to(device)
        lang_emb  = lang_emb.to(device)

        z1, z2, zw = model(ext1_emb, ext2_emb, wrist_emb, lang_emb)
        temp = raw_model.temperature

        val_loss_sum += float(symmetric_triangle_loss(
            z1, z2, zw,
            temperature=temp,
            label_smoothing=label_smoothing,
        ).item())

        acc_01to2_sum += _retrieval_acc(z1, z2, zw)
        acc_02to1_sum += _retrieval_acc(z1, zw, z2)
        acc_12to0_sum += _retrieval_acc(z2, zw, z1)
        n_batches += 1

    model.train()

    if n_batches == 0:
        return {}

    return {
        "val/loss":       val_loss_sum  / n_batches,
        "val/acc_01to2":  acc_01to2_sum / n_batches,
        "val/acc_02to1":  acc_02to1_sum / n_batches,
        "val/acc_12to0":  acc_12to0_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(rank: int, world_size: int, args: argparse.Namespace):
    is_ddp = world_size > 1
    if is_ddp:
        setup_ddp(rank, world_size, args.port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ---- Dataset (with language) ----
    dataset = MultiViewWithLangDataset(
        embedding_dir=args.embedding_dir,
        shuffle_shards=True,
        shuffle_buffer=args.shuffle_buffer,
        require_all_cameras=True,
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

    # ---- Validation ----
    val_loader_iter = None
    val_loader      = None
    if args.val_interval > 0:
        if args.val_shard_start >= 0 and args.val_shard_end >= 0:
            val_start, val_end = args.val_shard_start, args.val_shard_end
        elif args.shard_start > 0:
            val_start, val_end = 0, args.shard_start
        else:
            val_start = val_end = 0
        if val_end > val_start:
            val_dataset = MultiViewWithLangDataset(
                embedding_dir=args.embedding_dir,
                shuffle_shards=True,
                shuffle_buffer=min(args.shuffle_buffer, 4096),
                require_all_cameras=True,
                rank=rank,
                world_size=world_size,
                shard_start=val_start,
                shard_end=val_end,
            )
            val_num_workers = 0 if (val_end - val_start) == 1 else max(1, args.num_workers // 2)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=val_num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2 if val_num_workers > 0 else None,
            )
            val_loader_iter = iter(val_loader)
            if rank == 0:
                print(f"Validation dataset: shards [{val_start}, {val_end}), "
                      f"val_interval={args.val_interval}, val_batches={args.val_batches}")

    # ---- Model ----
    model = MultiViewFiLMAligner(input_dim=args.embed_dim, proj_dim=args.proj_dim).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.num_train_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    if args.use_wandb and rank == 0:
        import wandb
        wandb.init(project="droid-multiview-align", name=args.save_name, config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model.train()
    loss_accum = 0.0
    t_start = time.time()

    for ext1_emb, ext2_emb, wrist_emb, lang_emb in loader:
        if global_step >= args.num_train_steps:
            break

        ext1_emb  = ext1_emb.to(device)
        ext2_emb  = ext2_emb.to(device)
        wrist_emb = wrist_emb.to(device)
        lang_emb  = lang_emb.to(device)

        raw_model = model.module if is_ddp else model
        z1, z2, zw = model(ext1_emb, ext2_emb, wrist_emb, lang_emb)

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

        if global_step % args.log_freq == 0 and rank == 0:
            avg_loss = loss_accum / args.log_freq
            loss_accum = 0.0
            elapsed = time.time() - t_start
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step:8d} | loss={avg_loss:.4f} | "
                f"temp={temp.item():.4f} | lr={lr_now:.2e} | "
                f"elapsed={elapsed/60:.1f}m"
            )
            if args.use_wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/temp": temp.item(),
                           "train/lr": lr_now}, step=global_step)

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

        if (
            args.val_interval > 0
            and global_step % args.val_interval == 0
            and rank == 0
            and val_loader_iter is not None
        ):
            val_metrics = compute_val_metrics(
                model, val_loader_iter, args.val_batches,
                device, args.label_smoothing,
            )
            if not val_metrics:
                val_loader_iter = iter(val_loader)
                val_metrics = compute_val_metrics(
                    model, val_loader_iter, args.val_batches,
                    device, args.label_smoothing,
                )
            if val_metrics:
                print(
                    f"[VAL] step={global_step:8d} | "
                    f"loss={val_metrics['val/loss']:.4f} | "
                    f"acc (cam0+cam1)→cam2={val_metrics['val/acc_01to2']*100:.1f}% | "
                    f"acc (cam0+cam2)→cam1={val_metrics['val/acc_02to1']*100:.1f}% | "
                    f"acc (cam1+cam2)→cam0={val_metrics['val/acc_12to0']*100:.1f}%"
                )
                if args.use_wandb:
                    import wandb
                    wandb.log(val_metrics, step=global_step)

        if global_step % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if is_ddp:
        dist.destroy_process_group()

    if rank == 0:
        elapsed = time.time() - t_start
        print(f"Training complete. {global_step} steps in {elapsed/60:.1f} min.")


def main():
    parser = argparse.ArgumentParser(description="Multi-view + FiLM alignment with Triangle loss")

    parser.add_argument("--embedding_dir",  required=True)
    parser.add_argument("--embed_dim",       type=int, default=1024)
    parser.add_argument("--shuffle_buffer",  type=int, default=8192)
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--shard_start",     type=int, default=0)
    parser.add_argument("--shard_end",       type=int, default=-1)

    parser.add_argument("--proj_dim",        type=int, default=512)

    parser.add_argument("--num_train_steps", type=int,   default=200_000)
    parser.add_argument("--batch_size",      type=int,   default=512)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--warmup_steps",    type=int,   default=1000)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--log_freq",        type=int, default=50)
    parser.add_argument("--use_wandb",       action="store_true")

    parser.add_argument("--val_interval",    type=int, default=1000)
    parser.add_argument("--val_batches",     type=int, default=50)
    parser.add_argument("--val_shard_start", type=int, default=-1)
    parser.add_argument("--val_shard_end",   type=int, default=-1)

    parser.add_argument("--checkpoint_dir",  default="multiview_film_ckpts")
    parser.add_argument("--save_name",       default="droid_multiview_film")
    parser.add_argument("--save_interval",   type=int, default=5000)
    parser.add_argument("--max_checkpoints", type=int, default=10)
    parser.add_argument("--resume",          default=None)

    parser.add_argument("--world_size",      type=int, default=1)
    parser.add_argument("--port",            type=int, default=12356)

    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    if world_size > 1:
        train(local_rank, world_size, args)
    else:
        train(0, 1, args)


if __name__ == "__main__":
    main()
