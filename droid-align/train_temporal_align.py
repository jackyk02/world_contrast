#!/usr/bin/env python3
"""
Temporal / Subgoal Alignment Training (DROID)  –  (s_t, a_{t:t+k-1}, s_{t+k})
============================================================================
Trains lightweight MLP adapter heads on top of frozen precomputed SigLIP2
embeddings using the Triangle loss to align:

  - s_t          (current frame image embedding)
  - a_{t:t+k-1}   (action chunk: k normalised delta actions)
  - s_{t+k}      (subgoal frame, k steps ahead; default k=8)

The Triangle loss with s_t as the primary anchor learns:
  1. Which action sequence leads from s_t to a given s_{t+k}?
  2. Which future state s_{t+k} results from executing a_{t:t+k-1} at s_t?
  3. Implicitly, the relationship between action sequences and state transitions.

An optional auxiliary loss uses the action chunk as anchor:
  (a_{t:t+k-1},  s_t,  s_{t+k})  – which (s_t, s_{t+k}) pair does this action connect?

Architecture:
  ImgAdapter:      1024 → proj_dim  (shared weight for both s_t and s_{t+k})
  ActionEncoder:   per-step Linear(action_dim → proj_dim)
                   + sinusoidal positional embedding
                   + TransformerEncoder(d_model=proj_dim, nhead=8, layers=4,
                                        ffn=proj_dim×2, dropout=0.1)
                   → masked mean-pool → proj_dim
                   (matches finetune_droid_two_view_ddp.py trajectory_encoder)
  Learnable temperature.

Requires: delta_actions_bytes field in TFRecords (append_delta_actions.py).

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
from embedding_dataset import TemporalActionDataset, ACTION_DIM

torch.set_num_threads(8)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sincos_position_embedding(seq_len: int, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional embedding, shape [seq_len, dim].
    Copied from bridge_verifier/model.py (used identically in
    finetune_droid_two_view_ddp.py).
    """
    pos      = torch.arange(seq_len).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid = torch.einsum("i,j->ij", pos, inv_freq)
    return torch.cat((sinusoid.sin(), sinusoid.cos()), dim=-1)  # [seq_len, dim]


class ModalityAdapter(nn.Module):
    """
    Two-layer MLP projection head.

    Input:  [B, input_dim]   (float32)
    Output: [B, output_dim]  (float32, L2-normalised)
    """

    def __init__(self, input_dim: int, output_dim: int):
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


class ActionAlignmentModel(nn.Module):
    """
    Aligns (s_t, a_{t:t+k-1}, s_{t+k}) in a shared embedding space.

      ImgAdapter     – shared ModalityAdapter for s_t and s_{t+k}
      ActionEncoder  – Transformer encoder matching finetune_droid_two_view_ddp.py:
                         single_step_encoder : Linear(action_dim → proj_dim)
                         action_pos_embedding: sincos [1, k, proj_dim] (buffer)
                         trajectory_encoder  : TransformerEncoder
                                               (d_model=proj_dim, nhead=nhead,
                                                ffn=proj_dim×2, dropout=0.1,
                                                num_layers=num_encoder_layers)
                         output              : masked mean-pool → L2-normalise
      log_temp       – learnable temperature
    """

    def __init__(
        self,
        input_dim: int = 1024,
        proj_dim: int = 512,
        k: int = 8,
        action_dim: int = ACTION_DIM,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k          = k
        self.action_dim = action_dim

        # Image adapter (shared for s_t and s_{t+k})
        self.img_adapter = ModalityAdapter(input_dim, proj_dim)

        # Action Transformer encoder — identical to finetune_droid_two_view_ddp.py
        self.single_step_action_encoder = nn.Linear(action_dim, proj_dim)
        self.register_buffer(
            "action_pos_embedding",
            sincos_position_embedding(k, proj_dim).unsqueeze(0),  # [1, k, proj_dim]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=proj_dim * 2,
            dropout=dropout,
            batch_first=False,  # expects [seq, batch, dim] — matches finetune
        )
        self.trajectory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.log_temp = nn.Parameter(torch.tensor(2.6592))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(0.01, 100.0)

    def encode_actions(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        Encode a [B, k, A] action chunk into a [B, proj_dim] L2-normalised vector.

        Pipeline (matches finetune_droid_two_view_ddp.py forward_features):
          1. Per-step linear projection:  [B, k, A] → [B, k, proj_dim]
          2. Add sincos positional embedding
          3. Permute to [k, B, proj_dim] for TransformerEncoder
          4. TransformerEncoder → [k, B, proj_dim]
          5. Permute back → [B, k, proj_dim]
          6. Mean-pool over the k steps → [B, proj_dim]
          7. L2-normalise
        """
        # 1-2: per-step encode + positional embedding
        x = self.single_step_action_encoder(action_chunk)   # [B, k, proj_dim]
        x = x + self.action_pos_embedding                   # [B, k, proj_dim]

        # 3-5: Transformer (seq-first)
        x = x.permute(1, 0, 2)                             # [k, B, proj_dim]
        x = self.trajectory_encoder(x)                     # [k, B, proj_dim]
        x = x.permute(1, 0, 2)                             # [B, k, proj_dim]

        # 6-7: mean-pool + normalise
        z = x.mean(dim=1)                                   # [B, proj_dim]
        return F.normalize(z, dim=-1)

    def forward(
        self,
        st_emb:       torch.Tensor,  # [B, D]
        action_chunk: torch.Tensor,  # [B, k, A]
        stk_emb:      torch.Tensor,  # [B, D]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_st      = self.img_adapter(st_emb)
        z_actions = self.encode_actions(action_chunk)
        z_stk     = self.img_adapter(stk_emb)
        return z_st, z_actions, z_stk


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
# Validation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _retrieval_acc(
    anchor1: torch.Tensor,
    anchor2: torch.Tensor,
    candidates: torch.Tensor,
) -> float:
    """
    Given pairs (anchor1_i, anchor2_i), retrieve the correct candidates_i
    using the triangle area as a similarity score (smaller area = better match).
    """
    N = anchor1.shape[0]
    u = anchor1 - anchor2
    u_norm_sq = (u * u).sum(dim=1, keepdim=True)
    v = anchor1.unsqueeze(1) - candidates.unsqueeze(0)
    v_norm_sq = (v * v).sum(dim=2)
    uv_dot = (u.unsqueeze(1) * v).sum(dim=2)
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
    aux_loss_weight: float,
) -> dict:
    """
    Run validation for up to *val_batches* mini-batches and return:
      val/loss                 – triangle loss
      val/acc_st_to_stk        – (s_t + actions) → s_{t+k}  retrieval top-1
      val/acc_stk_to_st        – (actions + s_{t+k}) → s_t  retrieval top-1
    """
    raw_model = model.module if hasattr(model, "module") else model
    model.eval()

    val_loss_sum       = 0.0
    acc_st_to_stk_sum  = 0.0
    acc_stk_to_st_sum  = 0.0
    n_batches          = 0

    for _ in range(val_batches):
        try:
            st_emb, action_chunk, stk_emb = next(val_loader_iter)
        except StopIteration:
            break

        st_emb       = st_emb.to(device)
        action_chunk = action_chunk.to(device)
        stk_emb      = stk_emb.to(device)

        z_st, z_actions, z_stk = model(st_emb, action_chunk, stk_emb)
        temp = raw_model.temperature

        # Primary: s_t-anchored  (s_t, a_{t:t+k-1}, s_{t+k})
        loss_main = triangle_loss(
            z_st, z_actions, z_stk,
            temperature=temp,
            label_smoothing=label_smoothing,
        )
        if aux_loss_weight > 0.0:
            # Auxiliary: action-anchored  (a_{t:t+k-1}, s_t, s_{t+k})
            loss_aux = triangle_loss(
                z_actions, z_st, z_stk,
                temperature=temp,
                label_smoothing=label_smoothing,
            )
            val_loss_sum += float((loss_main + aux_loss_weight * loss_aux).item())
        else:
            val_loss_sum += float(loss_main.item())

        acc_st_to_stk_sum += _retrieval_acc(z_st,  z_actions, z_stk)  # (s_t,     act) → s_{t+k}
        acc_stk_to_st_sum += _retrieval_acc(z_stk, z_actions, z_st)   # (s_{t+k}, act) → s_t

        n_batches += 1

    model.train()

    if n_batches == 0:
        return {}

    return {
        "val/loss":           val_loss_sum      / n_batches,
        "val/acc_st_to_stk":  acc_st_to_stk_sum / n_batches,
        "val/acc_stk_to_st":  acc_stk_to_st_sum / n_batches,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(rank: int, world_size: int, args: argparse.Namespace):
    is_ddp = world_size > 1
    if is_ddp:
        setup_ddp(rank, world_size, args.port)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    dataset = TemporalActionDataset(
        embedding_dir=args.embedding_dir,
        k=args.k,
        camera=args.camera,
        shuffle_shards=True,
        shuffle_buffer=args.shuffle_buffer,
        rank=rank,
        world_size=world_size,
        shard_start=args.shard_start,
        shard_end=args.shard_end if args.shard_end >= 0 else None,
        successful_only=args.successful_only,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- Validation dataset ----
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
            val_dataset = TemporalActionDataset(
                embedding_dir=args.embedding_dir,
                k=args.k,
                camera=args.camera,
                shuffle_shards=True,
                shuffle_buffer=min(args.shuffle_buffer, 4096),
                rank=rank,
                world_size=world_size,
                shard_start=val_start,
                shard_end=val_end,
                successful_only=args.successful_only,
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
    model = ActionAlignmentModel(
        input_dim=args.embed_dim,
        proj_dim=args.proj_dim,
        k=args.k,
        action_dim=args.action_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
    ).to(device)

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
        wandb.init(project="droid-action-temporal-align", name=args.save_name, config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if rank == 0:
        print(
            f"ActionAlignmentModel: embed_dim={args.embed_dim}, proj_dim={args.proj_dim}, "
            f"k={args.k}, action_dim={args.action_dim} | "
            f"Transformer: nhead={args.nhead}, layers={args.num_encoder_layers}, "
            f"ffn={args.proj_dim * 2}, dropout={args.dropout}"
        )

    # ---- Training loop ----
    model.train()
    loss_accum            = 0.0
    acc_st_to_stk_accum   = 0.0  # (s_t,   act) → s_{t+k}
    acc_stk_to_st_accum   = 0.0  # (s_{t+k}, act) → s_t
    t_start               = time.time()

    for st_emb, action_chunk, stk_emb in loader:
        if global_step >= args.num_train_steps:
            break

        st_emb       = st_emb.to(device)
        action_chunk = action_chunk.to(device)
        stk_emb      = stk_emb.to(device)

        raw_model = model.module if is_ddp else model
        z_st, z_actions, z_stk = model(st_emb, action_chunk, stk_emb)

        # Cross-rank negatives
        if is_ddp and world_size > 1:
            z_st_all      = gather_embeddings(z_st,      world_size)
            z_actions_all = gather_embeddings(z_actions, world_size)
            z_stk_all     = gather_embeddings(z_stk,     world_size)
        else:
            z_st_all, z_actions_all, z_stk_all = z_st, z_actions, z_stk

        temp = raw_model.temperature

        # Primary: s_t-anchored  (s_t, a_{t:t+k-1}, s_{t+k})
        loss_main = triangle_loss(
            z_st_all, z_actions_all, z_stk_all,
            temperature=temp,
            label_smoothing=args.label_smoothing,
        )

        # Optional auxiliary: action-anchored  (a_{t:t+k-1}, s_t, s_{t+k})
        if args.aux_loss_weight > 0.0:
            loss_aux = triangle_loss(
                z_actions_all, z_st_all, z_stk_all,
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

        with torch.no_grad():
            acc_st_to_stk_accum += _retrieval_acc(z_st,  z_actions, z_stk)  # (s_t,     act) → s_{t+k}
            acc_stk_to_st_accum += _retrieval_acc(z_stk, z_actions, z_st)   # (s_{t+k}, act) → s_t

        global_step += 1

        # ---- Logging ----
        if global_step % args.log_freq == 0 and rank == 0:
            n = args.log_freq
            avg_loss        = loss_accum           / n
            avg_st_to_stk   = acc_st_to_stk_accum  / n
            avg_stk_to_st   = acc_stk_to_st_accum  / n
            loss_accum = acc_st_to_stk_accum = acc_stk_to_st_accum = 0.0
            elapsed  = time.time() - t_start
            lr_now   = optimizer.param_groups[0]["lr"]
            print(
                f"step={global_step:8d} | loss={avg_loss:.4f} | "
                f"acc(st+act→stk)={avg_st_to_stk*100:.1f}% | "
                f"acc(stk+act→st)={avg_stk_to_st*100:.1f}% | "
                f"temp={temp.item():.4f} | lr={lr_now:.2e} | "
                f"elapsed={elapsed/60:.1f}m"
            )
            if args.use_wandb:
                import wandb
                wandb.log(
                    {
                        "train/loss":            avg_loss,
                        "train/acc_st_to_stk":   avg_st_to_stk,
                        "train/acc_stk_to_st":   avg_stk_to_st,
                        "train/temp":            temp.item(),
                        "train/lr":              lr_now,
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

        # ---- Validation ----
        if (
            args.val_interval > 0
            and global_step % args.val_interval == 0
            and rank == 0
            and val_loader_iter is not None
        ):
            val_metrics = compute_val_metrics(
                model, val_loader_iter, args.val_batches,
                device, args.label_smoothing, args.aux_loss_weight,
            )
            if not val_metrics:
                val_loader_iter = iter(val_loader)
                val_metrics = compute_val_metrics(
                    model, val_loader_iter, args.val_batches,
                    device, args.label_smoothing, args.aux_loss_weight,
                )
            if val_metrics:
                print(
                    f"[VAL] step={global_step:8d} | "
                    f"loss={val_metrics['val/loss']:.4f} | "
                    f"acc(st+act→stk)={val_metrics['val/acc_st_to_stk']*100:.1f}% | "
                    f"acc(stk+act→st)={val_metrics['val/acc_stk_to_st']*100:.1f}%"
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Temporal alignment (s_t, a_{t:t+k-1}, s_{t+k}) with Triangle loss"
    )

    # Data
    parser.add_argument("--embedding_dir",   required=True,
                        help="Precomputed embedding TFRecord dir (with delta_actions_bytes)")
    parser.add_argument("--embed_dim",        type=int, default=1024,
                        help="SigLIP2 embedding dim (default: 1024)")
    parser.add_argument("--action_dim",       type=int, default=ACTION_DIM,
                        help=f"Action dimensionality per step (default: {ACTION_DIM})")
    parser.add_argument("--k",                type=int, default=8,
                        help="Temporal offset steps = action chunk length (default: 8)")
    parser.add_argument("--camera",           default="ext1",
                        choices=["ext1", "ext2", "wrist"],
                        help="Which camera embedding to use for s_t and s_{t+k}")
    parser.add_argument("--shuffle_buffer",   type=int, default=8192)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--shard_start",      type=int, default=0,
                        help="First shard index (inclusive)")
    parser.add_argument("--shard_end",        type=int, default=-1,
                        help="Last shard index (exclusive); -1 = all")
    parser.add_argument("--successful_only",  action="store_true", default=False,
                        help="Only use trajectories where is_successful=1")

    # Model
    parser.add_argument("--proj_dim",             type=int,   default=512)
    parser.add_argument("--nhead",                type=int,   default=8,
                        help="Number of attention heads in the action TransformerEncoder (default: 8)")
    parser.add_argument("--num_encoder_layers",   type=int,   default=4,
                        help="Number of TransformerEncoder layers (default: 4)")
    parser.add_argument("--dropout",              type=float, default=0.1,
                        help="Dropout in TransformerEncoderLayer (default: 0.1)")

    # Training
    parser.add_argument("--num_train_steps",  type=int,   default=200_000)
    parser.add_argument("--batch_size",       type=int,   default=512)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--warmup_steps",     type=int,   default=1000)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--label_smoothing",  type=float, default=0.1)
    parser.add_argument("--aux_loss_weight",  type=float, default=0.5,
                        help="Weight for action-anchored auxiliary triangle loss; 0 to disable")

    # Logging
    parser.add_argument("--log_freq",         type=int, default=50)
    parser.add_argument("--use_wandb",        action="store_true")

    # Validation
    parser.add_argument("--val_interval",     type=int, default=1000,
                        help="Run validation every N steps (0 = disabled)")
    parser.add_argument("--val_batches",      type=int, default=50,
                        help="Number of mini-batches per validation run")
    parser.add_argument("--val_shard_start",  type=int, default=-1,
                        help="First shard index for validation (inclusive)")
    parser.add_argument("--val_shard_end",    type=int, default=-1,
                        help="Last shard index for validation (exclusive)")

    # Checkpointing
    parser.add_argument("--checkpoint_dir",   default="temporal_ckpts")
    parser.add_argument("--save_name",        default="droid_action_temporal_align")
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
