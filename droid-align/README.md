# DROID Contrastive Alignment Verifiers

Two contrastive verifiers for DROID robot trajectories using the
**Triangle loss** (NeurIPS 2025):

| Verifier | Aligns | Loss |
|----------|--------|------|
| Multi-view | ext1 ↔ ext2 ↔ wrist (same timestep) | Symmetric Triangle |
| Temporal | language ↔ s_t ↔ s_{t+k} (k=8) | Language-anchored Triangle |

## Prerequisites

```bash
conda activate CLIP-DROID
```

The environment must have:
- `open_clip_torch` (for SigLIP2 backbone)
- `tensorflow`, `tensorflow_datasets`, `dlimp` (for RLDS data loading)
- `torch` ≥ 2.0 with CUDA
- `tqdm`, `numpy`, `Pillow`

---

## Directory Structure

```
/root/vla-clip/droid-align/
├── precompute_embeddings.py   # Step 1 – encode all images & text with SigLIP2
├── verify_embeddings.py       # Step 1 verification
├── triangle_loss.py           # Shared triangle loss (area_computation, etc.)
├── embedding_dataset.py       # PyTorch IterableDatasets for both verifiers
├── train_multiview_align.py   # Step 2 – multi-view camera alignment
├── train_temporal_align.py    # Step 3 – temporal / subgoal alignment
├── precompute.sh              # Launch script for Step 1
├── verify.sh                  # Launch script for Step 1 verification
├── train_multiview.sh         # Launch script for Step 2
├── train_temporal.sh          # Launch script for Step 3
└── README.md
```

Key data paths:
```
/root/data/                         # DROID RLDS dataset root (droid/ subfolder)
/root/data/droid_embeddings/        # Precomputed SigLIP2 embedding TFRecords
/root/vla-clip/droid-align/multiview_ckpts/   # Multi-view checkpoints
/root/vla-clip/droid-align/temporal_ckpts/    # Temporal checkpoints
```

---

## Step 1 – Precompute SigLIP2 Embeddings

Encodes all DROID images (ext1, ext2, wrist) and language instructions with
`ViT-L/16-SigLIP2-384` and stores the results as sharded TFRecords.

### Output TFRecord schema (one proto = one trajectory)

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `traj_id` | bytes | — | Unique episode file path |
| `num_steps` | int64 | — | T (number of steps) |
| `embed_dim` | int64 | — | D = 1024 |
| `cameras_avail` | int64 | — | Bitmask: bit0=ext1, bit1=ext2, bit2=wrist |
| `ext1_emb_bytes` | bytes | [T×D] float16 | Exterior camera 1 embeddings |
| `ext2_emb_bytes` | bytes | [T×D] float16 | Exterior camera 2 embeddings |
| `wrist_emb_bytes` | bytes | [T×D] float16 | Wrist camera embeddings |
| `lang_emb_bytes` | bytes | [D] float16 | Language instruction embedding |

### Run

```bash
bash precompute.sh
```

Edit `RLDS_DATA_DIR` and `OUTPUT_DIR` inside `precompute.sh` if needed:
```bash
RLDS_DATA_DIR="/root/data"                  # parent dir of droid/ TFDS folder
OUTPUT_DIR="/root/data/droid_embeddings"    # where to write TFRecords
NUM_SHARDS=256                              # number of output shards
IMG_BATCH_SIZE=128                          # reduce if GPU OOM
```

Or run directly:
```bash
conda activate CLIP-DROID
python precompute_embeddings.py \
    --rlds_data_dir /root/data \
    --output_dir    /root/data/droid_embeddings \
    --num_shards    256 \
    --img_batch_size 128 \
    --device        cuda
```

**Resume** from a partially completed run:
```bash
python precompute_embeddings.py ... --start_shard 50
```

### Verify

```bash
bash verify.sh
```

Or:
```bash
python verify_embeddings.py \
    --embedding_dir /root/data/droid_embeddings \
    --num_samples   200 \
    --k             8
```

Expected output:
```
[OK] All shapes correct
[OK] All dtypes float16
[OK] ext1_embs: norm mean≈1.000, max_dev<0.05
[OK] Temporal pairs: N valid (t, t+8) pairs
[PASS] All checks passed!
```

---

## Step 2 – Multi-view Camera Alignment

**Goal**: align the three camera views at the same timestep into a shared
embedding space using the symmetric Triangle loss.

**Architecture**: Three `ViewAdapter` MLP heads (one per camera), plus
learnable temperature. Each head: `Linear(1024→1024) → GELU → LayerNorm →
Linear(1024→512)` followed by L2-normalisation.

**Loss**: Symmetric Triangle loss averaged over all 3 anchor permutations
→ no camera is privileged.

### Run

```bash
bash train_multiview.sh
```

Key settings in `train_multiview.sh`:
```bash
EMBEDDING_DIR="/root/data/droid_embeddings"
CHECKPOINT_DIR="/root/vla-clip/droid-align/multiview_ckpts"
NUM_GPUS=4
BATCH_SIZE=512          # per-rank; effective batch = 4 × 512 = 2048
PROJ_DIM=512
LR=3e-4
NUM_TRAIN_STEPS=200000
```

Or run directly:
```bash
conda activate CLIP-DROID
torchrun --standalone --nproc_per_node 4 train_multiview_align.py \
    --embedding_dir  /root/data/droid_embeddings \
    --checkpoint_dir /root/vla-clip/droid-align/multiview_ckpts \
    --batch_size     512 \
    --num_train_steps 200000 \
    --use_wandb
```

**Resume**:
```bash
# In train_multiview.sh, set:
RESUME="/root/vla-clip/droid-align/multiview_ckpts/droid_multiview_align_step_50000.pt"
```

### Checkpoints

Saved to `/root/vla-clip/droid-align/multiview_ckpts/` as:
```
droid_multiview_align_step_{N}.pt
```

Each checkpoint contains `model_state_dict`, `optimizer_state_dict`,
`scheduler_state_dict`, `global_step`, and `args`.

---

## Step 3 – Temporal / Subgoal Alignment

**Goal**: align language instructions, current-frame embeddings (s_t), and
predicted subgoal embeddings (s_{t+k}, k=8) using a language-anchored
Triangle loss.

**Architecture**: `LangAdapter` (1024→512) + `ImgAdapter` (1024→512, shared
for both s_t and s_{t+k}), plus learnable temperature.

**Loss**:
- Primary: `triangle_loss(lang, s_t, s_{t+k})` — language as anchor.
- Auxiliary (optional, weight=0.5): `triangle_loss(s_t, lang, s_{t+k})` —
  further aligns the temporal pair through the language embedding.

### Run

```bash
bash train_temporal.sh
```

Key settings in `train_temporal.sh`:
```bash
EMBEDDING_DIR="/root/data/droid_embeddings"
CHECKPOINT_DIR="/root/vla-clip/droid-align/temporal_ckpts"
NUM_GPUS=4
K=8                     # temporal offset (steps between s_t and s_{t+k})
CAMERA="ext1"           # which camera: ext1 | ext2 | wrist
BATCH_SIZE=512
AUX_LOSS_WEIGHT=0.5
NUM_TRAIN_STEPS=200000
```

Or run directly:
```bash
conda activate CLIP-DROID
torchrun --standalone --nproc_per_node 4 train_temporal_align.py \
    --embedding_dir  /root/data/droid_embeddings \
    --checkpoint_dir /root/vla-clip/droid-align/temporal_ckpts \
    --k              8 \
    --camera         ext1 \
    --batch_size     512 \
    --num_train_steps 200000 \
    --aux_loss_weight 0.5 \
    --use_wandb
```

---

## Triangle Loss Details

From Cicchetti et al. (NeurIPS 2025). The core idea is to measure the area
of the triangle formed by three embedding vectors using the Lagrange identity:

```
area(a, b, c) = (||a-b||² · ||a-c||² − ((a-b)·(a-c))²) / 2
```

For a batch of N triplets, an N×N area matrix is computed:
```
area[i, j] = area(anchor_i, view1_j, view2_j)
```

The diagonal (matched triplets) should form degenerate/small triangles
(≈ 0 area when all three embeddings are aligned), while off-diagonal
(mismatched) triplets should form large triangles.

InfoNCE-style loss:
```python
loss = (cross_entropy(-area/T,   targets)
      + cross_entropy(-area.T/T, targets)) / 2
```

where `T` is the learnable temperature (initialised to 1/0.07).

---

## Expected Training Metrics

**Multi-view alignment** (convergence after ~50k steps):
- Loss should drop from ~4.0 to ~1.5
- Temperature should stabilise around 0.03–0.07

**Temporal alignment** (convergence after ~50k steps):
- Loss should drop from ~4.0 to ~2.5
- `sim(lang, st)` and `sim(lang, stk)` should increase from ~0 to ~0.3–0.5

---

## File Relationships

```
precompute_embeddings.py
        │
        ▼  (writes TFRecords)
/root/data/droid_embeddings/*.tfrecord
        │
        ├──▶ embedding_dataset.py (MultiViewDataset)
        │           │
        │           ▼
        │   train_multiview_align.py  ──uses──▶  triangle_loss.py
        │           │
        │           ▼
        │   multiview_ckpts/
        │
        └──▶ embedding_dataset.py (TemporalDataset)
                    │
                    ▼
            train_temporal_align.py  ──uses──▶  triangle_loss.py
                    │
                    ▼
            temporal_ckpts/
```
