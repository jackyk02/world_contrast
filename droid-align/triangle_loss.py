"""
Triangle loss implementation for three-way contrastive alignment.

Reference: "A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity"
           (Cicchetti et al., NeurIPS 2025)

The loss minimises the area of the triangle formed by matched (anchor_i, v1_i, v2_i)
triplets while pushing mismatched triplets to form larger triangles.

This file provides:
  area_computation         – raw [N, N] triangle-area matrix
  triangle_loss            – InfoNCE-style loss for one anchor direction
  symmetric_triangle_loss  – averaged over all three anchor permutations
"""

import torch
import torch.nn.functional as F


def area_computation(
    anchor: torch.Tensor,
    view1: torch.Tensor,
    view2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute triangle areas for all (anchor_i, view1_j, view2_j) combinations.

    All inputs must be L2-normalised and have the same shape [N, D].

    The area of the triangle with vertices (a, b, c) using the cross product of
    edge vectors (a-b) and (a-c) equals:
        area = || (a-b) × (a-c) ||^2 / 2
             = (||u||^2 * ||v||^2 - (u·v)^2) / 2     (Lagrange identity)
    where u = a - b, v = a - c.

    We omit the outer sqrt (no gradient issue; preserves ordering).

    Args:
        anchor: [N, D]
        view1:  [N, D]
        view2:  [N, D]

    Returns:
        area: [N, N]  area[i, j] = triangle area for (anchor_i, view1_j, view2_j)
                      Diagonal entries are matched triplets (should be small).
    """
    anchor_exp = anchor.unsqueeze(1)          # [N, 1, D]
    u = anchor_exp - view1.unsqueeze(0)       # [N, N, D]  anchor_i - view1_j
    v = anchor_exp - view2.unsqueeze(0)       # [N, N, D]  anchor_i - view2_j

    u_norm_sq = torch.sum(u * u, dim=2)       # [N, N]
    v_norm_sq = torch.sum(v * v, dim=2)       # [N, N]
    uv_dot    = torch.sum(u * v, dim=2)       # [N, N]

    area = (u_norm_sq * v_norm_sq - uv_dot ** 2) / 2.0   # [N, N]
    return area


def triangle_loss(
    anchor: torch.Tensor,
    view1: torch.Tensor,
    view2: torch.Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE triangle loss for one anchor direction.

    Uses the [N, N] area matrix as *negative* logits so that minimising
    the cross-entropy is equivalent to minimising the triangle area of
    correct triplets (diagonal) relative to incorrect ones (off-diagonal).

    Two loss terms:
      L1 = CE(-area,   targets)   anchor→(view1,view2) direction
      L2 = CE(-area.T, targets)   (view1,view2)→anchor direction

    Args:
        anchor, view1, view2: [N, D] L2-normalised embeddings
        temperature:          scales the logits (lower = sharper distribution)
        label_smoothing:      label smoothing for cross-entropy

    Returns:
        scalar loss
    """
    N = anchor.shape[0]
    targets = torch.arange(N, dtype=torch.long, device=anchor.device)

    area  = area_computation(anchor, view1, view2) / temperature   # [N, N]
    areaT = area.T

    loss = (
        F.cross_entropy(-area,  targets, label_smoothing=label_smoothing)
        + F.cross_entropy(-areaT, targets, label_smoothing=label_smoothing)
    ) / 2.0
    return loss


def symmetric_triangle_loss(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    emb3: torch.Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Fully symmetric triangle loss averaged over all three anchor permutations.

    Suitable when all three modalities are equally important (e.g., multi-view
    camera alignment where no single view acts as a privileged anchor).

    Each of the three permutations:
      (emb1, emb2, emb3),  (emb2, emb1, emb3),  (emb3, emb1, emb2)
    contributes one directional triangle_loss term.

    Args:
        emb1, emb2, emb3: [N, D] L2-normalised embeddings
        temperature, label_smoothing: passed to triangle_loss

    Returns:
        scalar loss
    """
    loss = (
        triangle_loss(emb1, emb2, emb3, temperature, label_smoothing)
        + triangle_loss(emb2, emb1, emb3, temperature, label_smoothing)
        + triangle_loss(emb3, emb1, emb2, temperature, label_smoothing)
    ) / 3.0
    return loss
