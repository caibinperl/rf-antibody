"""CDR-contact hotspot labelling (SI section 3.5 / RFantibody README)."""

from __future__ import annotations

import torch

from rfab_finetune.geometry import get_cb


def cdr_hotspots(
    xyz: torch.Tensor,
    target_mask: torch.Tensor,
    cdr_mask: torch.Tensor,
    cutoff: float = 8.0,
    k: int = 5,
) -> torch.Tensor:
    """Target residue is a hotspot if mean CB distance to the closest `k` CDR residues < cutoff.

    Args:
        xyz: [L, 3+, 3]
        target_mask: [L] bool
        cdr_mask: [L] bool
    Returns:
        [L] bool, True only on target residues that pass the cutoff.
    """
    length = xyz.shape[0]
    out = torch.zeros(length, dtype=torch.bool, device=xyz.device)
    if not bool(target_mask.any()) or not bool(cdr_mask.any()):
        return out

    cb = get_cb(xyz[None])[0]
    tgt_idx = target_mask.nonzero(as_tuple=False).flatten()
    cdr_idx = cdr_mask.nonzero(as_tuple=False).flatten()
    dist = torch.cdist(cb[tgt_idx], cb[cdr_idx])  # [n_tgt, n_cdr]
    kk = min(k, dist.shape[1])
    knn = dist.topk(kk, dim=1, largest=False).values.mean(dim=1)
    out[tgt_idx] = knn < cutoff
    return out


def sample_hotspot_labels(hotspots: torch.Tensor) -> torch.Tensor:
    """Provide a random fraction in [0, 1] of true hotspots (SI section 3.5)."""
    labels = torch.zeros_like(hotspots, dtype=torch.float32)
    idx = hotspots.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        return labels
    frac = float(torch.rand(1).item())
    n_keep = int(round(frac * idx.numel()))
    if n_keep <= 0:
        return labels
    perm = idx[torch.randperm(idx.numel(), device=idx.device)[:n_keep]]
    labels[perm] = 1.0
    return labels
