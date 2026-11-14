"""Losses from SI Table 1: frame MSE, distogram, masked-language modelling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rfab_finetune.constants import TrainConfig
from rfab_finetune.geometry import distogram_targets, frames_from_xyz, rotation_mse, translation_mse


def mlm_time_weight(t: int, cfg: TrainConfig) -> float:
    """Sigmoid so the sequence loss is concentrated at small t (SI section 3.4)."""
    return float(torch.sigmoid(torch.tensor((cfg.mlm_sigmoid_center - t) / cfg.mlm_sigmoid_temp)))


def frame_loss(pred_xyz: torch.Tensor, true_xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    rot_p, trans_p = frames_from_xyz(pred_xyz)
    rot_t, trans_t = frames_from_xyz(true_xyz)
    return translation_mse(trans_p, trans_t, mask) + rotation_mse(rot_p, rot_t, mask)


def distogram_loss(logits: torch.Tensor, true_xyz: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
    """logits: [B, C, L, L] or [B, L, L, C]. pair: [B, L, L] pairs that are supervised."""
    if logits.shape[-1] != logits.shape[-2] and logits.shape[1] != true_xyz.shape[1]:
        # [B, C, L, L]
        logits_bcll = logits
    elif logits.shape[-1] < 64 and logits.shape[-1] != logits.shape[-2]:
        logits_bcll = logits.permute(0, 3, 1, 2)
    else:
        logits_bcll = logits
    target = distogram_targets(true_xyz)
    loss = F.cross_entropy(logits_bcll, target, reduction="none")
    weight = pair.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def mlm_loss(logits_aa: torch.Tensor, seq_true: torch.Tensor, cdr_mask: torch.Tensor) -> torch.Tensor:
    """Categorical CE on CDR (masked) positions. logits [B, C, L] or [B, L, C]."""
    if logits_aa.dim() == 3 and logits_aa.shape[-1] == seq_true.shape[-1] and logits_aa.shape[1] <= 32:
        logits = logits_aa  # [B, C, L]
    else:
        logits = logits_aa.permute(0, 2, 1)
    loss = F.cross_entropy(logits, seq_true, reduction="none")
    w = cdr_mask.to(loss.dtype)
    if w.sum() < 1:
        return loss.mean() * 0.0
    return (loss * w).sum() / w.sum()


def total_loss(
    pred: dict,
    feats: dict,
    t: int,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict]:
    finite = ~torch.isnan(feats["xyz_true"]).any(dim=-1).any(dim=-1)
    diffuse = feats["diffuse_mask"]
    if diffuse.dim() == 1:
        diffuse = diffuse.unsqueeze(0)
    diffuse = diffuse.to(device=finite.device).bool()
    finite = finite.bool()
    # The target is an un-noised input. Frame loss covers the binder only.
    binder = (finite & diffuse).float()
    l_frame = frame_loss(pred["xyz"], feats["xyz_true"], binder)
    # Distogram skips pairs already given in t2d (framework-framework and target-target).
    templ = feats["template_mask"].to(device=finite.device).bool()
    if templ.dim() == 2:
        templ = templ.unsqueeze(0)
    pair = finite[:, :, None] & finite[:, None, :] & ~templ
    eye = torch.eye(pair.shape[-1], dtype=torch.bool, device=pair.device)
    pair = pair & ~eye.unsqueeze(0)
    l_2d = distogram_loss(pred["logits_dist"], feats["xyz_true"], pair)
    l_mlm = mlm_loss(pred["logits_aa"], feats["seq_true"], feats["cdr_mask"][None])
    w_mlm = mlm_time_weight(t, cfg)
    loss = cfg.w_frame * l_frame + cfg.w_2d * l_2d + cfg.w_mlm * w_mlm * l_mlm
    stats = {
        "loss": float(loss.detach()),
        "l_frame": float(l_frame.detach()),
        "l_2d": float(l_2d.detach()),
        "l_mlm": float(l_mlm.detach()),
        "w_mlm": w_mlm,
        "t": t,
    }
    return loss, stats
