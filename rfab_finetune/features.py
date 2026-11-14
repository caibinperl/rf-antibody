"""Build RFdiffusion input features for an antibody training example (SI Table 2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rfab_finetune.constants import MASK_IDX, N_AA, TrainConfig
from rfab_finetune.geometry import residue_index, template_torsions, xyz_to_t2d
from rfab_finetune.hotspots import cdr_hotspots, sample_hotspot_labels
from rfab_finetune.pdb import ComplexExample


def _one_hot_seq(seq: torch.Tensor) -> torch.Tensor:
    return F.one_hot(seq.clamp(0, N_AA - 1), num_classes=N_AA).float()


def _template_mask(ex: ComplexExample) -> torch.Tensor:
    """Intra-framework (including H-L) and intra-target pairs. CDR rows and columns are out."""
    loop = ex.cdr_mask
    fw = ex.framework_mask
    tgt = ex.target_mask
    block = (fw[:, None] & fw[None, :]) | (tgt[:, None] & tgt[None, :])
    return block & ~loop[:, None] & ~loop[None, :]


def featurize(
    ex: ComplexExample,
    xyz_t: torch.Tensor,
    timestep_conf: torch.Tensor,
    cfg: TrainConfig,
    hotspots_override: torch.Tensor | None = None,
) -> dict:
    """Return a batch-1 feature dict matching the antibody templating scheme.

    t2d is 44-d: 37 distance bins, 6 orientation channels, and the pair mask.
    Antibody-antigen pairs and CDR rows/columns are cleared. Framework geometry,
    including the heavy-light interface, is left in place.
    """
    if cfg.d_t2d != 44:
        raise ValueError(
            f"d_t2d={cfg.d_t2d}, but template features are 44 channels "
            "(37 distance bins + 6 orientations + pair mask)."
        )
    length = ex.length
    seq = ex.seq.clone()
    if cfg.mask_loop_sequence:
        seq[ex.cdr_mask] = MASK_IDX

    seq_oh = _one_hot_seq(seq)  # [L, 22]

    msa_masked = torch.zeros(1, 1, length, 48)
    msa_masked[0, 0, :, :22] = seq_oh
    msa_masked[0, 0, :, 22:44] = seq_oh

    msa_full = torch.zeros(1, 1, length, 25)
    msa_full[0, 0, :, :22] = seq_oh

    t1d = torch.zeros(1, length, cfg.d_t1d)
    t1d[0, :, :20] = seq_oh[:, :20]
    t1d[0, :, 20] = seq_oh[:, MASK_IDX]
    # Timestep confidence: 1 on target, 1 - t/T on the whole antibody (SI Table 2).
    conf = timestep_conf.clone()
    conf[ex.target_mask] = 1.0
    t1d[0, :, 21] = conf

    if hotspots_override is None:
        true_hs = cdr_hotspots(
            ex.xyz, ex.target_mask, ex.cdr_mask, cfg.hotspot_cutoff, cfg.hotspot_k
        )
        t1d[0, :, 22] = sample_hotspot_labels(true_hs)
    else:
        t1d[0, :, 22] = hotspots_override.float()

    # True geometry, then drop the dock and the CDR so those blocks cannot be copied.
    t2d = xyz_to_t2d(ex.xyz[None, :, :3]).clone()  # [1, L, L, 44]
    loop = ex.cdr_mask
    tgt = ex.target_mask
    t2d[:, loop] = 0
    t2d[:, :, loop] = 0
    dock = tgt[:, None] ^ tgt[None, :]
    t2d[:, dock] = 0
    template_mask = _template_mask(ex)

    xyz_in = xyz_t[:, :3].clone()
    xyz_in[torch.isnan(xyz_in)] = 0.0

    idx = residue_index(ex.chain, ex.resseq, ex.icode)
    alpha = template_torsions(ex.xyz[:, :4], ex.chain, keep=~ex.cdr_mask)

    return {
        "seq": seq[None],
        "msa_masked": msa_masked,
        "msa_full": msa_full,
        "t1d": t1d[None],          # [B=1, T=1, L, 23]
        "t2d": t2d[None],          # [B=1, T=1, L, L, 44]
        "t2d_template": t2d[None].clone(),
        "xyz": xyz_in[None],       # noised backbone [B, L, 3, 3]
        "xyz_t": xyz_in[None, None],
        "idx": idx[None],
        "alpha_t": alpha[None, None],
        "framework_mask": ex.framework_mask,
        "target_mask": tgt,
        "cdr_mask": loop,
        "diffuse_mask": ex.diffuse_mask,
        "template_mask": template_mask,
        "seq_true": ex.seq[None],
        "xyz_true": ex.xyz[None, :, :3],
    }


def inject_self_cond(feats: dict, pred_xyz: torch.Tensor) -> dict:
    """Second-pass features: predicted geometry, with framework and target blocks restored."""
    pred = pred_xyz.detach()
    if pred.dim() == 5:
        pred = pred[-1]
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    sc = xyz_to_t2d(pred[..., :3, :])
    t2d = feats["t2d"]
    if sc.dim() == 4 and t2d.dim() == 5:
        sc = sc.unsqueeze(1)
    templ = feats["template_mask"]
    while templ.dim() < t2d.dim() - 1:
        templ = templ.unsqueeze(0)
    mixed = torch.where(templ[..., None], t2d, sc)
    out = dict(feats)
    out["t2d"] = mixed
    out["sc2d"] = sc
    out["xyz_sc"] = pred[:, None, :, :3]
    return out
