"""Model wrappers: dummy (tests / dry-run) and official RFantibody RoseTTAFold."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfab_finetune.constants import TrainConfig


class DummyDiffusionModel(nn.Module):
    """Tiny stand-in so the data / loss / train loop can be tested without SE3-Transformer."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(3))
        self.dist_bias = nn.Parameter(torch.zeros(37))
        self.aa_bias = nn.Parameter(torch.zeros(22))

    def forward(self, feats: dict) -> dict:
        xyz = feats["xyz"]
        bsz, length = xyz.shape[:2]
        pred_xyz = xyz + self.shift.view(1, 1, 1, 3)
        logits_dist = self.dist_bias.view(1, 37, 1, 1).expand(bsz, 37, length, length).contiguous()
        logits_aa = self.aa_bias.view(1, 22, 1).expand(bsz, 22, length).contiguous()
        return {"xyz": pred_xyz, "logits_dist": logits_dist, "logits_aa": logits_aa}


def _first_logit(logits: Any) -> torch.Tensor:
    if isinstance(logits, (tuple, list)):
        return logits[0]
    return logits


class RFDiffusionAb(nn.Module):
    """Load RoseTTAFoldModule from RFantibody (or vanilla RFdiffusion) and map I/O."""

    def __init__(self, cfg: TrainConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        try:
            from rfantibody.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
        except ImportError as exc:
            raise ImportError(
                "Install RFantibody (https://github.com/RosettaCommons/RFantibody) "
                "to fine-tune the real network, or pass --dummy."
            ) from exc

        model_cfg = _default_model_kwargs(cfg)
        ckpt = None
        if cfg.ckpt_in:
            ckpt = torch.load(cfg.ckpt_in, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config_dict" in ckpt and "model" in ckpt["config_dict"]:
                model_cfg.update(_flatten_model_cfg(ckpt["config_dict"]["model"]))
                model_cfg["d_t1d"] = ckpt["config_dict"].get("preprocess", {}).get("d_t1d", cfg.d_t1d)
                model_cfg["d_t2d"] = ckpt["config_dict"].get("preprocess", {}).get("d_t2d", cfg.d_t2d)
                model_cfg["T"] = ckpt["config_dict"].get("diffuser", {}).get("T", cfg.T)
                model_cfg["use_selfcond_emb"] = ckpt["config_dict"].get("preprocess", {}).get(
                    "use_selfcond_emb", False
                )
        if model_cfg["d_t2d"] != 44:
            raise RuntimeError(
                f"Checkpoint d_t2d={model_cfg['d_t2d']}, but the training features are 44-d "
                "(37 distance bins + 6 orientations + pair mask)."
            )

        self.model = RoseTTAFoldModule(**model_cfg)
        if ckpt is not None:
            state = ckpt.get("model_state_dict") or ckpt.get("final_state_dict") or ckpt
            state = _align_state_dict(state, self.model)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            _raise_if_partial_load(self.model, missing, unexpected)

    def forward(self, feats: dict) -> dict:
        seq = feats["seq"]
        if seq.dim() == 2:
            seq_oh = F.one_hot(seq, num_classes=22).float()
        else:
            seq_oh = seq.float()
        t = feats.get("t", 1)
        t_tensor = torch.tensor(t, device=seq.device) if not torch.is_tensor(t) else t
        xyz_bb = feats["xyz"]
        if xyz_bb.dim() == 3:
            xyz_bb = xyz_bb.unsqueeze(0)

        t1d = feats["t1d"]
        t2d = feats["t2d"]
        # Model templ_emb expects [B, T, L, d]
        if t1d.dim() == 3:
            t1d = t1d.unsqueeze(1)
        if t2d.dim() == 4:
            t2d = t2d.unsqueeze(1)

        xyz_t = feats.get("xyz_t")
        if xyz_t is None:
            zeros = torch.zeros(*xyz_bb.shape[:-2], 24, 3, device=xyz_bb.device, dtype=xyz_bb.dtype)
            xyz_t = torch.cat([xyz_bb, zeros], dim=-2).unsqueeze(1)
        alpha_t = feats.get("alpha_t", torch.zeros(xyz_bb.shape[0], 1, xyz_bb.shape[1], 30, device=xyz_bb.device))

        # A separate self-conditioning track keeps the template t2d and reads sc2d.
        # Without that module, self-conditioning is already mixed into t2d.
        sc2d = feats.get("sc2d")
        xyz_sc = feats.get("xyz_sc")
        if hasattr(self.model, "selfcond_emb"):
            t2d = feats.get("t2d_template", t2d)
            if torch.is_tensor(t2d) and t2d.dim() == 4:
                t2d = t2d.unsqueeze(1)
            if sc2d is None:
                sc2d = torch.zeros_like(t2d)
            elif sc2d.dim() == 4:
                sc2d = sc2d.unsqueeze(1)
            if xyz_sc is None:
                xyz_sc = torch.zeros_like(xyz_t)

        motif_mask = ~feats["diffuse_mask"]
        if motif_mask.dim() == 1:
            motif_mask = motif_mask.unsqueeze(0)

        out = self.model(
            msa_latent=feats["msa_masked"],
            msa_full=feats["msa_full"],
            seq=seq_oh if seq_oh.dim() == 3 else seq_oh,
            xyz=xyz_bb,
            idx=feats["idx"],
            t=t_tensor,
            t1d=t1d,
            t2d=t2d,
            xyz_t=xyz_t,
            alpha_t=alpha_t,
            sc2d=sc2d,
            xyz_sc=xyz_sc,
            use_checkpoint=self.cfg.use_checkpoint,
            motif_mask=motif_mask,
            return_w_msa_prev=False,
        )
        logits, logits_aa, _logits_exp, xyz_pred, _alpha, _lddt = out[:6]
        # xyz_pred is [blocks, B, L, 3, 3]; take last block.
        if xyz_pred.dim() == 5:
            xyz_pred = xyz_pred[-1]
        return {
            "xyz": xyz_pred,
            "logits_dist": _first_logit(logits),
            "logits_aa": logits_aa,
        }


def _align_state_dict(state: dict, module: nn.Module) -> dict:
    """Drop module./model. prefixes when that matches the network."""
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state is {type(state).__name__}, expected a dict of tensors.")
    model_keys = set(module.state_dict())

    def score(keys) -> int:
        return sum(1 for k in list(keys)[:64] if k in model_keys)

    stripped = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        stripped[name] = value
    unwrapped = {k[6:] if k.startswith("model.") else k: v for k, v in stripped.items()}
    return max((state, stripped, unwrapped), key=lambda candidate: score(candidate))


def _raise_if_partial_load(module: nn.Module, missing: list, unexpected: list) -> None:
    n_param = len(module.state_dict())
    critical = [key for key in missing if "templ_emb.emb.weight" in key]
    if critical or (n_param and len(missing) / n_param > 0.02):
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)}/{n_param} parameters "
            f"(first missing: {missing[:8]}). Refusing to train from a partial load."
        )
    if missing:
        print(f"Missing keys while loading ckpt ({len(missing)}): {missing[:8]} ...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:8]} ...")


def _default_model_kwargs(cfg: TrainConfig) -> dict:
    return dict(
        n_extra_block=4,
        n_main_block=32,
        n_ref_block=4,
        d_msa=256,
        d_msa_full=64,
        d_pair=128,
        d_templ=64,
        n_head_msa=8,
        n_head_pair=4,
        n_head_templ=4,
        d_hidden=32,
        d_hidden_templ=32,
        p_drop=0.15,
        d_t1d=cfg.d_t1d,
        d_t2d=cfg.d_t2d,
        d_time_emb=None,
        d_time_emb_proj=None,
        T=cfg.T,
        use_motif_timestep=False,
        freeze_track_motif=False,
        use_selfcond_emb=False,
        SE3_param_full={
            "num_layers": 1, "num_channels": 32, "num_degrees": 2, "n_heads": 4, "div": 4,
            "l0_in_features": 8, "l0_out_features": 8, "l1_in_features": 3, "l1_out_features": 2,
            "num_edge_features": 32,
        },
        SE3_param_topk={
            "num_layers": 1, "num_channels": 32, "num_degrees": 2, "n_heads": 4, "div": 4,
            "l0_in_features": 64, "l0_out_features": 64, "l1_in_features": 3, "l1_out_features": 2,
            "num_edge_features": 64,
        },
        input_seq_onehot=True,
    )


def _flatten_model_cfg(raw: dict) -> dict:
    keep = {
        "n_extra_block", "n_main_block", "n_ref_block", "d_msa", "d_msa_full", "d_pair",
        "d_templ", "n_head_msa", "n_head_pair", "n_head_templ", "d_hidden", "d_hidden_templ",
        "p_drop", "SE3_param_full", "SE3_param_topk", "freeze_track_motif", "use_motif_timestep",
        "d_time_emb", "d_time_emb_proj", "input_seq_onehot",
    }
    return {k: v for k, v in raw.items() if k in keep}


def build_model(cfg: TrainConfig, device: torch.device) -> nn.Module:
    if cfg.dummy:
        return DummyDiffusionModel(cfg).to(device)
    return RFDiffusionAb(cfg, device).to(device)
