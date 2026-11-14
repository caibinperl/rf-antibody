"""Forward noising of residue frames.

Matches vanilla RFdiffusion: translations follow the Euclidean beta schedule,
rotations are drawn from IGSO(3). Only binder residues are noised. The motif
(target) is centered at the origin first, which is what `diffuse_pose` does so
absolute crystal coordinates cannot leak the dock.
"""

from __future__ import annotations

import math

import torch

from rfab_finetune.constants import TrainConfig
from rfab_finetune.geometry import apply_frames, frames_from_xyz

# (sigma tuple) -> (omega [W], cdf [T, W])
_IGSO3_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def beta_schedule(cfg: TrainConfig, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear schedule, rescaled as in RFdiffusion (`b *= 200 / T`)."""
    t = cfg.T
    b0 = cfg.b0 * (200.0 / t)
    bT = cfg.bT * (200.0 / t)
    beta = torch.linspace(b0, bT, t, device=device)
    alphabar = torch.cumprod(1.0 - beta, dim=0)
    return beta, alphabar


def so3_sigma(t: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    """Linear IGSO(3) schedule. `t` is 1-indexed in `[1, T]`."""
    cont = t.float() / cfg.T
    return cfg.so3_min_sigma + cont * cfg.so3_min_b + 0.5 * cont.square() * (cfg.so3_max_b - cfg.so3_min_b)


def _build_igso3_cdf(
    sigmas: torch.Tensor, num_omega: int = 512, truncation: int = 1000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-CDF table for the IGSO(3) rotation angle at each sigma.

    Density on the rotation angle ω is the truncated series
    `(1 - cos ω) / π * Σ_l (2l+1) exp(-l(l+1)σ²/2) sin((l+1/2)ω) / sin(ω/2)`.
    """
    omega = torch.linspace(0, math.pi, num_omega + 1, dtype=torch.float64)[1:]
    sig = sigmas.detach().to(dtype=torch.float64).cpu()
    expansion = torch.zeros(sig.shape[0], omega.shape[0], dtype=torch.float64)
    half = torch.sin(omega / 2).clamp_min(1e-8)
    levels = torch.arange(truncation, dtype=torch.float64)
    for start in range(0, truncation, 32):
        ls = levels[start : start + 32]
        coef = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * sig.pow(2)[:, None] / 2.0)
        basis = torch.sin(omega[:, None] * (ls + 0.5)) / half[:, None]
        expansion += coef @ basis.T
    pdf = (expansion * (1.0 - torch.cos(omega)) / math.pi).clamp_min(0)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
    return omega.float(), cdf.float()


def _igso3_table(cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = so3_sigma(torch.arange(1, cfg.T + 1), cfg)
    key = tuple(round(float(s), 6) for s in sigmas)
    cached = _IGSO3_CACHE.get(key)
    if cached is None:
        cached = _build_igso3_cdf(sigmas)
        _IGSO3_CACHE[key] = cached
    return cached


def _axis_angle_to_matrix(rotvec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues formula. rotvec [..., 3] -> [..., 3, 3]."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp_min(eps)
    k = rotvec / theta
    kx, ky, kz = k.unbind(-1)
    zeros = torch.zeros_like(kx)
    k_cross = torch.stack(
        [zeros, -kz, ky, kz, zeros, -kx, -ky, kx, zeros], dim=-1
    ).reshape(*rotvec.shape[:-1], 3, 3)
    outer = k[..., :, None] * k[..., None, :]
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand_as(k_cross)
    st = torch.sin(theta)[..., None]
    ct = torch.cos(theta)[..., None]
    return eye * ct + st * k_cross + (1.0 - ct) * outer


def center_coords(xyz: torch.Tensor, diffuse_mask: torch.Tensor) -> torch.Tensor:
    """Subtract the motif Cα center of mass. Motif = residues that are not noised."""
    ca = xyz[:, 1, :]
    finite = torch.isfinite(ca).all(dim=-1)
    motif = (~diffuse_mask) & finite
    use = motif if bool(motif.any()) else finite
    if not bool(use.any()):
        return xyz.clone()
    com = ca[use].mean(dim=0)
    return xyz - com


class FrameDiffuser:
    def __init__(self, cfg: TrainConfig, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device
        _, self.alphabar = beta_schedule(cfg, device)
        self._igso3 = _igso3_table(cfg)

    def _sample_rotation_vectors(self, t: int, length: int, device: torch.device) -> torch.Tensor:
        """IGSO(3) samples as axis-angle vectors. Angle from the series CDF, axis uniform."""
        omega, cdf = self._igso3
        u = torch.rand(length, device=device)
        cdf_t = cdf[t - 1].to(device)
        bins = omega.to(device)
        idx = torch.searchsorted(cdf_t, u).clamp(max=bins.shape[0] - 1)
        angles = bins[idx]
        axis = torch.randn(length, 3, device=device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return axis * angles[:, None]

    def noise_example(
        self,
        xyz: torch.Tensor,
        diffuse_mask: torch.Tensor,
        t: int | None = None,
    ) -> dict:
        """Noise binder frames at a random (or given) timestep.

        Args:
            xyz: [L, 3+, 3] ground-truth coordinates (unscaled).
            diffuse_mask: [L] True = residue is noised (binder).
            t: 1-indexed timestep in [1, T]. Sampled uniformly if omitted.

        Returns:
            dict with xyz_t, xyz_true (motif-centered), t, rotations and translations.
        """
        if t is None:
            t = int(torch.randint(1, self.cfg.T + 1, (1,)).item())
        t_idx = t - 1
        device = xyz.device
        length = xyz.shape[0]
        diffuse_mask = diffuse_mask.to(device)
        xyz = center_coords(xyz.to(device), diffuse_mask)
        rot0, trans0 = frames_from_xyz(xyz[None])
        rot0, trans0 = rot0[0], trans0[0]

        abar = self.alphabar.to(device)[t_idx]
        trans_scale = self.cfg.crd_scale
        trans0_s = trans0 * trans_scale
        eps = torch.randn_like(trans0_s)
        trans_t = torch.sqrt(abar) * trans0_s + torch.sqrt(1.0 - abar) * eps
        trans_t = torch.where(diffuse_mask[:, None], trans_t, trans0_s)

        rotvec = self._sample_rotation_vectors(t, length, device)
        rotvec = torch.where(diffuse_mask[:, None], rotvec, torch.zeros_like(rotvec))
        r_noise = _axis_angle_to_matrix(rotvec)
        rot_t = torch.matmul(r_noise, rot0)
        rot_t = torch.where(diffuse_mask[:, None, None], rot_t, rot0)

        # Local N/CA/C relative to the true CA / frame, then place into noisy frames.
        bb = xyz[:, :3] - trans0[:, None, :]
        local = torch.matmul(
            rot0.transpose(-1, -2).unsqueeze(1), bb.unsqueeze(-1)
        ).squeeze(-1)
        xyz_bb = apply_frames(rot_t, trans_t / trans_scale, local)

        xyz_t = xyz.clone()
        xyz_t[:, :3] = xyz_bb
        # Target and binder sidechains are not an input. Backbone atoms carry the frame.
        xyz_t[:, 3:] = float("nan")

        conf = torch.full((length,), 1.0 - t / self.cfg.T, device=device)
        return {
            "xyz_t": xyz_t,
            "xyz_true": xyz,
            "t": t,
            "timestep_conf": conf,
            "rot0": rot0,
            "trans0": trans0,
            "rot_t": rot_t,
            "trans_t": trans_t / trans_scale,
        }
