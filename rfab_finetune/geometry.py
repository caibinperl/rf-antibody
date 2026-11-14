"""Backbone frames, C? reconstruction, and t2d templates (RF kinematics)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rfab_finetune.constants import DBINS, DMAX, DMIN

# Ideal CB reconstruction coefficients used throughout RoseTTAFold / RFdiffusion.
_CB_A = -0.58273431
_CB_B = 0.56802827
_CB_C = -0.54067466


def get_cb(xyz: torch.Tensor) -> torch.Tensor:
    """Reconstruct CB from N, CA, C. xyz is [..., 3, 3] or [..., 27, 3]."""
    n = xyz[..., 0, :]
    ca = xyz[..., 1, :]
    c = xyz[..., 2, :]
    b = ca - n
    cc = c - ca
    a = torch.cross(b, cc, dim=-1)
    return _CB_A * a + _CB_B * b + _CB_C * cc + ca


def rigid_from_3_points(
    n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build residue frames. Returns rotation [..., 3, 3] and CA translation [..., 3]."""
    v1 = c - ca
    v2 = n - ca
    e1 = F.normalize(v1, dim=-1, eps=eps)
    u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)
    return rot, ca


def frames_from_xyz(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """xyz [..., L, 3+, 3] -> rotations [..., L, 3, 3], translations [..., L, 3]."""
    return rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :])


def apply_frames(rot: torch.Tensor, trans: torch.Tensor, xyz_local: torch.Tensor) -> torch.Tensor:
    """Place local N/CA/C atoms into global frames."""
    return torch.einsum("...ij,...aj->...ai", rot, xyz_local) + trans[..., None, :]


def rotation_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Frobenius MSE between predicted and true rotation matrices."""
    err = (pred - true).pow(2).sum(dim=(-1, -2)) / 9.0
    if mask is None:
        return err.mean()
    w = mask.to(err.dtype)
    return (err * w).sum() / w.sum().clamp_min(1.0)


def translation_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred - true).pow(2).sum(dim=-1) / 3.0
    if mask is None:
        return err.mean()
    w = mask.to(err.dtype)
    return (err * w).sum() / w.sum().clamp_min(1.0)


def dist_to_onehot(dist: torch.Tensor) -> torch.Tensor:
    dstep = (DMAX - DMIN) / DBINS
    bins = torch.linspace(
        DMIN + dstep, DMAX, DBINS, dtype=dist.dtype, device=dist.device
    )
    d = dist.clone()
    d[torch.isnan(d)] = 999.9
    idx = torch.bucketize(d.contiguous(), bins).long()
    return F.one_hot(idx, num_classes=DBINS + 1).float()


def distogram_targets(xyz: torch.Tensor) -> torch.Tensor:
    """Integer distogram class ids from CA/CB coordinates. xyz [B, L, 3+, 3]."""
    cb = get_cb(xyz)
    dist = torch.cdist(cb, cb)
    dist = dist + 999.9 * torch.eye(dist.shape[-1], device=dist.device)[None]
    dstep = (DMAX - DMIN) / DBINS
    bins = torch.linspace(DMIN + dstep, DMAX, DBINS, dtype=dist.dtype, device=dist.device)
    return torch.bucketize(dist.contiguous(), bins).long()


def xyz_to_t2d(xyz: torch.Tensor) -> torch.Tensor:
    """Pairwise template features: 36+1 dist bins, sin/cos of 3 angles, mask.

    xyz: [B, L, 3+, 3]  ->  t2d: [B, L, L, 44]
    """
    if xyz.dim() == 4 and xyz.shape[1] == 1:
        xyz = xyz[:, 0]
    bsz, length = xyz.shape[:2]
    n = xyz[:, :, 0]
    ca = xyz[:, :, 1]
    cb = get_cb(xyz)

    dist = torch.cdist(cb, cb)
    dist = dist + 999.9 * torch.eye(length, device=xyz.device)[None]
    dist_oh = dist_to_onehot(dist)

    def _dih(a, b, c, d):
        b0 = a - b
        b1 = c - b
        b2 = d - c
        b1 = F.normalize(b1, dim=-1)
        v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
        w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
        x = (v * w).sum(-1)
        y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
        return torch.atan2(y, x)

    def _ang(a, b, c):
        v = F.normalize(a - b, dim=-1)
        w = F.normalize(c - b, dim=-1)
        return torch.acos((v * w).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))

    # Pairwise angles are expensive if done densely; compute only where dist < DMAX.
    omega = torch.zeros(bsz, length, length, device=xyz.device)
    theta = torch.zeros_like(omega)
    phi = torch.zeros_like(omega)
    mask = dist < DMAX
    bi, i, j = torch.where(mask)
    if i.numel():
        omega[bi, i, j] = _dih(ca[bi, i], cb[bi, i], cb[bi, j], ca[bi, j])
        theta[bi, i, j] = _dih(n[bi, i], ca[bi, i], cb[bi, i], cb[bi, j])
        phi[bi, i, j] = _ang(ca[bi, i], cb[bi, i], cb[bi, j])

    orien = torch.stack(
        [torch.sin(omega), torch.cos(omega), torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)],
        dim=-1,
    ) * mask[..., None].to(xyz.dtype)
    t2d = torch.cat([dist_oh, orien, mask[..., None].to(xyz.dtype)], dim=-1)
    return t2d


def _cos_sin_dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Return cos/sin of the dihedral a-b-c-d. Last dim is 2."""
    b0 = a - b
    b1 = F.normalize(c - b, dim=-1, eps=1e-8)
    b2 = d - c
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
    norm = torch.sqrt(x * x + y * y).clamp_min(1e-8)
    return torch.stack((x / norm, y / norm), dim=-1)


def _atoms_finite(*atoms: torch.Tensor) -> torch.Tensor:
    ok = torch.ones(atoms[0].shape[0], dtype=torch.bool, device=atoms[0].device)
    for atom in atoms:
        ok = ok & torch.isfinite(atom).all(dim=-1)
    return ok


def template_torsions(
    xyz: torch.Tensor,
    chain: list,
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backbone torsion template, layout (cos, sin, mask) x 10 angles.

    Angles are omega, phi, psi, four chis, CB bend, CB twist, CG bend.
    Only omega/phi/psi are filled: the parsed PDB has no sidechain beyond CB,
    and chi angles would need those atoms. `keep` is False on CDR residues so
    the loop conformation is not copied into the template. Returns [L, 30].
    """
    length = xyz.shape[0]
    device = xyz.device
    tors = torch.zeros(length, 10, 2, device=device)
    mask = torch.zeros(length, 10, device=device)
    if keep is None:
        keep = torch.ones(length, dtype=torch.bool, device=device)
    else:
        keep = keep.to(device=device, dtype=torch.bool)

    n, ca, c, o = xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 3]
    if length > 1:
        same = torch.tensor(
            [chain[i] == chain[i + 1] for i in range(length - 1)],
            dtype=torch.bool,
            device=device,
        )
        both = keep[:-1] & keep[1:] & same
        omega = _cos_sin_dihedral(ca[:-1], c[:-1], n[1:], ca[1:])
        omega_ok = both & _atoms_finite(ca[:-1], c[:-1], n[1:], ca[1:])
        tors[:-1, 0] = omega
        mask[:-1, 0] = omega_ok.to(mask.dtype)

        phi = _cos_sin_dihedral(c[:-1], n[1:], ca[1:], c[1:])
        phi_ok = both & _atoms_finite(c[:-1], n[1:], ca[1:], c[1:])
        tors[1:, 1] = phi
        mask[1:, 1] = phi_ok.to(mask.dtype)

    # RoseTTAFold stores psi with the sign flipped relative to atan2(N, CA, C, O).
    psi = -_cos_sin_dihedral(n, ca, c, o)
    psi_ok = keep & _atoms_finite(n, ca, c, o)
    tors[:, 2] = psi
    mask[:, 2] = psi_ok.to(mask.dtype)

    fill = torch.zeros_like(tors)
    fill[..., 0] = 1.0
    tors = torch.where(mask.unsqueeze(-1) > 0.5, tors, fill)
    return torch.cat((tors, mask.unsqueeze(-1)), dim=-1).reshape(length, 30)


def residue_index(chain: list, resseq: list, icode: list | None = None) -> torch.Tensor:
    """PDB residue numbers, with +1 for insertion codes and +200 at chain breaks."""
    length = len(chain)
    icode = list(icode or [])
    if len(icode) < length:
        icode = icode + [""] * (length - len(icode))
    adjusted = []
    prev_c, prev_r, prev_out = None, None, 0
    for c, r, _ic in zip(chain, resseq, icode):
        r = int(r)
        if c != prev_c or prev_c is None:
            out = r
        elif r > prev_r:
            out = prev_out + (r - prev_r)
        else:
            out = prev_out + 1
        adjusted.append(out)
        prev_c, prev_r, prev_out = c, r, out
    idx = torch.tensor(adjusted, dtype=torch.long)
    for i in range(1, length):
        if chain[i] != chain[i - 1]:
            idx[i:] += 200
    return idx
