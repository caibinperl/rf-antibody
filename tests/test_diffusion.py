import torch

from rfab_finetune.constants import TrainConfig
from rfab_finetune.diffusion import FrameDiffuser
from rfab_finetune.synthetic import make_toy_complex


def _angle(rot_t, rot0):
    rel = rot_t @ rot0.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine)


def test_target_is_not_noised_and_motif_is_centered():
    ex = make_toy_complex()
    out = FrameDiffuser(TrainConfig()).noise_example(ex.xyz, ex.diffuse_mask, t=40)
    tgt = ex.target_mask
    com = out["xyz_true"][tgt, 1].mean(dim=0)
    assert torch.allclose(com, torch.zeros(3), atol=1e-4)
    assert torch.allclose(out["xyz_t"][tgt, :3], out["xyz_true"][tgt, :3], atol=1e-5)
    assert torch.allclose(out["rot_t"][tgt], out["rot0"][tgt], atol=1e-5)


def test_binder_moves_at_high_t():
    ex = make_toy_complex()
    diffuser = FrameDiffuser(TrainConfig())
    torch.manual_seed(1)
    out = diffuser.noise_example(ex.xyz, ex.diffuse_mask, t=50)
    binder = ex.diffuse_mask
    delta = (out["xyz_t"][binder, 1] - out["xyz_true"][binder, 1]).norm(dim=-1).mean()
    assert delta > 1.0


def test_igso3_rotation_grows_with_time():
    ex = make_toy_complex()
    diffuser = FrameDiffuser(TrainConfig())
    torch.manual_seed(0)
    low = diffuser.noise_example(ex.xyz, ex.diffuse_mask, t=1)
    torch.manual_seed(0)
    high = diffuser.noise_example(ex.xyz, ex.diffuse_mask, t=50)
    binder = ex.diffuse_mask
    assert float(_angle(high["rot_t"], high["rot0"])[binder].mean()) > float(
        _angle(low["rot_t"], low["rot0"])[binder].mean()
    )
