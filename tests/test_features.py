from dataclasses import replace

import torch

from rfab_finetune.constants import TrainConfig
from rfab_finetune.diffusion import FrameDiffuser
from rfab_finetune.features import featurize, inject_self_cond
from rfab_finetune.geometry import residue_index
from rfab_finetune.synthetic import make_toy_complex


def _featurize(ex, t, cfg=None):
    cfg = cfg or TrainConfig()
    noised = FrameDiffuser(cfg).noise_example(ex.xyz, ex.diffuse_mask, t=t)
    ex = replace(ex, xyz=noised["xyz_true"])
    return ex, featurize(ex, noised["xyz_t"], noised["timestep_conf"], cfg)


def test_feature_shapes():
    cfg = TrainConfig()
    ex = make_toy_complex()
    ex, feats = _featurize(ex, t=10, cfg=cfg)
    length = ex.length
    assert feats["msa_masked"].shape == (1, 1, length, 48)
    assert feats["msa_full"].shape == (1, 1, length, 25)
    assert feats["t1d"].shape == (1, 1, length, 23)
    assert feats["t2d"].shape == (1, 1, length, length, 44)
    assert feats["xyz"].shape == (1, length, 3, 3)
    assert feats["alpha_t"].shape == (1, 1, length, 30)
    assert float(feats["alpha_t"].abs().sum()) > 0
    # Target confidence is 1; binder is 1 - t/T.
    assert float(feats["t1d"][0, 0, ex.target_mask, 21].min()) == 1.0
    # CDR backbone torsions are masked (mask bit of psi, slot 2).
    psi_mask = feats["alpha_t"][0, 0, :, 8]
    assert float(psi_mask[ex.cdr_mask].abs().sum()) == 0.0
    assert float(psi_mask[ex.framework_mask].max()) == 1.0


def test_t2d_masks_dock_and_keeps_framework():
    ex = make_toy_complex(n_h=8, n_l=6, n_t=4)
    ex, feats = _featurize(ex, t=10)
    t2d = feats["t2d"][0, 0]
    tgt = ex.target_mask
    dock = tgt[:, None] ^ tgt[None, :]
    assert float(t2d[dock].abs().max()) == 0.0
    loop = ex.cdr_mask
    assert float(t2d[loop].abs().max()) == 0.0
    assert float(t2d[:, loop].abs().max()) == 0.0
    # Heavy-light framework geometry stays. Zeroing it was the bug.
    h = torch.tensor([c == "H" for c in ex.chain])
    light = torch.tensor([c == "L" for c in ex.chain])
    hl = (h[:, None] & light[None, :]) | (light[:, None] & h[None, :])
    assert float(t2d[hl].abs().sum()) > 0.0
    fw = ex.framework_mask
    intra = (fw[:, None] & fw[None, :]) & ~torch.eye(ex.length, dtype=torch.bool)
    assert float(t2d[intra, 43].min()) == 1.0


def test_selfcond_restores_template_blocks():
    ex = make_toy_complex()
    _, feats = _featurize(ex, t=8)
    shifted = feats["xyz"].clone()
    shifted[..., 0] += 3.0
    updated = inject_self_cond(feats, shifted)
    assert updated["t2d"].shape[-1] == 44
    templ = feats["template_mask"]
    assert torch.allclose(updated["t2d"][0, 0][templ], feats["t2d"][0, 0][templ])
    assert not torch.allclose(updated["t2d"][0, 0][~templ], feats["t2d"][0, 0][~templ])


def test_residue_index_keeps_insertions_and_chain_breaks():
    idx = residue_index(["H", "H", "H", "T"], [1, 2, 2, 10], ["", "", "A", ""])
    assert idx.tolist() == [1, 2, 3, 210]
