from dataclasses import replace

import torch

from rfab_finetune.constants import TrainConfig
from rfab_finetune.diffusion import FrameDiffuser
from rfab_finetune.features import featurize
from rfab_finetune.losses import frame_loss, mlm_time_weight, total_loss
from rfab_finetune.model import DummyDiffusionModel
from rfab_finetune.synthetic import make_toy_complex


def test_mlm_weight_larger_at_small_t():
    cfg = TrainConfig()
    assert mlm_time_weight(5, cfg) > mlm_time_weight(40, cfg)


def _prepared(cfg, t):
    ex = make_toy_complex()
    noised = FrameDiffuser(cfg).noise_example(ex.xyz, ex.diffuse_mask, t=t)
    ex = replace(ex, xyz=noised["xyz_true"])
    feats = featurize(ex, noised["xyz_t"], noised["timestep_conf"], cfg)
    return feats


def test_frame_loss_ignores_the_target():
    cfg = TrainConfig()
    feats = _prepared(cfg, t=5)
    true = feats["xyz_true"]
    moved = true.clone()
    moved[0, feats["target_mask"], 1] += 5
    mask = feats["diffuse_mask"].float()[None]
    assert float(frame_loss(moved, true, mask)) < 1e-5
    moved[0, feats["diffuse_mask"], 1] += 5
    assert float(frame_loss(moved, true, mask)) > 1.0


def test_dummy_forward_and_loss():
    cfg = TrainConfig(dummy=True)
    feats = _prepared(cfg, t=12)
    model = DummyDiffusionModel(cfg)
    pred = model(feats)
    loss, stats = total_loss(pred, feats, t=12, cfg=cfg)
    assert torch.isfinite(loss)
    assert set(stats) >= {"loss", "l_frame", "l_2d", "l_mlm"}
    loss.backward()
    assert model.shift.grad is not None
