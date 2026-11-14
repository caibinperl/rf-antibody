import torch

from rfab_finetune.hotspots import cdr_hotspots, sample_hotspot_labels
from rfab_finetune.synthetic import make_toy_complex


def test_contacting_complex_has_hotspots():
    ex = make_toy_complex(contact=True)
    hs = cdr_hotspots(ex.xyz, ex.target_mask, ex.cdr_mask, cutoff=8.0, k=5)
    assert hs.any()
    assert not hs[ex.cdr_mask].any()
    assert hs[ex.target_mask].any()


def test_far_complex_has_no_hotspots():
    ex = make_toy_complex(contact=False)
    hs = cdr_hotspots(ex.xyz, ex.target_mask, ex.cdr_mask, cutoff=8.0, k=5)
    assert not hs.any()


def test_hotspot_subsampling_is_subset():
    torch.manual_seed(0)
    hs = torch.zeros(20, dtype=torch.bool)
    hs[10:16] = True
    labels = sample_hotspot_labels(hs)
    assert (labels[~hs] == 0).all()
    assert int(labels.sum().item()) <= 6
