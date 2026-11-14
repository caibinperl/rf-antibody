from rfab_finetune.constants import TrainConfig
from rfab_finetune.geometry import frames_from_xyz, get_cb, xyz_to_t2d
from rfab_finetune.synthetic import make_toy_complex


def test_cb_and_frames_shapes():
    ex = make_toy_complex()
    cb = get_cb(ex.xyz[None])
    assert cb.shape == (1, ex.length, 3)
    rot, trans = frames_from_xyz(ex.xyz[None])
    assert rot.shape == (1, ex.length, 3, 3)
    assert trans.shape == (1, ex.length, 3)


def test_t2d_channels():
    ex = make_toy_complex()
    t2d = xyz_to_t2d(ex.xyz[None, :, :3])
    assert t2d.shape[-1] == 44
    assert t2d.shape[1] == t2d.shape[2] == ex.length
