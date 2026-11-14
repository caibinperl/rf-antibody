import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from rfab_finetune.constants import TrainConfig
from rfab_finetune.dataset import MixedAntibodyDataset, group_by_cluster
from rfab_finetune.model import _align_state_dict, _raise_if_partial_load
from rfab_finetune.pdb import crop_around_interface, parse_hlt
from rfab_finetune.synthetic import make_toy_complex, write_hlt
from rfab_finetune.train import train


def _atom(serial, name, resn, chain, resseq, icode, x):
    return (
        f"ATOM  {serial:5d}  {name:<3s} {resn} {chain}{resseq:4d}{icode:1s}   "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           {name[0]}\n"
    )


def test_hlt_roundtrip(tmp_path: Path):
    ex = make_toy_complex()
    pdb = tmp_path / "toy.pdb"
    write_hlt(ex, pdb)
    loaded = parse_hlt(pdb)
    assert loaded.length == ex.length
    assert loaded.cdr_mask.sum() == ex.cdr_mask.sum()
    assert loaded.chain[0] == "H"
    assert loaded.chain[-1] == "T"
    assert torch.allclose(loaded.xyz, ex.xyz, equal_nan=True)


def test_insertion_codes_and_remark_order(tmp_path: Path):
    pdb = tmp_path / "ins.pdb"
    pdb.write_text(
        "".join(
            [
                _atom(1, "CA", "ALA", "T", 1, "", 0.0),
                _atom(2, "N", "GLY", "H", 100, "", 1.0),
                _atom(3, "CA", "GLY", "H", 100, "", 2.0),
                _atom(4, "C", "GLY", "H", 100, "", 3.0),
                _atom(5, "N", "GLY", "H", 100, "A", 4.0),
                _atom(6, "CA", "GLY", "H", 100, "A", 5.0),
                _atom(7, "C", "GLY", "H", 100, "A", 6.0),
                "REMARK PDBinfo-LABEL:    2 H3\n",
                "END\n",
            ]
        )
    )
    loaded = parse_hlt(pdb)
    assert loaded.length == 3
    assert loaded.icode == ["", "A", ""]
    assert loaded.resseq[:2] == [100, 100]
    # Remark 2 is the first H residue in file order, which sorts ahead of the target.
    assert loaded.chain[0] == "H"
    assert bool(loaded.cdr_mask[0]) and not bool(loaded.cdr_mask[1])
    assert loaded.chain[-1] == "T"
    assert not bool(loaded.cdr_mask[-1])


def test_crop_keeps_cdrs_on_a_long_antigen():
    ex = make_toy_complex(n_h=12, n_t=40)
    cropped = crop_around_interface(ex, 10)
    assert cropped.length == 10
    assert int(cropped.cdr_mask.sum()) == int(ex.cdr_mask.sum())
    assert not any(c == "T" and r == 40 for c, r in zip(cropped.chain, cropped.resseq))


def test_clusters_are_sampled_as_groups():
    groups = group_by_cluster(
        [
            {"pdb": "a", "cluster": "c1"},
            {"pdb": "b", "cluster": "c1"},
            {"pdb": "c", "cluster": "c2"},
        ]
    )
    assert sorted(len(group) for group in groups) == [1, 2]


def test_dataset_dummy_toy():
    cfg = TrainConfig(dummy=True, examples_per_epoch=2, crop_size=64)
    ds = MixedAntibodyDataset(cfg)
    item = ds[0]
    assert "xyz" in item and "t" in item
    assert item["source"] == "sabdab"


def test_dataset_from_manifest(tmp_path: Path):
    ex = make_toy_complex()
    pdb = tmp_path / "a.pdb"
    write_hlt(ex, pdb)
    man = tmp_path / "sabdab.jsonl"
    man.write_text(json.dumps({"pdb": str(pdb), "cluster": "c1"}) + "\n")
    cfg = TrainConfig(sabdab_manifest=str(man), examples_per_epoch=1, dummy=True)
    item = MixedAntibodyDataset(cfg)[0]
    assert item["name"] == "a"


def test_partial_checkpoint_load_is_refused():
    module = nn.Linear(2, 2)
    try:
        _raise_if_partial_load(module, ["weight", "bias"], [])
    except RuntimeError as exc:
        assert "Refusing" in str(exc)
    else:
        raise AssertionError("partial load was accepted")
    aligned = _align_state_dict(
        {"module.weight": module.weight.detach().clone(), "module.bias": module.bias.detach().clone()},
        module,
    )
    assert set(aligned) == {"weight", "bias"}


def test_dummy_train_flushes_partial_batch(tmp_path: Path):
    cfg = TrainConfig(
        dummy=True,
        epochs=1,
        examples_per_epoch=5,
        pseudo_batch_size=4,
        warmup_steps=1,
        crop_size=64,
        ckpt_out=str(tmp_path / "model.pt"),
        log_dir=str(tmp_path / "logs"),
    )
    train(cfg, seed=0)
    lines = (tmp_path / "logs" / "train.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_resume_continues_from_checkpoint(tmp_path: Path):
    cfg = TrainConfig(
        dummy=True,
        epochs=1,
        examples_per_epoch=4,
        pseudo_batch_size=2,
        warmup_steps=1,
        crop_size=64,
        ckpt_out=str(tmp_path / "model.pt"),
        log_dir=str(tmp_path / "logs"),
    )
    train(cfg, seed=0)
    train(replace(cfg, epochs=2), seed=0, resume=True)
    lines = (tmp_path / "logs" / "train.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4
    blob = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert blob["epoch"] == 1
    assert "optimizer_state_dict" in blob


def test_dummy_train_one_step(tmp_path: Path):
    cfg = TrainConfig(
        dummy=True,
        epochs=1,
        examples_per_epoch=16,
        pseudo_batch_size=4,
        warmup_steps=1,
        crop_size=64,
        ckpt_out=str(tmp_path / "model.pt"),
        log_dir=str(tmp_path / "logs"),
    )
    train(cfg, seed=0)
    assert (tmp_path / "model.pt").exists()
