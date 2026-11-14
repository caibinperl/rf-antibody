"""Mixed SAbDab / TCR / PDB-loop dataset (SI section 3.1)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from rfab_finetune.constants import TrainConfig
from rfab_finetune.diffusion import FrameDiffuser
from rfab_finetune.features import featurize
from rfab_finetune.pdb import crop_around_interface, parse_hlt, with_centered_xyz


def load_manifest(path: str | Path) -> List[dict]:
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        print(f"WARNING: manifest not found: {path}")
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def group_by_cluster(rows: List[dict]) -> List[List[dict]]:
    """One group per cluster id. A missing id is its own cluster."""
    groups: dict[str, List[dict]] = {}
    for row in rows:
        key = str(row.get("cluster") or row["pdb"])
        groups.setdefault(key, []).append(row)
    return list(groups.values())


class MixedAntibodyDataset(Dataset):
    """Sample training clusters with the 50 / 10 / 40 mixture, then noise one complex."""

    def __init__(self, cfg: TrainConfig, epoch_size: int | None = None):
        self.cfg = cfg
        self.epoch_size = epoch_size or cfg.examples_per_epoch
        self.pools = {
            "sabdab": load_manifest(cfg.sabdab_manifest),
            "tcr": load_manifest(cfg.tcr_manifest),
            "pdb_loop": load_manifest(cfg.pdb_loop_manifest),
        }
        self._toy = None
        if not any(self.pools.values()) and cfg.dummy:
            from rfab_finetune.synthetic import make_toy_complex

            self._toy = make_toy_complex()
            self.pools["sabdab"] = [{"pdb": "__toy__", "cluster": "toy"}]
        self.probs = {
            "sabdab": cfg.sabdab_prob,
            "tcr": cfg.tcr_prob,
            "pdb_loop": cfg.pdb_loop_prob,
        }
        self.clusters = {name: group_by_cluster(rows) for name, rows in self.pools.items()}
        self._warn_mixture()
        self.diffuser = FrameDiffuser(cfg)

    def _warn_mixture(self) -> None:
        if self._toy:
            return
        missing = [name for name, rows in self.pools.items() if not rows]
        if not missing:
            return
        active = self.available_sources()
        if not active:
            print("WARNING: no complexes loaded; every manifest is empty.")
            return
        weights = torch.tensor([self.probs[name] for name in active], dtype=torch.float)
        weights = weights / weights.sum()
        shown = ", ".join(f"{name}={float(w):.2f}" for name, w in zip(active, weights))
        print(f"WARNING: manifests missing for {missing}; mixture renormalized to {shown}")

    def available_sources(self) -> List[str]:
        return [k for k, v in self.pools.items() if v]

    def _sample_source(self) -> str:
        keys = self.available_sources()
        if not keys:
            raise RuntimeError(
                "No training manifests found. Write jsonl files of "
                '{"pdb": "/path/to/file.pdb", "cluster": "..."} and pass them '
                "as --sabdab-manifest / --tcr-manifest / --pdb-loop-manifest."
            )
        weights = torch.tensor([self.probs[k] for k in keys], dtype=torch.float)
        weights = weights / weights.sum()
        idx = int(torch.multinomial(weights, 1).item())
        return keys[idx]

    def _sample_row(self, source: str) -> dict:
        cluster = random.choice(self.clusters[source])
        return random.choice(cluster)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict:
        source = self._sample_source()
        row = self._sample_row(source)
        if row["pdb"] == "__toy__":
            ex = self._toy
        else:
            ex = parse_hlt(row["pdb"], source=source)
        ex = crop_around_interface(ex, self.cfg.crop_size)
        noised = self.diffuser.noise_example(ex.xyz, ex.diffuse_mask)
        ex = with_centered_xyz(ex, noised["xyz_true"])
        feats = featurize(ex, noised["xyz_t"], noised["timestep_conf"], self.cfg)
        feats["t"] = noised["t"]
        feats["source"] = source
        feats["name"] = ex.name
        return feats


def collate_singleton(batch: list) -> dict:
    """RFdiffusion trains at batch size 1 (pseudo-batch via grad accumulation)."""
    assert len(batch) == 1
    return batch[0]
