"""Parse HLT-formatted antibody complexes used by RFantibody."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List

import torch

from rfab_finetune.constants import AA_TO_IDX, CHOTHIA_CDR, UNK_IDX

ATOM_ORDER = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
_CHAIN_RANK = {"H": 0, "L": 1, "T": 2}


@dataclass
class ComplexExample:
    name: str
    xyz: torch.Tensor          # [L, 5, 3]  N,CA,C,O,CB
    seq: torch.Tensor          # [L] long
    chain: List[str]           # per-residue chain id (H/L/T)
    resseq: List[int]
    target_mask: torch.Tensor  # [L]
    cdr_mask: torch.Tensor     # [L]
    framework_mask: torch.Tensor
    diffuse_mask: torch.Tensor
    cdr_names: List[str] = field(default_factory=list)
    icode: List[str] = field(default_factory=list)
    source: str = "sabdab"     # sabdab | tcr | pdb_loop

    @property
    def length(self) -> int:
        return int(self.xyz.shape[0])


_REMARK_RE = re.compile(r"REMARK PDBinfo-LABEL:\s+(\d+)\s+(\S+)")


def _aa1(resname: str) -> str:
    table = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        "MSE": "M",
    }
    return table.get(resname.upper(), "X")


def _take(ex: ComplexExample, index: torch.Tensor) -> ComplexExample:
    """Keep residues in `index`, preserving their order."""
    index = index.long()
    chosen = index.tolist()

    def _list(values: list) -> list:
        return [values[i] for i in chosen]

    icode = ex.icode if len(ex.icode) == ex.length else [""] * ex.length
    names = ex.cdr_names if len(ex.cdr_names) == ex.length else [""] * ex.length
    return ComplexExample(
        name=ex.name,
        xyz=ex.xyz[index],
        seq=ex.seq[index],
        chain=_list(ex.chain),
        resseq=_list(ex.resseq),
        target_mask=ex.target_mask[index],
        cdr_mask=ex.cdr_mask[index],
        framework_mask=ex.framework_mask[index],
        diffuse_mask=ex.diffuse_mask[index],
        cdr_names=_list(names),
        icode=_list(icode),
        source=ex.source,
    )


def _permute(ex_parts: dict, perm: list) -> dict:
    index = torch.tensor(perm, dtype=torch.long)

    def _list(values: list) -> list:
        return [values[i] for i in perm]

    ex_parts["xyz"] = ex_parts["xyz"][index]
    ex_parts["seq"] = ex_parts["seq"][index]
    ex_parts["cdr_mask"] = ex_parts["cdr_mask"][index]
    ex_parts["chain"] = _list(ex_parts["chain"])
    ex_parts["resseq"] = _list(ex_parts["resseq"])
    ex_parts["icode"] = _list(ex_parts["icode"])
    ex_parts["cdr_names"] = _list(ex_parts["cdr_names"])
    return ex_parts


def parse_hlt(path: str | Path, source: str = "sabdab") -> ComplexExample:
    """Load an HLT PDB: chains H, L, T and optional CDR remarks.

    REMARK PDBinfo-LABEL indices are 1-based positions in file order, matching
    the RFantibody parser. Residues are then reordered to H, L, T. The residue
    key includes the insertion code so Chothia 100 / 100A / 100B stay distinct.
    """
    path = Path(path)
    atoms: Dict[tuple, Dict[str, List[float]]] = {}
    names: Dict[tuple, str] = {}
    order: List[tuple] = []
    remarks: Dict[int, str] = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("REMARK PDBinfo-LABEL"):
                match = _REMARK_RE.search(line)
                if match:
                    remarks[int(match.group(1))] = match.group(2)
                continue
            if not line.startswith("ATOM"):
                continue
            alt = line[16:17].strip()
            if alt not in ("", "A"):
                continue
            atom = line[12:16].strip()
            if atom not in ATOM_ORDER:
                continue
            resname = line[17:20].strip()
            chain_id = line[21].strip() or "T"
            resseq_i = int(line[22:26])
            icode = line[26:27].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            key = (chain_id, resseq_i, icode)
            if key not in atoms:
                order.append(key)
                atoms[key] = {}
            atoms[key][atom] = [x, y, z]
            names[key] = _aa1(resname)

    length = len(order)
    xyz = torch.full((length, 5, 3), float("nan"))
    seq = torch.full((length,), UNK_IDX, dtype=torch.long)
    chain: List[str] = []
    resseq: List[int] = []
    icode: List[str] = []
    for i, key in enumerate(order):
        chain.append(key[0])
        resseq.append(key[1])
        icode.append(key[2])
        seq[i] = AA_TO_IDX.get(names[key], UNK_IDX)
        for atom, idx in ATOM_ORDER.items():
            if atom in atoms[key]:
                xyz[i, idx] = torch.tensor(atoms[key][atom])

    cdr_mask = torch.zeros(length, dtype=torch.bool)
    cdr_names = [""] * length
    if remarks:
        for abs_idx, label in remarks.items():
            j = abs_idx - 1
            if 0 <= j < length and label[:2] in CHOTHIA_CDR:
                cdr_mask[j] = True
                cdr_names[j] = label[:2]

    perm = sorted(
        range(length),
        key=lambda i: (_CHAIN_RANK.get(chain[i], 9), resseq[i], icode[i]),
    )
    if perm != list(range(length)):
        parts = _permute(
            {
                "xyz": xyz,
                "seq": seq,
                "cdr_mask": cdr_mask,
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "cdr_names": cdr_names,
            },
            perm,
        )
        xyz, seq, cdr_mask = parts["xyz"], parts["seq"], parts["cdr_mask"]
        chain, resseq, icode, cdr_names = (
            parts["chain"], parts["resseq"], parts["icode"], parts["cdr_names"]
        )

    if not remarks:
        for i, (c, r) in enumerate(zip(chain, resseq)):
            for loop, (lo, hi) in CHOTHIA_CDR.items():
                if loop[0] == c and lo <= r <= hi:
                    cdr_mask[i] = True
                    cdr_names[i] = loop

    target_mask = torch.tensor([c == "T" for c in chain], dtype=torch.bool)
    framework_mask = ~target_mask & ~cdr_mask
    # SI: noise the antibody/TCR/loop chain. An unbound example noises everything.
    if bool(target_mask.any()):
        diffuse_mask = ~target_mask
    else:
        diffuse_mask = torch.ones(length, dtype=torch.bool)

    return ComplexExample(
        name=path.stem,
        xyz=xyz,
        seq=seq,
        chain=chain,
        resseq=resseq,
        target_mask=target_mask,
        cdr_mask=cdr_mask,
        framework_mask=framework_mask,
        diffuse_mask=diffuse_mask,
        cdr_names=cdr_names,
        icode=icode,
        source=source,
    )


def _closest(mask: torch.Tensor, dist: torch.Tensor, n_keep: int) -> torch.Tensor:
    chosen = torch.zeros_like(mask)
    if n_keep <= 0 or not bool(mask.any()):
        return chosen
    idx = mask.nonzero(as_tuple=False).flatten()
    order = torch.argsort(dist[idx], stable=True)
    chosen[idx[order[:n_keep]]] = True
    return chosen


def crop_around_interface(ex: ComplexExample, crop_size: int) -> ComplexExample:
    """Keep CDRs, the nearby epitope, then the closest remaining residues.

    Selection is by Cα distance to the CDRs. A contiguous residue-index window
    is pulled into the antigen whenever the target is the longest chain, and
    that drops the paratope.
    """
    if ex.length <= crop_size:
        return ex
    ca = torch.nan_to_num(ex.xyz[:, 1, :], nan=1e6)
    anchor_mask = ex.cdr_mask if bool(ex.cdr_mask.any()) else ex.diffuse_mask
    if not bool(anchor_mask.any()):
        anchor_mask = torch.ones(ex.length, dtype=torch.bool)
    dist = torch.cdist(ca, ca[anchor_mask]).min(dim=1).values
    dist = dist.clone()
    dist[anchor_mask] = 0

    cdr = ex.cdr_mask
    epitope = ex.target_mask & (dist <= 10.0) & ~cdr
    if int(cdr.sum()) >= crop_size:
        print(
            f"WARNING: {ex.name} has {int(cdr.sum())} CDR residues and crop_size={crop_size}; "
            "keeping the CDR residues closest to the epitope."
        )
        keep = _closest(cdr, dist, crop_size)
    else:
        must = cdr | epitope
        if int(must.sum()) > crop_size:
            keep = cdr.clone()
            keep = keep | _closest(epitope, dist, crop_size - int(cdr.sum()))
        else:
            keep = must.clone()
            keep = keep | _closest(~keep, dist, crop_size - int(keep.sum()))
    return _take(ex, keep.nonzero(as_tuple=False).flatten())


def with_centered_xyz(ex: ComplexExample, xyz: torch.Tensor) -> ComplexExample:
    return replace(ex, xyz=xyz)
