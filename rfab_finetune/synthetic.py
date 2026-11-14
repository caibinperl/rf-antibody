"""Deterministic mini-complexes for tests and --dummy dry runs."""

from __future__ import annotations

import torch

from rfab_finetune.constants import AA_TO_IDX
from rfab_finetune.pdb import ComplexExample


def _backbone(ca: torch.Tensor, cb_z: float) -> torch.Tensor:
    atoms = torch.zeros(5, 3)
    atoms[0] = ca + torch.tensor([-0.8, 0.5, 0.0])
    atoms[1] = ca
    atoms[2] = ca + torch.tensor([0.8, 0.4, 0.0])
    atoms[3] = ca + torch.tensor([0.9, 1.6, 0.0])
    atoms[4] = ca + torch.tensor([0.0, -0.8, cb_z])
    return atoms


def make_toy_complex(
    n_h: int = 12, n_t: int = 10, n_l: int = 0, contact: bool = True
) -> ComplexExample:
    """A short binder (chain H, optional L) plus a target (chain T)."""
    length = n_h + n_l + n_t
    xyz = torch.zeros(length, 5, 3)
    y_binder = 4.0 if contact else 40.0
    for i in range(n_h):
        xyz[i] = _backbone(torch.tensor([float(i), y_binder, 0.0]), 1.1)
    for i in range(n_l):
        xyz[n_h + i] = _backbone(torch.tensor([float(i), y_binder, 5.0]), 1.1)
    for j in range(n_t):
        xyz[n_h + n_l + j] = _backbone(torch.tensor([float(j), 0.0, 0.0]), -1.1)

    seq = torch.zeros(length, dtype=torch.long)
    seq[: n_h + n_l] = AA_TO_IDX["S"]
    seq[n_h + n_l :] = AA_TO_IDX["A"]
    chain = ["H"] * n_h + ["L"] * n_l + ["T"] * n_t
    resseq = (
        list(range(1, n_h + 1))
        + list(range(1, n_l + 1))
        + list(range(1, n_t + 1))
    )
    cdr = torch.zeros(length, dtype=torch.bool)
    cdr[max(n_h - 4, 0) : n_h] = True
    target = torch.tensor([c == "T" for c in chain])
    framework = ~target & ~cdr
    return ComplexExample(
        name="toy_vhh",
        xyz=xyz,
        seq=seq,
        chain=chain,
        resseq=resseq,
        target_mask=target,
        cdr_mask=cdr,
        framework_mask=framework,
        diffuse_mask=~target,
        cdr_names=[("H3" if cdr[i] else "") for i in range(length)],
        icode=[""] * length,
        source="sabdab",
    )


def write_hlt(ex: ComplexExample, path: str) -> None:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    three = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }
    atoms = ["N", "CA", "C", "O", "CB"]
    serial = 1
    lines = []
    icodes = ex.icode if len(ex.icode) == ex.length else [""] * ex.length
    for i in range(ex.length):
        resn = three[aa[int(ex.seq[i].clamp(0, 19))]]
        for aidx, aname in enumerate(atoms):
            x, y, z = ex.xyz[i, aidx].tolist()
            lines.append(
                f"ATOM  {serial:5d}  {aname:<3s} {resn} {ex.chain[i]}{ex.resseq[i]:4d}{icodes[i]:1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {aname[0]:s}"
            )
            serial += 1
    for i, name in enumerate(ex.cdr_names):
        if name:
            lines.append(f"REMARK PDBinfo-LABEL:{i + 1:5d} {name}")
    lines.append("END")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
