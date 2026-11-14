#!/usr/bin/env python3
"""Download a small real example set and write HLT PDBs plus jsonl manifests.

The full RFdiffusion antibody training table is the Zenodo CSV
(10.5281/zenodo.15741710, CC-BY 4.0). Coordinates come from the RCSB PDB.
Antibody CDRs follow Chothia boundaries returned by Abnum
(http://www.bioinf.org.uk/abs/abnum/). TCR CDR sequences are the STCRDab
annotations for 1AO7 and 2CKB. Protease-inhibitor loops are the interface
stretch on the inhibitor.

This is an example subset, not the full training corpus.
"""

from __future__ import annotations

import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CSV_PATH = DATA / "AntibodyTrainingDataset.csv"
CSV_MD5 = "e2eb7b90e3733e0ac9247dd53857c5aa"
ZENODO_CSV = (
    "https://zenodo.org/api/records/15741710/files/"
    "AntibodyTrainingDataset.csv/content"
)
ABNUM_URL = "http://www.bioinf.org.uk/abs/abnum/abnum.cgi"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

AA3 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}
BACKBONE = ("N", "CA", "C", "O", "CB")
# Same Chothia ranges as rfab_finetune.constants.CHOTHIA_CDR.
CDR_RANGES = {
    "H1": (26, 32), "H2": (52, 56), "H3": (95, 102),
    "L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
}
FV_CROP = {"H": 115, "L": 110}

# Preferred rows from the Zenodo table, with fallbacks if numbering fails.
SABDAB_CANDIDATES = [
    ("vhh_protein", "2p45", "B", "", "A"),
    ("vhh_protein", "6jb2", "A", "", "B"),
    ("vhh_protein", "6z1v", "B", "", "A"),
    ("vhh_protein", "7n0r", "D", "", "B"),
    ("fab_protein", "6ddm", "B", "A", "C"),
    ("fab_protein", "5kvg", "H", "L", "E"),
    ("fab_protein", "2vxt", "H", "L", "I"),
    ("fab_protein", "7wsl", "H", "L", "D"),
    ("fab_peptide", "6uce", "H", "L", "C"),
    ("fab_peptide", "6wfy", "H", "L", "P"),
    ("fab_peptide", "5vxr", "H", "L", "P"),
    ("vhh_peptide", "6s0y", "B", "", "D"),
    ("vhh_peptide", "5hm1", "C", "", "F"),
]
SABDAB_QUOTA = {
    "vhh_protein": 2,
    "fab_protein": 2,
    "fab_peptide": 1,
    "vhh_peptide": 1,
}

# Beta -> H, alpha -> L, pMHC chains -> T. CDR strings are STCRDab IMGT loops.
TCR_EXAMPLES = [
    {
        "pdb": "1ao7",
        "name": "A6 TCR / Tax / HLA-A*02:01",
        "heavy": "E",
        "light": "D",
        "targets": ["A", "B", "C"],
        "cdrs": {
            "H": {"H1": "MNHEY", "H2": "SVGAGI", "H3": "ASRPGLAGGRPEQY"},
            "L": {"L1": "DRGSQS", "L2": "IYSNGD", "L3": "AVTTDSWGKLQ"},
        },
    },
    {
        "pdb": "2ckb",
        "name": "2C TCR / dEV8 / H-2Kb",
        "heavy": "B",
        "light": "A",
        "targets": ["H", "L", "P"],
        "cdrs": {
            "H": {"H1": "NNHNN", "H2": "SYGAGS", "H3": "ASGGGGTLY"},
            "L": {"L1": "YSATPY", "L2": "YYSGDPVV", "L3": "AVSGFASALT"},
        },
    },
]

# Inhibitor is the binder (H); protease is the target (T).
LOOP_EXAMPLES = [
    {
        "pdb": "2ptc",
        "name": "BPTI / trypsin",
        "binder": "I",
        "targets": ["E"],
        "expect": "PCK",
    },
    {
        "pdb": "1cho",
        "name": "turkey ovomucoid / alpha-chymotrypsin",
        "binder": "I",
        "targets": ["E", "F", "G"],
        "expect": "CTLE",
    },
]


def fetch(url: str, dest: Path | None = None, data: bytes | None = None, headers: dict | None = None) -> bytes:
    hdrs = {"User-Agent": UA}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs)
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
        blob = resp.read()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    return blob


def ensure_csv() -> None:
    import hashlib

    if CSV_PATH.is_file():
        digest = hashlib.md5(CSV_PATH.read_bytes()).hexdigest()
        if digest == CSV_MD5:
            print(f"csv ok {CSV_PATH}")
            return
    print("downloading Zenodo AntibodyTrainingDataset.csv")
    fetch(
        ZENODO_CSV,
        CSV_PATH,
        headers={
            "Accept": "text/csv,application/octet-stream,*/*",
            "Referer": "https://zenodo.org/records/15741710",
        },
    )


def load_rows() -> list[dict]:
    with CSV_PATH.open() as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict], pdb: str, heavy: str, light: str, antigen: str) -> dict:
    for row in rows:
        if (
            row["pdb"] == pdb
            and row["Hchain"] == heavy
            and row["Lchain"] == light
            and row["antigen_chain"] == antigen
        ):
            return row
    raise RuntimeError(f"{pdb} H={heavy} L={light or '-'} T={antigen} is not in the Zenodo table")


def pdb_path(pdb: str) -> Path:
    dest = RAW / f"{pdb.lower()}.pdb"
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    url = f"https://files.rcsb.org/download/{pdb.upper()}.pdb"
    print(f"downloading {url}")
    fetch(url, dest)
    return dest


def parse_pdb(path: Path) -> dict[str, list[dict]]:
    """First model, standard residues, grouped by chain in file order."""
    chains: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith("ENDMDL"):
            break
        if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
            continue
        alt = raw[16:17]
        if alt not in (" ", "A"):
            continue
        resn = raw[17:20].strip()
        if resn not in AA3:
            continue
        atom = raw[12:16].strip()
        if atom not in BACKBONE:
            continue
        chain = raw[21:22].strip() or "_"
        try:
            resseq = int(raw[22:26])
        except ValueError:
            continue
        icode = raw[26:27]
        try:
            xyz = (float(raw[30:38]), float(raw[38:46]), float(raw[46:54]))
        except ValueError:
            continue
        key = (chain, resseq, icode)
        if key not in seen:
            seen.add(key)
            chains.setdefault(chain, []).append(
                {
                    "resn": "MET" if resn == "MSE" else resn,
                    "aa": AA3[resn],
                    "resseq": resseq,
                    "icode": icode,
                    "atoms": {},
                }
            )
        residue = chains[chain][-1]
        if (residue["resseq"], residue["icode"]) != (resseq, icode):
            continue
        residue["atoms"][atom] = xyz
    for chain, residues in chains.items():
        chains[chain] = [r for r in residues if {"N", "CA", "C"} <= r["atoms"].keys()]
    return chains


def sequence(residues: list[dict]) -> str:
    return "".join(r["aa"] for r in residues)


def abnum(seq: str) -> list[tuple[int, str, str]]:
    """Chothia numbers for one variable-domain sequence: (num, insert, aa)."""
    cache = RAW / "abnum" / f"{seq}.txt"
    if cache.is_file():
        text = cache.read_text()
    else:
        payload = urllib.parse.urlencode(
            {"scheme": "-c", "plain": "1", "aaseq": seq}
        ).encode()
        time.sleep(0.4)
        blob = fetch(
            ABNUM_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        text = blob.decode("latin1", errors="replace")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    if "<html" in text.lower() and "H1 " not in text and "L1 " not in text:
        raise RuntimeError(f"Abnum failed: {text[:200]!r}")
    numbered = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[1] not in AA3.values():
            continue
        token, aa = parts
        if token[0] not in ("H", "L"):
            continue
        body = token[1:]
        digits = "".join(ch for ch in body if ch.isdigit())
        insert = "".join(ch for ch in body if ch.isalpha())
        if not digits:
            continue
        numbered.append((int(digits), insert, aa))
    if len(numbered) < 90:
        raise RuntimeError(f"Abnum returned only {len(numbered)} residues")
    return numbered


def take_fv(residues: list[dict], chain_type: str) -> tuple[list[dict], list[str]]:
    seq = sequence(residues)
    numbered = abnum(seq)
    num_seq = "".join(aa for _, _, aa in numbered)
    start = seq.find(num_seq)
    if start < 0:
        raise RuntimeError("Abnum sequence is not a contiguous substring of the chain")
    crop = FV_CROP[chain_type]
    kept = []
    labels = []
    for offset, (num, _insert, aa) in enumerate(numbered):
        if num > crop:
            break
        if residues[start + offset]["aa"] != aa:
            raise RuntimeError("Abnum alignment drifted")
        label = ""
        for name, (lo, hi) in CDR_RANGES.items():
            if name[0] == chain_type and lo <= num <= hi:
                label = name
                break
        kept.append(residues[start + offset])
        labels.append(label)
    if not any(labels):
        raise RuntimeError(f"no {chain_type} CDRs after Chothia crop")
    return kept, labels


def format_atom(serial: int, aname: str, resn: str, chain: str, resseq: int, xyz: tuple[float, float, float]) -> str:
    line = [" "] * 80
    def put(col: int, text: str) -> None:
        for i, ch in enumerate(text):
            line[col - 1 + i] = ch
    x, y, z = xyz
    put(1, "ATOM")
    put(7, f"{serial:5d}")
    put(13, f"{aname:>4s}")
    put(18, f"{resn:>3s}")
    put(22, chain)
    put(23, f"{resseq:4d}")
    put(31, f"{x:8.3f}")
    put(39, f"{y:8.3f}")
    put(47, f"{z:8.3f}")
    put(55, "  1.00")
    put(61, " 20.00")
    put(77, f"{aname.strip()[0]:>2s}")
    return "".join(line)


def write_hlt(path: Path, groups: list[tuple[str, list[dict], list[str]]]) -> None:
    """groups: (chain id H/L/T, residues, per-residue CDR label or '')."""
    lines = []
    serial = 1
    abs_index = 1
    # Residue numbers must be unique inside a chain id. Several biological
    # chains are all written as T, so the counter continues across those groups.
    resseq = {"H": 1, "L": 1, "T": 1}
    remarks = []
    for chain, residues, labels in groups:
        for residue, label in zip(residues, labels):
            number = resseq[chain]
            resseq[chain] = number + 1
            for aname in BACKBONE:
                if aname not in residue["atoms"]:
                    continue
                lines.append(
                    format_atom(serial, aname, residue["resn"], chain, number, residue["atoms"][aname])
                )
                serial += 1
            if label:
                remarks.append(f"REMARK PDBinfo-LABEL: {abs_index:4d} {label}")
            abs_index += 1
    lines.extend(remarks)
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def ca(residue: dict) -> tuple[float, float, float]:
    return residue["atoms"]["CA"]


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def interface_loop(binder: list[dict], target: list[dict], expect: str) -> list[str]:
    dists = [
        min(dist(ca(b), ca(t)) for t in target)
        for b in binder
    ]
    best = None
    for win in (8, 10, 12):
        if len(binder) < win:
            continue
        for i in range(len(binder) - win + 1):
            span = binder[i + win - 1]["resseq"] - binder[i]["resseq"]
            if span > win + 4:
                continue
            mean = sum(dists[i:i + win]) / win
            if best is None or mean < best[0]:
                best = (mean, i, win)
    if best is None or best[0] > 8.0:
        raise RuntimeError(f"no tight interface loop (best={best})")
    mean, i, win = best
    labels = [""] * len(binder)
    for j in range(i, i + win):
        labels[j] = "H3"
    loop_seq = sequence(binder)[i:i + win]
    print(f"  interface loop {loop_seq} meanCA={mean:.2f}A")
    if expect not in loop_seq:
        raise RuntimeError(f"loop {loop_seq} does not contain {expect}")
    return labels


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_hlt(path: Path, expected: int | None = None) -> None:
    from rfab_finetune.pdb import parse_hlt

    ex = parse_hlt(path)
    if expected is not None and ex.length != expected:
        raise RuntimeError(f"{path.name} length {ex.length} != {expected}")
    if ex.length < 30:
        raise RuntimeError(f"{path.name} is too short ({ex.length})")
    if not bool(ex.target_mask.any()):
        raise RuntimeError(f"{path.name} has no target chain")
    if int(ex.cdr_mask.sum()) < 5:
        raise RuntimeError(f"{path.name} has too few CDR residues")
    if "H" not in ex.chain:
        raise RuntimeError(f"{path.name} has no heavy/binder chain")
    missing = int(ex.xyz[:, 1].isnan().any(dim=-1).sum())
    if missing:
        raise RuntimeError(f"{path.name} has {missing} residues without CA")


def write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} -> {rel(path)}")


def build_sabdab(rows: list[dict]) -> list[dict]:
    out_dir = DATA / "examples" / "sabdab"
    counts = {k: 0 for k in SABDAB_QUOTA}
    manifest = []
    for bucket, pdb, heavy, light, antigen in SABDAB_CANDIDATES:
        if counts[bucket] >= SABDAB_QUOTA[bucket]:
            continue
        row = find_row(rows, pdb, heavy, light, antigen)
        try:
            chains = parse_pdb(pdb_path(pdb))
            groups = []
            fv, labels = take_fv(chains[heavy], "H")
            groups.append(("H", fv, labels))
            if light:
                fv, labels = take_fv(chains[light], "L")
                groups.append(("L", fv, labels))
            target = chains[antigen]
            groups.append(("T", target, [""] * len(target)))
            name = f"{pdb}_{heavy}_{light or 'nan'}_{antigen}.pdb"
            dest = out_dir / name
            write_hlt(dest, groups)
            check_hlt(dest, expected=sum(len(res) for _, res, _ in groups))
        except Exception as exc:
            print(f"skip {pdb} ({bucket}): {exc}")
            continue
        counts[bucket] += 1
        manifest.append(
            {
                "pdb": rel(dest),
                "cluster": row["ab_cluster"],
                "pdb_id": pdb,
                "bucket": bucket,
                "resolution": row["resolution"],
                "antigen_name": row["antigen_name"],
                "antigen_type": row["antigen_type"],
            }
        )
        print(
            f"sabdab {pdb} {bucket} res={row['resolution']} "
            f"{row['antigen_name'][:48]}"
        )
    missing = [k for k, n in SABDAB_QUOTA.items() if counts[k] < n]
    if missing:
        raise SystemExit(f"could not fill SAbDab quotas: {missing} got {counts}")
    return manifest


def build_tcr() -> list[dict]:
    out_dir = DATA / "examples" / "tcr"
    manifest = []
    for spec in TCR_EXAMPLES:
        chains = parse_pdb(pdb_path(spec["pdb"]))
        groups = []
        for chain_id, src in (("H", spec["heavy"]), ("L", spec["light"])):
            residues = chains[src]
            seq = sequence(residues)
            labels = [""] * len(residues)
            for cdr_name, motif in spec["cdrs"][chain_id].items():
                idx = seq.find(motif)
                if idx < 0:
                    raise RuntimeError(f"{spec['pdb']} {chain_id} missing {cdr_name} {motif}")
                for j in range(idx, idx + len(motif)):
                    if labels[j]:
                        raise RuntimeError(f"{spec['pdb']} overlapping CDRs at {j}")
                    labels[j] = cdr_name
            groups.append((chain_id, residues, labels))
        for src in spec["targets"]:
            residues = chains[src]
            groups.append(("T", residues, [""] * len(residues)))
        dest = out_dir / f"{spec['pdb']}.pdb"
        write_hlt(dest, groups)
        check_hlt(dest, expected=sum(len(res) for _, res, _ in groups))
        manifest.append(
            {
                "pdb": rel(dest),
                "cluster": spec["pdb"],
                "pdb_id": spec["pdb"],
                "description": spec["name"],
            }
        )
        print(f"tcr {spec['pdb']} {spec['name']}")
    return manifest


def build_loops() -> list[dict]:
    out_dir = DATA / "examples" / "pdb_loop"
    manifest = []
    for spec in LOOP_EXAMPLES:
        chains = parse_pdb(pdb_path(spec["pdb"]))
        binder = chains[spec["binder"]]
        target_groups = []
        all_target = []
        for src in spec["targets"]:
            target_groups.append(chains[src])
            all_target.extend(chains[src])
        labels = interface_loop(binder, all_target, spec["expect"])
        groups = [("H", binder, labels)]
        for residues in target_groups:
            groups.append(("T", residues, [""] * len(residues)))
        dest = out_dir / f"{spec['pdb']}.pdb"
        write_hlt(dest, groups)
        check_hlt(dest, expected=sum(len(res) for _, res, _ in groups))
        manifest.append(
            {
                "pdb": rel(dest),
                "cluster": spec["pdb"],
                "pdb_id": spec["pdb"],
                "description": spec["name"],
            }
        )
        print(f"pdb_loop {spec['pdb']} {spec['name']}")
    return manifest


def main() -> None:
    ensure_csv()
    rows = load_rows()
    sabdab = build_sabdab(rows)
    tcr = build_tcr()
    loops = build_loops()
    write_manifest(DATA / "sabdab.jsonl", sabdab)
    write_manifest(DATA / "tcr.jsonl", tcr)
    write_manifest(DATA / "pdb_loops.jsonl", loops)
    print(
        f"example set: {len(sabdab)} SAbDab, {len(tcr)} TCR, {len(loops)} PDB loops"
    )


if __name__ == "__main__":
    main()
