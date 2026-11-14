#!/usr/bin/env python3
"""Scan a directory of HLT PDBs and write a jsonl manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_dir")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--cluster-from-stem", action="store_true")
    args = parser.parse_args()
    pdbs = sorted(Path(args.pdb_dir).rglob("*.pdb"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for pdb in pdbs:
            cluster = pdb.stem.split("_")[0] if args.cluster_from_stem else pdb.stem
            handle.write(json.dumps({"pdb": str(pdb.resolve()), "cluster": cluster}) + "\n")
    print(f"wrote {len(pdbs)} entries to {args.output}")


if __name__ == "__main__":
    main()
