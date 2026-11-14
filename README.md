# Fine-tune RFdiffusion for antibody design

Training code for the antibody RFdiffusion recipe in Bennett et al., *Nature* 2025 (Supplementary Methods section 3). Official training scripts were never released; this package reconstructs them from the paper and from the [RFantibody](https://github.com/RosettaCommons/RFantibody) inference model.

It does **not** reimplement RoseTTAFold. For a real run, install RFantibody and start from a vanilla RFdiffusion checkpoint. `--dummy` trains a tiny stand-in so you can check the data and loss pipeline on a laptop.

## What is implemented

| Piece | Source |
|---|---|
| Data mix 50% SAbDab / 10% TCR / 40% PDB loop interfaces | SI section 3.1 |
| Noise only the binder; keep the target backbone+sequence, no sidechains | SI section 3.1 |
| Framework as t2d distances/dihedrals, **no** antibody-antigen dock | SI section 3.2, Alg. 2 |
| Hotspot = mean C? to closest 5 CDR residues < 8 �; show 0-100% | SI section 3.5 |
| Losses: frame MSE + distogram + masked sequence CE, weights 1.0 | SI Table 1 |
| Crop 768, pseudo-batch 16, 70 � 512, lr 5e-4 with 100-step warmup | SI section 3.6 |

## Install

```bash
conda env create -f environment.yml
conda activate rf-antibody
# real model:
# git clone https://github.com/RosettaCommons/RFantibody.git && cd RFantibody && uv sync
```

## Prepare data

A small real example set is already in `data/`. It is not the full training corpus.

| File | What it is |
|---|---|
| `data/AntibodyTrainingDataset.csv` | Paper training table, Zenodo [15741710](https://zenodo.org/records/15741710) (CC-BY 4.0) |
| `data/examples/sabdab/` | Six complexes taken from that table. Coordinates are the RCSB entries; CDRs are Chothia ranges from Abnum. |
| `data/examples/tcr/` | 1AO7 (A6 / Tax / HLA-A*02:01) and 2CKB (2C / dEV8 / H-2Kb). Beta is H, alpha is L, pMHC is T. CDR sequences are from STCRDab. |
| `data/examples/pdb_loop/` | 2PTC (BPTI / trypsin) and 1CHO (ovomucoid / chymotrypsin). The inhibitor loop is H3. |
| `data/sabdab.jsonl`, `data/tcr.jsonl`, `data/pdb_loops.jsonl` | Manifests for the files above. |

SAbDab examples: 2P45 (VHH–RNase, 1.1 Å), 6JB2 (VHH–lysozyme, 1.5 Å), 6DDM (Fab–MICA, 1.3 Å), 5KVG (Fab–Zika DIII, 1.4 Å), 6UCE (Fab–HIV fusion peptide, 1.38 Å), 6S0Y (VHH–M2 peptide, 1.81 Å).

Rebuild or refresh the download with:

```bash
python scripts/fetch_example_data.py
```

For a full training run, download every complex listed in the Zenodo CSV, convert each one to RFantibody **HLT** format (`chothia_to_HLT.py`: chain H, optional L, target as T, CDR remarks at the end), and write jsonl manifests:

```bash
python scripts/prepare_manifest.py /data/sabdab_hlt -o data/sabdab.jsonl
python scripts/prepare_manifest.py /data/tcr_hlt -o data/tcr.jsonl
python scripts/prepare_manifest.py /data/pdb_loops_hlt -o data/pdb_loops.jsonl
```

Each line is `{"pdb": "/abs/path.pdb", "cluster": "cluster_id"}`.

## Dry-run (no GPU, dummy network)

Toy complex, no structures required:

```bash
python -m rfab_finetune.train --dummy --epochs 1 --examples-per-epoch 16 --pseudo-batch-size 4
```

The real example manifests (run from the repository root):

```bash
python -m rfab_finetune.train --dummy --epochs 1 --examples-per-epoch 8 --pseudo-batch-size 4 \
    --sabdab-manifest data/sabdab.jsonl \
    --tcr-manifest data/tcr.jsonl \
    --pdb-loop-manifest data/pdb_loops.jsonl
```

## Fine-tune the real network

```bash
python -m rfab_finetune.train \
    --ckpt-in /path/to/vanilla_RFdiffusion.pt \
    --sabdab-manifest data/sabdab.jsonl \
    --tcr-manifest data/tcr.jsonl \
    --pdb-loop-manifest data/pdb_loops.jsonl \
    --ckpt-out checkpoints/rfdiffusion_ab.pt
```

Paper hardware: ~3 days on 2� 80 GB A100, crop 768. 80 GB is required at that crop; drop `--crop-size` if you have less memory.

After training, point RFantibody inference at your checkpoint:

```text
inference.ckpt_override_path=checkpoints/rfdiffusion_ab.pt
```

The released RFantibody weights are already this fine-tune; re-run this code only if you want to change the data mix, hotspot rule, or starting checkpoint.

## Tests

```bash
pytest -q
```
