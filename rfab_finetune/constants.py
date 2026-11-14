"""Hyperparameters from Bennett et al., Nature 2025 Supplementary Methods section 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

AA_TO_IDX: Dict[str, int] = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
}
IDX_TO_AA = {v: k for k, v in AA_TO_IDX.items()}
UNK_IDX = 20
MASK_IDX = 21
N_AA = 22

# Chothia CDR residue numbers (used if HLT remarks are missing).
CHOTHIA_CDR: Dict[str, Tuple[int, int]] = {
    "H1": (26, 32),
    "H2": (52, 56),
    "H3": (95, 102),
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}

# Distogram bins matching RF kinematics.
DMIN = 2.0
DMAX = 20.0
DBINS = 36  # plus 1 overflow bin -> 37 classes


@dataclass
class TrainConfig:
    """SI section 3.6 plus the antibody-specific noising / conditioning recipe."""

    # Data mixture (SI section 3.1)
    sabdab_prob: float = 0.50
    tcr_prob: float = 0.10
    pdb_loop_prob: float = 0.40

    # Optimization (SI section 3.6)
    crop_size: int = 768
    pseudo_batch_size: int = 16
    epochs: int = 70
    examples_per_epoch: int = 512
    lr: float = 5e-4
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Diffusion (vanilla RFdiffusion defaults; SI: remaining settings unchanged)
    T: int = 50
    b0: float = 1e-2
    bT: float = 7e-2
    crd_scale: float = 0.25
    # Linear IGSO(3) schedule from RFdiffusion (sigma(t) for t in [0, 1]).
    # max_sigma in the public config is unused: the linear schedule sets it to sigma(1).
    so3_min_sigma: float = 0.02
    so3_max_sigma: float = 1.5
    so3_min_b: float = 1.5
    so3_max_b: float = 2.5

    # Conditioning
    hotspot_cutoff: float = 8.0
    hotspot_k: int = 5
    self_cond_prob: float = 0.50
    mask_target_sidechains: bool = True
    mask_loop_sequence: bool = True

    # Losses (SI Table 1)
    w_frame: float = 1.0
    w_2d: float = 1.0
    w_mlm: float = 1.0
    mlm_sigmoid_center: float = 50.0
    mlm_sigmoid_temp: float = 5.0

    # Model
    d_t1d: int = 23
    # 37 distance bins + 6 orientations + pair mask. Matches Templ_emb.
    d_t2d: int = 44
    dummy: bool = False
    use_checkpoint: bool = True

    # Paths
    sabdab_manifest: str = ""
    tcr_manifest: str = ""
    pdb_loop_manifest: str = ""
    ckpt_in: str = ""
    ckpt_out: str = "checkpoints/rfdiffusion_ab.pt"
    log_dir: str = "runs/rfab_ft"

    extra: Dict[str, float] = field(default_factory=dict)

    def steps_per_epoch(self) -> int:
        return self.examples_per_epoch

    def total_steps(self) -> int:
        return self.epochs * self.examples_per_epoch
