"""Fine-tune RFdiffusion for antibody design (Bennett et al. 2025 SI section 3.6)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rfab_finetune.constants import TrainConfig
from rfab_finetune.dataset import MixedAntibodyDataset, collate_singleton
from rfab_finetune.features import inject_self_cond
from rfab_finetune.losses import total_loss
from rfab_finetune.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def move_feats(feats: dict, device: torch.device) -> dict:
    out = {}
    for key, val in feats.items():
        if torch.is_tensor(val):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def _checkpoint(model, opt, epoch: int, step: int, cfg: TrainConfig) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": cfg.__dict__,
    }


def latest_checkpoint(cfg: TrainConfig) -> Path | None:
    """Final checkpoint if it is at least as new as the last epoch file, else that epoch file."""
    out = Path(cfg.ckpt_out)
    epoch_files = []
    if out.parent.is_dir():
        prefix = f"{out.stem}_e"
        for path in out.parent.glob(f"{prefix}*.pt"):
            suffix = path.stem[len(prefix):]
            if suffix.isdigit():
                epoch_files.append((int(suffix), path))
    epoch_files.sort()
    final = out if out.is_file() else None
    if final is not None and epoch_files:
        last = epoch_files[-1][1]
        if final.stat().st_mtime >= last.stat().st_mtime - 1:
            return final
        return last
    if final is not None:
        return final
    if epoch_files:
        return epoch_files[-1][1]
    return None


def _optimizer_step(model, opt, pending: int, global_step: int, cfg: TrainConfig) -> tuple[int, float]:
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(pending)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    lr = lr_at_step(global_step, cfg)
    for group in opt.param_groups:
        group["lr"] = lr
    opt.step()
    opt.zero_grad(set_to_none=True)
    return global_step + 1, lr


def train(cfg: TrainConfig, seed: int = 0, resume: bool = False) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} dummy={cfg.dummy}")

    dataset = MixedAntibodyDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_singleton,
    )
    model = build_model(cfg, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir = Path(cfg.ckpt_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.log_dir) / "train.jsonl"

    global_step = 0
    start_epoch = 0
    if resume:
        resume_path = latest_checkpoint(cfg)
        if resume_path is None:
            raise FileNotFoundError(f"--resume found no checkpoint next to {cfg.ckpt_out}")
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        if "epoch" not in blob or "model_state_dict" not in blob:
            raise RuntimeError(f"{resume_path} has no epoch/model_state_dict; cannot resume")
        model.load_state_dict(blob["model_state_dict"])
        if "optimizer_state_dict" in blob:
            opt.load_state_dict(blob["optimizer_state_dict"])
        start_epoch = int(blob["epoch"]) + 1
        global_step = int(blob.get("step", 0))
        print(f"resumed {resume_path} at epoch {start_epoch} step {global_step}")
        if start_epoch >= cfg.epochs:
            print(f"checkpoint already reached epoch {blob['epoch']}; nothing to do")
            return

    accum = cfg.pseudo_batch_size
    opt.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, cfg.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        pending = 0
        n_batches = len(loader)
        for i, feats in enumerate(loader):
            feats = move_feats(feats, device)
            t = int(feats["t"])
            if random.random() < cfg.self_cond_prob:
                with torch.no_grad():
                    cond = model(feats)
                feats = inject_self_cond(feats, cond["xyz"])
            pred = model(feats)
            loss, stats = total_loss(pred, feats, t, cfg)
            loss.backward()
            pending += 1
            epoch_loss += stats["loss"]

            if pending == accum or i + 1 == n_batches:
                global_step, lr = _optimizer_step(model, opt, pending, global_step, cfg)
                stats = {**stats, "epoch": epoch, "step": global_step, "lr": lr, "accum": pending}
                pending = 0
                with log_path.open("a") as handle:
                    handle.write(json.dumps(stats) + "\n")
                if global_step % 10 == 0:
                    print(
                        f"epoch {epoch} step {global_step} t={t} "
                        f"loss={stats['loss']:.4f} frame={stats['l_frame']:.4f} "
                        f"2d={stats['l_2d']:.4f} mlm={stats['l_mlm']:.4f} lr={lr:.2e}"
                    )

        ckpt_path = Path(cfg.ckpt_out).with_name(f"{Path(cfg.ckpt_out).stem}_e{epoch}.pt")
        torch.save(_checkpoint(model, opt, epoch, global_step, cfg), ckpt_path)
        print(
            f"epoch {epoch} done in {time.time() - t0:.1f}s "
            f"mean_loss={epoch_loss / max(n_batches, 1):.4f} saved {ckpt_path}"
        )

    torch.save(_checkpoint(model, opt, cfg.epochs - 1, global_step, cfg), cfg.ckpt_out)
    print(f"wrote {cfg.ckpt_out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sabdab-manifest", default="")
    p.add_argument("--tcr-manifest", default="")
    p.add_argument("--pdb-loop-manifest", default="")
    p.add_argument("--ckpt-in", default="", help="Vanilla RFdiffusion or previous Ab checkpoint.")
    p.add_argument("--ckpt-out", default="checkpoints/rfdiffusion_ab.pt")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--examples-per-epoch", type=int, default=512)
    p.add_argument("--crop-size", type=int, default=768)
    p.add_argument("--pseudo-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dummy", action="store_true", help="Use DummyDiffusionModel (no RFantibody).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint.")
    args = p.parse_args()
    cfg = TrainConfig(
        sabdab_manifest=args.sabdab_manifest,
        tcr_manifest=args.tcr_manifest,
        pdb_loop_manifest=args.pdb_loop_manifest,
        ckpt_in=args.ckpt_in,
        ckpt_out=args.ckpt_out,
        epochs=args.epochs,
        examples_per_epoch=args.examples_per_epoch,
        crop_size=args.crop_size,
        pseudo_batch_size=args.pseudo_batch_size,
        lr=args.lr,
        dummy=args.dummy,
    )
    return cfg, args.seed, args.resume


def main() -> None:
    cfg, seed, resume = parse_args()
    train(cfg, seed=seed, resume=resume)


if __name__ == "__main__":
    main()
