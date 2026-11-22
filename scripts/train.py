#!/usr/bin/env python3
"""Train a reference-annotation splice-only CNN-Mamba-k7 model.

This is the architecture-matched replacement for the historical pure-Mamba
Figure-1 checkpoints.  Existing 10 kb NPZ windows are reused and selected by
chromosome; no windows or labels are regenerated.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba3
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIES_CONFIG = REPO_ROOT / "configs" / "species.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--species", required=True)
    p.add_argument("--species-config", type=Path, default=DEFAULT_SPECIES_CONFIG)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cnn-kernel-size", type=int, default=7)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke-batches", type=int, default=0)
    return p.parse_args()


def chrom_from_name(path: Path) -> str:
    match = re.match(r"(.+?)_(?:plus|minus)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse chromosome from {path.name}")
    chrom = match.group(1)
    return chrom if chrom.startswith("chr") else "chr" + chrom


def discover_files(root: Path):
    """Discover all existing windows, verifying duplicate names are identical."""
    by_chrom: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob("*/*.npz")):
        chrom = chrom_from_name(path)
        records = by_chrom.setdefault(chrom, {})
        previous = records.get(path.name)
        if previous is not None:
            if previous.stat().st_size != path.stat().st_size or not filecmp.cmp(previous, path, shallow=False):
                raise ValueError(f"Conflicting duplicate windows: {previous} and {path}")
            continue
        records[path.name] = path.resolve()
    return {chrom: sorted(records.values()) for chrom, records in by_chrom.items()}


def select_split(by_chrom, chromosomes, split):
    missing = [chrom for chrom in chromosomes if chrom not in by_chrom]
    if missing:
        raise ValueError(f"Missing {split} chromosomes in existing windows: {missing}")
    return [path for chrom in chromosomes for path in by_chrom[chrom]]


class NPZDataset(Dataset):
    def __init__(self, files):
        self.files = [str(x) for x in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            sequence = torch.from_numpy(data["sequence"].astype(np.int64, copy=False))
            labels = torch.from_numpy(data["labels"].astype(np.int64, copy=False))
        return sequence, labels


class BiMamba3Block(nn.Module):
    def __init__(self, d_model, d_state, headdim):
        super().__init__()
        self.fw = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim)
        self.bw = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim)

    def forward(self, x):
        return self.fw(x) + self.bw(x.flip(1)).flip(1)


class CNNBiMamba3Block(nn.Module):
    def __init__(self, d_model, d_state, dropout, kernel_size):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("CNN kernel size must be positive and odd")
        self.cnn_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.cnn_drop = nn.Dropout(dropout)
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba = BiMamba3Block(d_model, d_state, min(64, d_model))
        self.mamba_drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.cnn_norm(x).transpose(1, 2)
        y = self.pointwise(F.gelu(self.depthwise(y))).transpose(1, 2)
        x = x + self.cnn_drop(y)
        return x + self.mamba_drop(self.mamba(self.mamba_norm(x)))


class SpliceCNNMambaK7(nn.Module):
    def __init__(self, d_model, d_state, n_layers, dropout, kernel_size):
        super().__init__()
        self.embedding = nn.Embedding(5, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.embed_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            CNNBiMamba3Block(d_model, d_state, dropout, kernel_size)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3)
        )

    def forward(self, sequence):
        x = self.embed_drop(self.input_proj(self.embedding(sequence)))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x))


def candidate_masks(sequence):
    gt = (sequence[:, :-1] == 2) & (sequence[:, 1:] == 3)
    ag = (sequence[:, :-1] == 0) & (sequence[:, 1:] == 2)
    gt = F.pad(gt, (0, 1), value=False)
    ag = F.pad(ag, (0, 1), value=False)
    return gt, ag


def masked_loss(logits, labels, sequence):
    gt, ag = candidate_masks(sequence)
    mask = gt | ag | (labels == 1) | (labels == 2)
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    batches = 0
    score_parts = {"donor": [], "acceptor": []}
    truth_parts = {"donor": [], "acceptor": []}
    for sequence, labels in loader:
        sequence = sequence.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(sequence)
        loss_sum += float(masked_loss(logits, labels, sequence).item())
        batches += 1
        probs = torch.softmax(logits.float(), dim=-1)
        gt, ag = candidate_masks(sequence)
        if gt.any():
            score_parts["donor"].append(probs[..., 1][gt].cpu().numpy())
            truth_parts["donor"].append((labels[gt] == 1).cpu().numpy())
        if ag.any():
            score_parts["acceptor"].append(probs[..., 2][ag].cpu().numpy())
            truth_parts["acceptor"].append((labels[ag] == 2).cpu().numpy())
    result = {"loss": loss_sum / max(batches, 1)}
    for role in ("donor", "acceptor"):
        scores = np.concatenate(score_parts[role])
        truth = np.concatenate(truth_parts[role])
        result[role] = {
            "AP": float(average_precision_score(truth, scores)),
            "candidates": int(len(truth)),
            "positives": int(truth.sum()),
        }
    result["mean_AP"] = 0.5 * (result["donor"]["AP"] + result["acceptor"]["AP"])
    return result


def optimizer_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "embedding" in name or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_lr(step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def atomic_torch_save(payload, path: Path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def main():
    args = parse_args()
    species_config = args.species_config.resolve()
    with species_config.open() as handle:
        all_species = json.load(handle)
    if args.species not in all_species:
        raise ValueError(
            f"Unknown species {args.species!r}; choices={sorted(all_species)}"
        )
    cfg = all_species[args.species]
    configured_root = Path(cfg["windows"])
    root = args.data_root or (
        configured_root if configured_root.is_absolute()
        else REPO_ROOT / configured_root
    )
    root = root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "best_model.pt").exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    by_chrom = discover_files(root)
    files = {
        split: select_split(by_chrom, cfg[split], split)
        for split in ("train", "val", "test")
    }
    manifest = {
        "species": args.species,
        "reference": cfg["reference"],
        "data_root": str(root.resolve()),
        "selection": "existing NPZ windows selected by chromosome; no regeneration",
        "window_size": 10_000,
        "stride": 5_000,
        "splits": {
            split: {"chromosomes": cfg[split], "window_count": len(files[split])}
            for split in files
        },
    }
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        NPZDataset(files["train"]), batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        NPZDataset(files["val"]), batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None,
    )

    model = SpliceCNNMambaK7(
        args.d_model, args.d_state, args.n_layers, args.dropout, args.cnn_kernel_size
    ).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95)
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * updates_per_epoch
    warmup = max(20, min(200, total_steps // 10))
    start_epoch = 1
    opt_step = 0
    history = []
    best_mean_ap = -1.0
    last_path = args.output_dir / "last_model.pt"
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        opt_step = int(saved["opt_step"])
        history = saved["history"]
        best_mean_ap = float(saved["best_validation_mean_AP"])
        print(f"Resuming at epoch {start_epoch}", flush=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(args.output_dir.resolve())
    run_config["species_config"] = str(species_config)
    run_config["data_root"] = str(root)
    run_config.update({
        "architecture": "CNN-Mamba-k7", "window_size": 10_000, "stride": 5_000,
        "precision": "FP32", "objective": "canonical-candidate-masked 3-class CE",
        "selection_metric": "validation donor/acceptor mean AP",
    })
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        batches = 0
        t0 = time.time()
        for batch_index, (sequence, labels) in enumerate(train_loader, start=1):
            sequence = sequence.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lr = get_lr(opt_step, total_steps, args.lr, warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = masked_loss(model(sequence), labels, sequence)
            (loss / args.grad_accum).backward()
            loss_sum += float(loss.item())
            batches += 1
            if batch_index % args.grad_accum == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
            if batch_index % 500 == 0:
                print(
                    f"epoch={epoch} batch={batch_index:,}/{len(train_loader):,} "
                    f"loss={loss_sum / batches:.6f} lr={lr:.3g}", flush=True,
                )
            if args.smoke_batches and batch_index >= args.smoke_batches:
                print("Smoke-batch limit reached; exiting without checkpoint.", flush=True)
                return

        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "validation": metrics,
            "elapsed_seconds": time.time() - t0,
            "lr_end": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        checkpoint = {
            "epoch": epoch,
            "opt_step": opt_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_validation_mean_AP": max(best_mean_ap, metrics["mean_AP"]),
            "config": run_config,
            "split_manifest": manifest,
        }
        atomic_torch_save(checkpoint, last_path)
        if metrics["mean_AP"] > best_mean_ap:
            best_mean_ap = metrics["mean_AP"]
            checkpoint["best_validation_mean_AP"] = best_mean_ap
            atomic_torch_save(checkpoint, args.output_dir / "best_model.pt")
            print(f"Saved new best: mean AP={best_mean_ap:.6f}", flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
