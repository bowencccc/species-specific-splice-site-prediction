"""
Score canonical splice-site candidates with a trained CNN--Mamba checkpoint.

By default this scans GT/AG on the + strand and GT/AG on the reverse
complement for the - strand. The latter is equivalent to AC/CT on the
forward genome sequence. Output is streamed to disk to avoid keeping tens of
millions of rows in memory.
"""

import argparse
import gzip
import os
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba3
from pyfaidx import Fasta


DEFAULT_CHROMS = ["NC_000001.11", "NC_000009.12"]
BASE2ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
RC_TO_GENOME_MOTIF = {"GT": "AC", "AG": "CT"}


class BiMamba3Block(nn.Module):
    def __init__(self, d_model, d_state, headdim):
        super().__init__()
        self.fw = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim)
        self.bw = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim)

    def forward(self, x):
        y_fw = self.fw(x)
        y_bw = self.bw(x.flip(1)).flip(1)
        return y_fw + y_bw


class CNNBiMamba3Block(nn.Module):
    """CNN-Mamba-k7 block used by the reference-trained Figure-1 models."""

    def __init__(self, d_model, d_state, headdim, dropout, kernel_size=7):
        super().__init__()
        self.cnn_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.cnn_drop = nn.Dropout(dropout)
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba = BiMamba3Block(d_model, d_state, headdim)
        self.mamba_drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.cnn_norm(x).transpose(1, 2)
        y = self.pointwise(F.gelu(self.depthwise(y))).transpose(1, 2)
        x = x + self.cnn_drop(y)
        return x + self.mamba_drop(self.mamba(self.mamba_norm(x)))


class SpliceMamba(nn.Module):
    def __init__(
        self,
        d_model=128,
        d_state=64,
        n_layers=4,
        n_classes=3,
        vocab_size=5,
        dropout=0.1,
        feat_dim=2,
        bidirectional=True,
        use_candidate_feats=True,
        architecture="mamba",
        cnn_kernel_size=7,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.use_candidate_feats = use_candidate_feats
        self.architecture = architecture

        self.embedding = nn.Embedding(vocab_size, d_model)

        if use_candidate_feats:
            self.feat_proj = nn.Sequential(
                nn.Linear(feat_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.input_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.embed_drop = nn.Dropout(dropout)
        if architecture == "cnn_mamba":
            self.layers = nn.ModuleList(
                [
                    CNNBiMamba3Block(
                        d_model=d_model,
                        d_state=d_state,
                        headdim=min(64, d_model),
                        dropout=dropout,
                        kernel_size=cnn_kernel_size,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            if bidirectional:
                self.layers = nn.ModuleList(
                    [
                        BiMamba3Block(
                            d_model=d_model,
                            d_state=d_state,
                            headdim=min(64, d_model),
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
                self.layers = nn.ModuleList(
                    [
                        Mamba3(
                            d_model=d_model,
                            d_state=d_state,
                            headdim=min(64, d_model),
                        )
                        for _ in range(n_layers)
                    ]
                )
            self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, input_ids, candidate_feats):
        x_seq = self.embedding(input_ids)

        if self.use_candidate_feats:
            x_feat = self.feat_proj(candidate_feats.float())
            x = torch.cat([x_seq, x_feat], dim=-1)
            x = self.input_proj(x)
        else:
            x = self.input_proj(x_seq)

        x = self.embed_drop(x)

        if self.architecture == "cnn_mamba":
            for layer in self.layers:
                x = layer(x)
        else:
            for norm, layer, drop in zip(self.norms, self.layers, self.drops):
                residual = x
                x_norm = norm(x)
                x_mamba = layer(x_norm)
                x = residual + drop(x_mamba)

        x = self.final_norm(x)
        return self.head(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream canonical splice-site scores for selected chromosomes."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Genome FASTA.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output TSV path. Use .gz suffix for gzip output.",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        required=True,
        help="Chromosome names as they appear in the FASTA.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Inference window size. Defaults to checkpoint config window_size or 10000.",
    )
    parser.add_argument("--stride", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--plus-only",
        action="store_true",
        help="Only score the + strand.",
    )
    parser.add_argument(
        "--minus-only",
        action="store_true",
        help="Only score the - strand.",
    )
    parser.add_argument(
        "--minus-motif-label",
        choices=["rc", "genome"],
        default="rc",
        help=(
            "For - strand rows, write motif as strand-oriented RC GT/AG "
            "or as forward-genome AC/CT."
        ),
    )
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Debug/sanity mode: only score the first N windows per chromosome/strand.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--fsync",
        action="store_true",
        help="fsync after each chromosome/strand. Safer but slower.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA float16 autocast for inference. Faster, but may produce "
            "NaNs for this Mamba3 checkpoint on N-heavy genome windows."
        ),
    )
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_seq(seq):
    seq = seq.upper()
    return np.fromiter((BASE2ID.get(b, 4) for b in seq), dtype=np.int64, count=len(seq))


def reverse_complement(seq):
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(seq))


def fetch_window_with_padding(fa, chrom, start, window):
    chrom_len = len(fa[chrom])
    end = start + window

    left_pad = max(0, -start)
    right_pad = max(0, end - chrom_len)

    real_start = max(0, start)
    real_end = min(chrom_len, end)

    seq = str(fa[chrom][real_start:real_end]).upper()
    if left_pad:
        seq = "N" * left_pad + seq
    if right_pad:
        seq = seq + "N" * right_pad

    assert len(seq) == window
    return seq


def get_keep_bounds(win_idx, n_windows, window, stride):
    overlap = window - stride
    half_overlap = overlap // 2

    if n_windows == 1:
        return 0, window
    if win_idx == 0:
        return 0, window - half_overlap
    if win_idx == n_windows - 1:
        return half_overlap, window
    return half_overlap, window - half_overlap


def get_oriented_keep_bounds(win_idx, n_windows, window, stride, rc):
    """Return motif-start bounds in the sequence orientation being scored.

    ``get_keep_bounds`` partitions forward-genome motif start coordinates.
    Reverse-complement inference reverses those coordinates, so its half-open
    bounds must also be reversed.  The ``window - 1`` terms account for a
    two-base motif whose genomic start maps from oriented index ``i`` as
    ``window - 2 - i``.
    """
    keep_l, keep_r = get_keep_bounds(win_idx, n_windows, window, stride)
    if not rc:
        return keep_l, keep_r
    return max(0, window - 1 - keep_r), min(
        window - 1, window - 1 - keep_l
    )


def batched(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def open_output(path):
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def build_model_from_checkpoint(ckpt, device):
    cfg = ckpt.get("config", {})
    state = ckpt["model_state_dict"]

    # Some early checkpoints contain stale architecture values in ``config``
    # (notably checkpoints_dmel_no_feats: 128/64/4 in config, while the saved
    # tensors are the actual 256/32/8 model).  Tensor shapes are authoritative:
    # infer the architecture and warn when the metadata disagrees.
    d_model = int(state["embedding.weight"].shape[1])
    layer_ids = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    n_layers = max(layer_ids) + 1
    b_keys = [key for key in state if key.startswith("layers.0.") and key.endswith("B_bias")]
    if not b_keys:
        raise KeyError("Cannot infer d_state: no layer-0 B_bias tensor")
    b_key = b_keys[0]
    d_state = int(state[b_key].shape[-1])
    architecture = "cnn_mamba" if any(".depthwise." in key for key in state) else "mamba"
    bidirectional = any(
        key.startswith("layers.0.bw.") or key.startswith("layers.0.mamba.bw.")
        for key in state
    )
    use_candidate_feats = any(key.startswith("feat_proj.") for key in state)
    cnn_kernel_size = (
        int(state["layers.0.depthwise.weight"].shape[-1])
        if architecture == "cnn_mamba"
        else 7
    )

    inferred = {
        "d_model": d_model,
        "d_state": d_state,
        "n_layers": n_layers,
        "bidirectional": bidirectional,
        "use_candidate_feats": use_candidate_feats,
        "architecture": architecture,
        "cnn_kernel_size": cnn_kernel_size,
    }
    configured = dict(cfg)
    if configured.get("architecture") == "CNN-Mamba-k7":
        configured["architecture"] = "cnn_mamba"
    disagreements = {
        key: {"config": cfg.get(key), "weights": value}
        for key, value in inferred.items()
        if key in configured and configured.get(key) != value
    }
    if disagreements:
        print(
            "WARNING: checkpoint config disagrees with tensor shapes; "
            f"using tensor-derived architecture: {disagreements}"
        )

    model = SpliceMamba(
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        n_classes=3,
        vocab_size=5,
        dropout=0.1,
        feat_dim=2,
        bidirectional=bidirectional,
        use_candidate_feats=use_candidate_feats,
        architecture=architecture,
        cnn_kernel_size=cnn_kernel_size,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def maybe_fsync(fout, enabled):
    fout.flush()
    if enabled and not isinstance(fout, gzip.GzipFile):
        os.fsync(fout.fileno())


def score_chrom_strand(
    *,
    fa,
    model,
    chrom,
    rc,
    fout,
    device,
    window,
    stride,
    batch_size,
    minus_motif_label,
    limit_windows,
    progress_every,
    amp,
):
    chrom_len = len(fa[chrom])
    starts = list(range(0, chrom_len, stride))
    if limit_windows is not None:
        starts = starts[:limit_windows]
    n_windows = len(starts)
    strand = "-" if rc else "+"
    counts = Counter()
    n_rows = 0
    t0 = time.time()

    print(
        f"  {chrom} [{strand}]: length={chrom_len:,}, windows={n_windows:,}, "
        f"batch_size={batch_size}"
    )

    for batch_idx, batch_starts in enumerate(batched(starts, batch_size)):
        seq_strs = []
        seq_ids = []

        for start in batch_starts:
            seq = fetch_window_with_padding(fa, chrom, start, window)
            if rc:
                seq = reverse_complement(seq)
            seq_strs.append(seq)
            seq_ids.append(encode_seq(seq))

        seq_batch = torch.tensor(np.stack(seq_ids), dtype=torch.long, device=device)
        feat_batch = torch.zeros(
            (seq_batch.shape[0], seq_batch.shape[1], 2),
            dtype=torch.float32,
            device=device,
        )

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), amp_context:
            logits = model(seq_batch, feat_batch)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        lines = []
        for b, start in enumerate(batch_starts):
            win_idx = start // stride
            keep_l, keep_r = get_oriented_keep_bounds(
                win_idx, n_windows, window, stride, rc
            )
            donor_prob = probs[b, :, 1]
            accept_prob = probs[b, :, 2]
            seq = seq_strs[b]
            max_i = min(keep_r - 1, window - 2)

            for i in range(keep_l, max_i + 1):
                oriented_motif = seq[i : i + 2]
                if oriented_motif not in ("GT", "AG"):
                    continue

                if rc:
                    gpos0 = start + window - 1 - (i + 1)
                else:
                    gpos0 = start + i

                if gpos0 < 0 or gpos0 + 1 >= chrom_len:
                    continue

                donor = float(donor_prob[i])
                acceptor = float(accept_prob[i])
                score = donor if oriented_motif == "GT" else acceptor
                out_motif = oriented_motif
                if rc and minus_motif_label == "genome":
                    out_motif = RC_TO_GENOME_MOTIF[oriented_motif]

                lines.append(
                    f"{chrom}\t{gpos0}\t{gpos0 + 1}\t{strand}\t{out_motif}\t"
                    f"{score:.6f}\t{donor:.6f}\t{acceptor:.6f}\n"
                )
                n_rows += 1
                counts[(strand, out_motif)] += 1

        fout.write("".join(lines))

        if progress_every and batch_idx % progress_every == 0:
            done_windows = min((batch_idx + 1) * batch_size, n_windows)
            elapsed = max(time.time() - t0, 1e-6)
            windows_per_s = done_windows / elapsed
            eta_s = (n_windows - done_windows) / max(windows_per_s, 1e-6)
            print(
                f"    progress: {done_windows:,}/{n_windows:,} windows, "
                f"{n_rows:,} rows, {windows_per_s:.1f} win/s, ETA {eta_s / 60:.1f} min"
            )

    return n_rows, counts


def main():
    args = parse_args()
    if args.plus_only and args.minus_only:
        raise ValueError("--plus-only and --minus-only are mutually exclusive")
    device = choose_device(args.device)
    if device.type != "cuda":
        print(
            "WARNING: CUDA is not available. This Mamba3 build uses CUDA/Triton "
            "ops for forward passes, so real scoring should be run on a GPU node."
        )

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model, cfg = build_model_from_checkpoint(ckpt, device)
    window = args.window or cfg.get("window_size", 10000)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Best cand_f1: {ckpt.get('best_cand_f1', 'N/A')}")
    print(f"Config: {cfg}")
    print(f"Window/stride: {window}/{args.stride}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output: {args.out}")

    fa = Fasta(args.fasta, as_raw=True, sequence_always_upper=True)
    missing = [chrom for chrom in args.chroms if chrom not in fa.keys()]
    if missing:
        raise ValueError(f"Chromosome(s) not found in FASTA: {missing}")

    total_rows = 0
    total_counts = Counter()
    t0 = time.time()

    with open_output(args.out) as fout:
        fout.write("chrom\tpos0\tpos1\tstrand\tmotif\tscore\tdonor_prob\tacceptor_prob\n")

        for chrom in args.chroms:
            print(f"\n{chrom}:")
            if not args.minus_only:
                n_rows, counts = score_chrom_strand(
                    fa=fa,
                    model=model,
                    chrom=chrom,
                    rc=False,
                    fout=fout,
                    device=device,
                    window=window,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    minus_motif_label=args.minus_motif_label,
                    limit_windows=args.limit_windows,
                    progress_every=args.progress_every,
                    amp=args.amp,
                )
                total_rows += n_rows
                total_counts.update(counts)
                print(f"    + strand done: {n_rows:,} rows")
                maybe_fsync(fout, args.fsync)

            if not args.plus_only:
                n_rows, counts = score_chrom_strand(
                    fa=fa,
                    model=model,
                    chrom=chrom,
                    rc=True,
                    fout=fout,
                    device=device,
                    window=window,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    minus_motif_label=args.minus_motif_label,
                    limit_windows=args.limit_windows,
                    progress_every=args.progress_every,
                    amp=args.amp,
                )
                total_rows += n_rows
                total_counts.update(counts)
                print(f"    - strand done: {n_rows:,} rows")
                maybe_fsync(fout, args.fsync)

    elapsed_min = (time.time() - t0) / 60
    print("\n=== DONE ===")
    print(f"Saved: {args.out}")
    print(f"Total rows: {total_rows:,}")
    print(f"Elapsed: {elapsed_min:.1f} min")
    print("Counts:")
    for (strand, motif), count in sorted(total_counts.items()):
        print(f"  {strand} {motif}: {count:,}")


if __name__ == "__main__":
    main()
