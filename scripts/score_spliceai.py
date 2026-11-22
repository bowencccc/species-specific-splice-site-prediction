#!/usr/bin/env python3
"""Score a unified canonical-candidate table with the official 5-model ensemble."""

from __future__ import annotations

import argparse
import csv
import gzip
import time
from pathlib import Path

import numpy as np
from pyfaidx import Fasta


COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode, newline="") if path.suffix == ".gz" else path.open(mode, newline="")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def fetch_with_padding(fasta, chrom: str, start: int, length: int) -> str:
    chrom_length = len(fasta[chrom])
    left = max(0, -start)
    right = max(0, start + length - chrom_length)
    sequence = str(fasta[chrom][max(start, 0) : min(start + length, chrom_length)]).upper()
    return "N" * left + sequence + "N" * right


def one_hot_batch(sequences: list[str]) -> np.ndarray:
    lookup = np.zeros((256, 4), dtype=np.float32)
    for base, channel in zip(b"ACGT", range(4)):
        lookup[base, channel] = 1.0
        lookup[base + 32, channel] = 1.0
    encoded = np.empty((len(sequences), len(sequences[0]), 4), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        encoded[i] = lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    return encoded


def parse_chrom_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        display, source = value.split("=", 1)
        result[display] = source
    return result


def count_rows(path: Path) -> int:
    with open_text(path) as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def load_candidates(path: Path, chrom_codes: dict[str, int]):
    n = count_rows(path)
    chrom = np.empty(n, np.uint8)
    pos0 = np.empty(n, np.int32)
    strand = np.empty(n, np.uint8)
    role = np.empty(n, np.uint8)
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"chrom", "pos", "strand", "role", "label"}
        if not expected <= set(reader.fieldnames or []):
            raise ValueError(f"Candidate columns are not valid: {reader.fieldnames}")
        for i, row in enumerate(reader):
            chrom[i] = chrom_codes[row["chrom"]]
            motif_pos0 = int(row["pos"]) - 1
            is_minus = row["strand"] == "-"
            is_acceptor = row["role"] == "acceptor"
            # SpliceAI probabilities are positioned on the exon-boundary base,
            # while the shared candidate key is the intronic motif start.
            if not is_minus and not is_acceptor:      # + donor: exon last base
                boundary_pos0 = motif_pos0 - 1
            elif not is_minus and is_acceptor:        # + acceptor: exon first base
                boundary_pos0 = motif_pos0 + 2
            elif is_minus and not is_acceptor:        # - donor: high exon first base
                boundary_pos0 = motif_pos0 + 2
            else:                                     # - acceptor: low exon last base
                boundary_pos0 = motif_pos0 - 1
            pos0[i] = boundary_pos0
            strand[i] = is_minus
            role[i] = is_acceptor
    return chrom, pos0, strand, role


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom-map", nargs="+", required=True, help="display=FASTA_name")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-length", type=int, default=5000)
    parser.add_argument("--context", type=int, default=10000)
    args = parser.parse_args()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for the optional official SpliceAI scorer."
        ) from exc
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    chrom_map = parse_chrom_map(args.chrom_map)
    display_chroms = list(chrom_map)
    chrom_codes = {name: i for i, name in enumerate(display_chroms)}
    code_to_display = {i: name for name, i in chrom_codes.items()}

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__}; GPUs={gpus}", flush=True)
    model_paths = [args.model_dir / f"spliceai{i}.h5" for i in range(1, 6)]
    models = [tf.keras.models.load_model(path, compile=False) for path in model_paths]
    print(f"Loaded official five-model ensemble: {model_paths}", flush=True)

    chrom, boundary_pos0, strand, role = load_candidates(args.candidates, chrom_codes)
    output_starts = np.floor_divide(boundary_pos0, args.output_length) * args.output_length
    valid = boundary_pos0 >= 0
    valid_indices = np.flatnonzero(valid)
    keys = (
        (chrom[valid_indices].astype(np.uint64) << np.uint64(40))
        | (strand[valid_indices].astype(np.uint64) << np.uint64(39))
        | output_starts[valid_indices].astype(np.uint64)
    )
    order_in_valid = np.argsort(keys, kind="mergesort")
    order = valid_indices[order_in_valid]
    sorted_keys = keys[order_in_valid]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], len(order)]
    scores = np.zeros(len(chrom), np.float32)
    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)
    input_length = args.output_length + args.context
    flank = args.context // 2
    t0 = time.time()

    for batch0 in range(0, len(starts), args.batch_size):
        groups = range(batch0, min(batch0 + args.batch_size, len(starts)))
        sequences = []
        metadata = []
        for group in groups:
            idx = order[starts[group]]
            display = code_to_display[int(chrom[idx])]
            source = chrom_map[display]
            output_start = int(output_starts[idx])
            is_minus = bool(strand[idx])
            sequence = fetch_with_padding(fasta, source, output_start - flank, input_length)
            if is_minus:
                sequence = reverse_complement(sequence)
            if len(sequence) != input_length:
                raise RuntimeError(f"Unexpected sequence length: {len(sequence)}")
            sequences.append(sequence)
            metadata.append((group, output_start, is_minus))

        x = one_hot_batch(sequences)
        prediction_sum = None
        for model in models:
            prediction = model(x, training=False).numpy()
            prediction_sum = prediction if prediction_sum is None else prediction_sum + prediction
        predictions = prediction_sum / 5.0

        for batch_i, (group, output_start, is_minus) in enumerate(metadata):
            indices = order[starts[group] : ends[group]]
            local = (
                output_start + args.output_length - 1 - boundary_pos0[indices]
                if is_minus
                else boundary_pos0[indices] - output_start
            ).astype(np.int64)
            if np.any(local < 0) or np.any(local >= args.output_length):
                raise RuntimeError("Candidate mapped outside SpliceAI output chunk")
            classes = np.where(role[indices] == 1, 1, 2)
            scores[indices] = predictions[batch_i, local, classes]

        if batch0 == 0 or (batch0 // args.batch_size) % 100 == 0:
            done = min(batch0 + args.batch_size, len(starts))
            rate = done / max(time.time() - t0, 1e-9)
            print(
                f"chunks {done:,}/{len(starts):,}; {rate:.2f}/s; "
                f"ETA {(len(starts)-done)/max(rate,1e-9)/60:.1f} min",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_text(args.candidates) as source, open_text(args.output, "wt") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["chrom", "pos", "strand", "role", "label", "score"])
        for i, row in enumerate(reader):
            writer.writerow([row["chrom"], row["pos"], row["strand"], row["role"], row["label"], float(scores[i])])
    print(f"Saved {args.output}; candidates={len(scores):,}; chunks={len(starts):,}", flush=True)


if __name__ == "__main__":
    main()
