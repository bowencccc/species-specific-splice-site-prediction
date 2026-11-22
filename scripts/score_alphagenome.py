#!/usr/bin/env python3
"""Score the shared Figure-1 candidates with the official AlphaGenome API.

The API's dense SPLICE_SITES output is indexed at the exon-side splice
boundary, while the shared candidate table is indexed at the first base of
the intronic canonical dinucleotide.  The empirically verified offsets are:

  + donor -1, + acceptor +2, - donor +2, - acceptor -1.

Predictions use overlapping 1 MiB sequence requests.  Only the central owned
portion of each tile is retained, and every tile's candidate scores are cached
as a compressed NPZ so interrupted runs can resume without repeating API
requests.  Non-human sequences use AlphaGenome's human output head, which is
the only applicable official v1 organism head for zero-shot custom sequence
prediction (the API exposes human and mouse heads only).
"""

from __future__ import annotations

import argparse
from array import array
import gzip
import json
from pathlib import Path
import time

import numpy as np
from pyfaidx import Fasta

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIES_CONFIG = REPO_ROOT / "configs" / "species.json"

GROUPS = (("+", "donor"), ("+", "acceptor"), ("-", "donor"), ("-", "acceptor"))
BOUNDARY_OFFSETS = {
    ("+", "donor"): -1,
    ("+", "acceptor"): 2,
    ("-", "donor"): 2,
    ("-", "acceptor"): -1,
}
GROUP_KEY = {
    ("+", "donor"): "plus_donor",
    ("+", "acceptor"): "plus_acceptor",
    ("-", "donor"): "minus_donor",
    ("-", "acceptor"): "minus_acceptor",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--species-config", type=Path, default=DEFAULT_SPECIES_CONFIG)
    parser.add_argument(
        "--candidate-file",
        type=Path,
        help="Defaults to data/candidates/<species>.canonical_candidates.tsv.gz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Defaults to data/site_scores/<species>.alphagenome.tsv.gz",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "data" / "alphagenome_tiles",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".config" / "alphagenome" / "api_key",
    )
    parser.add_argument("--stride", type=int, default=524_288)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--limit-tiles", type=int, default=None)
    return parser.parse_args()


def load_candidates(path: Path):
    """Load compact uint32 positions and uint8 labels by chrom/strand/role."""
    positions: dict[tuple[str, str, str], array] = {}
    labels: dict[tuple[str, str, str], array] = {}
    with gzip.open(path, "rt") as handle:
        header = next(handle).rstrip("\n").split("\t")
        expected = ["chrom", "pos", "strand", "role", "label", "motif", "source_chrom"]
        if header != expected:
            raise ValueError(f"Unexpected candidate header: {header}")
        for line in handle:
            chrom, pos, strand, role, label, _motif, _source = line.rstrip("\n").split("\t")
            key = (chrom, strand, role)
            positions.setdefault(key, array("I")).append(int(pos))
            labels.setdefault(key, array("B")).append(int(label))
    result = {}
    for key in positions:
        pos = np.frombuffer(positions[key], dtype=np.uint32)
        lab = np.frombuffer(labels[key], dtype=np.uint8)
        if len(pos) > 1 and np.any(pos[1:] <= pos[:-1]):
            raise ValueError(f"Candidate positions are not strictly increasing for {key}")
        result[key] = {"pos": pos, "label": lab, "score": np.zeros(len(pos), dtype=np.float32)}
    return result


def fetch_padded(fasta: Fasta, chrom: str, start: int, length: int) -> str:
    chrom_len = len(fasta[chrom])
    real_end = min(start + length, chrom_len)
    seq = str(fasta[chrom][start:real_end]).upper()
    # The official API accepts A/C/G/T/N only.  Reference assemblies can also
    # contain IUPAC ambiguity codes (for example Y in TAIR10.1); preserve the
    # fact that the base is unknown by mapping every such code to N.
    seq = "".join(base if base in "ACGTN" else "N" for base in seq)
    if len(seq) < length:
        seq += "N" * (length - len(seq))
    if len(seq) != length:
        raise AssertionError((chrom, start, len(seq), length))
    return seq


def keep_bounds(tile_index: int, n_tiles: int, window: int, stride: int):
    if n_tiles == 1:
        return 0, window
    overlap = window - stride
    left = overlap // 2
    if tile_index == 0:
        return 0, window - left
    if tile_index == n_tiles - 1:
        return left, window
    return left, window - left


def predict_with_retry(client, sequence: str, max_retries: int):
    for attempt in range(1, max_retries + 1):
        try:
            return client.predict_sequence(
                sequence,
                organism=dna_client.Organism.HOMO_SAPIENS,
                requested_outputs=[dna_client.OutputType.SPLICE_SITES],
                ontology_terms=None,
            ).splice_sites
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = min(120, 5 * (2 ** (attempt - 1)))
            print(
                f"API request failed ({type(exc).__name__}: {exc}); "
                f"retry {attempt}/{max_retries} after {delay}s",
                flush=True,
            )
            time.sleep(delay)


def cache_one_tile(
    *, client, fasta, source_chrom, display_chrom, start, keep_l, keep_r,
    groups, cache_path, window, max_retries,
):
    seq = fetch_padded(fasta, source_chrom, start, window)
    t0 = time.time()
    track = predict_with_retry(client, seq, max_retries)
    metadata = {
        (row["strand"], row["name"]): index
        for index, row in track.metadata.iterrows()
    }
    expected_channels = set(GROUPS)
    if set(metadata) != expected_channels:
        raise ValueError(f"Unexpected AlphaGenome splice metadata: {metadata}")

    payload = {
        "display_chrom": np.asarray(display_chrom),
        "source_chrom": np.asarray(source_chrom),
        "start0": np.asarray(start, dtype=np.int64),
        "keep_l": np.asarray(keep_l, dtype=np.int64),
        "keep_r": np.asarray(keep_r, dtype=np.int64),
    }
    global_lo = start + keep_l
    global_hi = start + keep_r
    for strand, role in GROUPS:
        key = (display_chrom, strand, role)
        pos = groups[key]["pos"]
        label = groups[key]["label"]
        offset = BOUNDARY_OFFSETS[(strand, role)]
        # target0 = pos1 - 1 + offset; select global_lo <= target0 < global_hi
        lo = int(np.searchsorted(pos, global_lo + 1 - offset, side="left"))
        hi = int(np.searchsorted(pos, global_hi + 1 - offset, side="left"))
        selected_pos = pos[lo:hi]
        selected_label = label[lo:hi]
        target_index = selected_pos.astype(np.int64) - 1 + offset - start
        if len(target_index) and (target_index.min() < 0 or target_index.max() >= window):
            raise AssertionError((key, start, target_index.min(), target_index.max()))
        score = np.asarray(
            track.values[target_index, metadata[(strand, role)]], dtype=np.float32
        )
        prefix = GROUP_KEY[(strand, role)]
        payload[f"{prefix}_index_lo"] = np.asarray(lo, dtype=np.int64)
        payload[f"{prefix}_pos"] = selected_pos
        payload[f"{prefix}_label"] = selected_label
        payload[f"{prefix}_score"] = score

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(cache_path)
    print(
        f"cached {display_chrom}:{start:,} in {time.time() - t0:.1f}s -> {cache_path.name}",
        flush=True,
    )


def restore_tile(cache_path: Path, display_chrom: str, groups):
    with np.load(cache_path, allow_pickle=False) as data:
        if str(data["display_chrom"]) != display_chrom:
            raise ValueError(f"Wrong chromosome in {cache_path}")
        for strand, role in GROUPS:
            prefix = GROUP_KEY[(strand, role)]
            lo = int(data[f"{prefix}_index_lo"])
            pos = data[f"{prefix}_pos"]
            label = data[f"{prefix}_label"]
            score = data[f"{prefix}_score"]
            key = (display_chrom, strand, role)
            hi = lo + len(pos)
            if not np.array_equal(groups[key]["pos"][lo:hi], pos):
                raise ValueError(f"Cached candidate positions disagree in {cache_path}: {key}")
            if not np.array_equal(groups[key]["label"][lo:hi], label):
                raise ValueError(f"Cached labels disagree in {cache_path}: {key}")
            groups[key]["score"][lo:hi] = score


def write_output(path: Path, cfg, groups):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", compresslevel=6) as out:
        out.write("chrom\tpos\tstrand\trole\tlabel\tscore\n")
        for display_chrom, _source_chrom in cfg["chroms"]:
            for strand, role in GROUPS:
                group = groups[(display_chrom, strand, role)]
                for pos, label, score in zip(group["pos"], group["label"], group["score"]):
                    out.write(
                        f"{display_chrom}\t{int(pos)}\t{strand}\t{role}\t"
                        f"{int(label)}\t{float(score):.8g}\n"
                    )


def main():
    global dna_client
    args = parse_args()
    try:
        from alphagenome.models import dna_client
    except ImportError as exc:
        raise SystemExit(
            "The optional official AlphaGenome client is not installed."
        ) from exc
    with args.species_config.open() as handle:
        all_species = json.load(handle)
    if args.species not in all_species:
        raise ValueError(
            f"Unknown species {args.species!r}; choices={sorted(all_species)}"
        )
    cfg = all_species[args.species]
    candidate_file = args.candidate_file or (
        REPO_ROOT / "data" / "candidates" /
        f"{args.species}.canonical_candidates.tsv.gz"
    )
    out = args.out or (
        REPO_ROOT / "data" / "site_scores" /
        f"{args.species}.alphagenome.tsv.gz"
    )
    fasta_path = Path(cfg["fasta"])
    if not fasta_path.is_absolute():
        fasta_path = REPO_ROOT / fasta_path
    key = args.api_key_file.read_text().strip()
    if not key:
        raise ValueError(f"Empty API key file: {args.api_key_file}")
    if args.stride <= 0 or args.stride > int(dna_client.SEQUENCE_LENGTH_1MB):
        raise ValueError("stride must be in (0, 1 MiB]")

    print(f"Loading candidates: {candidate_file}", flush=True)
    groups = load_candidates(candidate_file)
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    client = dna_client.create(key, timeout=60)
    window = int(dna_client.SEQUENCE_LENGTH_1MB)
    made = 0

    for display_chrom, source_chrom in cfg["chroms"]:
        chrom_len = len(fasta[source_chrom])
        starts = list(range(0, chrom_len, args.stride))
        print(
            f"{args.species} {display_chrom}: length={chrom_len:,}, "
            f"tiles={len(starts)}, window={window:,}, stride={args.stride:,}",
            flush=True,
        )
        for tile_index, start in enumerate(starts):
            cache_path = (
                args.cache_dir / args.species / display_chrom /
                f"start_{start:012d}.npz"
            )
            if not cache_path.exists():
                keep_l, keep_r = keep_bounds(tile_index, len(starts), window, args.stride)
                cache_one_tile(
                    client=client,
                    fasta=fasta,
                    source_chrom=source_chrom,
                    display_chrom=display_chrom,
                    start=start,
                    keep_l=keep_l,
                    keep_r=keep_r,
                    groups=groups,
                    cache_path=cache_path,
                    window=window,
                    max_retries=args.max_retries,
                )
                made += 1
                if args.limit_tiles is not None and made >= args.limit_tiles:
                    print("Stopped after requested number of new tiles; output not assembled.", flush=True)
                    return
            restore_tile(cache_path, display_chrom, groups)

    write_output(out, cfg, groups)
    audit = {
        "species": args.species,
        "model": "AlphaGenome official v1 API SPLICE_SITES",
        "organism_head": "HOMO_SAPIENS",
        "tissue_agnostic": True,
        "splice_site_usage_used": False,
        "sequence_window": window,
        "stride": args.stride,
        "boundary_offsets_from_intronic_dinucleotide_start": {
            f"{strand}_{role}": offset
            for (strand, role), offset in BOUNDARY_OFFSETS.items()
        },
        "candidate_file": str(candidate_file.resolve()),
        "output": str(out.resolve()),
        "missing_candidates_assigned_zero": True,
        "sequence_sanitization": "IUPAC ambiguity characters outside A/C/G/T/N mapped to N",
        "note": (
            "The official API returned dense splice-site tracks. Only candidates "
            "whose shifted boundary coordinate falls outside the chromosome can "
            "remain zero; no score truncation was applied."
        ),
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
