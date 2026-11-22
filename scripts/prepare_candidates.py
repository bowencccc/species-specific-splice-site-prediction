#!/usr/bin/env python3
"""Build the exact shared canonical-candidate universe for Figure 1.

Coordinates in the output are 1-based positions of the first base of the
forward-reference dinucleotide.  Thus minus-strand donor/acceptor candidates
are forward-reference AC/CT motif starts, respectively.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pyfaidx import Fasta


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIES_CONFIG = REPO_ROOT / "configs" / "species.json"

MOTIFS = (
    ("+", "donor", "GT"),
    ("+", "acceptor", "AG"),
    ("-", "donor", "AC"),
    ("-", "acceptor", "CT"),
)


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_attributes(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for raw in text.strip().strip(";").split(";"):
        field = raw.strip()
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
        elif " " in field:
            key, value = field.split(None, 1)
            value = value.strip().strip('"')
        else:
            continue
        result[key].extend(x for x in value.split(",") if x)
    return result


def load_truth(annotation: Path, chrom_map: dict[str, str]):
    transcripts: dict[tuple[str, str, str], set[tuple[int, int]]] = defaultdict(set)
    exon_rows = 0
    with open_text(annotation) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2].lower() != "exon":
                continue
            display_chrom = chrom_map.get(fields[0])
            strand = fields[6]
            if display_chrom is None or strand not in {"+", "-"}:
                continue
            attrs = parse_attributes(fields[8])
            parents = attrs.get("transcript_id") or attrs.get("Parent")
            if not parents:
                continue
            start1, end1 = int(fields[3]), int(fields[4])
            for parent in parents:
                transcripts[(display_chrom, strand, parent)].add((start1, end1))
            exon_rows += 1

    truth = {"donor": set(), "acceptor": set()}
    invalid_adjacent_pairs = 0
    introns = 0
    for (chrom, strand, _), exon_set in transcripts.items():
        exons = sorted(exon_set)
        for (_, low_end1), (high_start1, _) in zip(exons, exons[1:]):
            if high_start1 <= low_end1 + 1:
                invalid_adjacent_pairs += 1
                continue
            introns += 1
            if strand == "+":
                donor_pos0 = low_end1
                acceptor_pos0 = high_start1 - 3
            else:
                donor_pos0 = high_start1 - 3
                acceptor_pos0 = low_end1
            truth["donor"].add((chrom, strand, donor_pos0))
            truth["acceptor"].add((chrom, strand, acceptor_pos0))
    audit = {
        "transcripts_with_exons": len(transcripts),
        "exon_rows": exon_rows,
        "adjacent_exon_pairs": introns,
        "invalid_or_overlapping_adjacent_pairs_skipped": invalid_adjacent_pairs,
        "annotation_unique_donor_sites": len(truth["donor"]),
        "annotation_unique_acceptor_sites": len(truth["acceptor"]),
    }
    return truth, audit


def motif_positions(sequence: str, motif: str):
    start = 0
    while True:
        pos = sequence.find(motif, start)
        if pos < 0:
            return
        yield pos
        start = pos + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument("--species-config", type=Path, default=DEFAULT_SPECIES_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "candidates")
    args = parser.parse_args()

    with args.species_config.open() as handle:
        all_species = json.load(handle)
    if args.species not in all_species:
        raise ValueError(
            f"Unknown species {args.species!r}; choices={sorted(all_species)}"
        )
    cfg = all_species[args.species]
    fasta_path = Path(cfg["fasta"])
    annotation_path = Path(cfg["annotation"])
    if not fasta_path.is_absolute():
        fasta_path = REPO_ROOT / fasta_path
    if not annotation_path.is_absolute():
        annotation_path = REPO_ROOT / annotation_path
    for path in (fasta_path, annotation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.species}.canonical_candidates.tsv.gz"
    audit_path = args.out_dir / f"{args.species}.candidate_audit.json"
    if out_path.exists() or audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output for {args.species}")

    truth, audit = load_truth(annotation_path, cfg["annotation_chroms"])
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    candidate_counts: dict[str, int] = defaultdict(int)
    positive_counts: dict[str, int] = defaultdict(int)

    with gzip.open(out_path, "wt", newline="", compresslevel=6) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["chrom", "pos", "strand", "role", "label", "motif", "source_chrom"])
        for display_chrom, source_chrom in cfg["chroms"]:
            if source_chrom not in fasta:
                raise KeyError(f"{source_chrom} absent from {fasta_path}")
            sequence = str(fasta[source_chrom][:]).upper()
            for strand, role, motif in MOTIFS:
                for pos0 in motif_positions(sequence, motif):
                    key = (display_chrom, strand, pos0)
                    label = int(key in truth[role])
                    writer.writerow([display_chrom, pos0 + 1, strand, role, label, motif, source_chrom])
                    candidate_counts[role] += 1
                    positive_counts[role] += label

    audit.update(
        {
            "species": args.species,
            "coordinate_definition": (
                "pos is 1-based forward-reference start of the canonical dinucleotide; "
                "plus donor/acceptor=GT/AG and minus donor/acceptor=AC/CT"
            ),
            "label_definition": (
                "all-transcript adjacent-exon intron boundary union, deduplicated by "
                "(chrom,strand,role,pos); no protein-coding filter"
            ),
            "candidate_counts": dict(candidate_counts),
            "positive_counts": dict(positive_counts),
            "annotation_noncanonical_or_sequence_mismatch_sites": {
                role: len(truth[role]) - positive_counts[role] for role in ("donor", "acceptor")
            },
            "fasta": str(fasta_path.resolve()),
            "fasta_sha256": sha256(fasta_path),
            "annotation": str(annotation_path.resolve()),
            "annotation_sha256": sha256(annotation_path),
            "candidate_file": str(out_path.resolve()),
            "candidate_file_sha256": sha256(out_path),
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
