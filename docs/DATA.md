# Data and artifact layout

Large genomes, annotations, NPZ training windows, checkpoints, candidate
tables, and per-site score tables are intentionally excluded from Git.  Their
expected locations are defined in `configs/species.json`.

## Raw inputs

| Species | Assembly | Annotation used for the reported benchmark | Test chromosome(s) |
|---|---|---|---|
| Human | GRCh38.p14 | CHESS 3.1.3 | chr1 and chr9 |
| Chicken | GRCg7b, GCF_016699485.2 | RefSeq GCF_016699485.2 | chr1 |
| *Arabidopsis thaliana* | TAIR10.1, GCF_000001735.4 | RefSeq/Araport | chr1 |
| *Drosophila melanogaster* | Release 6 plus ISO1 MT, GCF_000001215.4 | RefSeq/FlyBase 6.54 | chrX |

The exact FASTA and annotation SHA256 values are preserved in
`results/candidate_audits/*.candidate_audit.json`.

## Training-window contract

Each NPZ file represents one 10,000-nt, strand-oriented window.  Adjacent
windows start 5,000 nt apart.  Required arrays are:

```text
sequence  int64 [10000]  A=0, C=1, G=2, T=3, N=4
labels    int64 [10000]  background=0, donor=1, acceptor=2
```

Files must be stored below the configured species window directory and named
so that chromosome and strand precede the coordinate, for example
`chr1_plus_100000.npz`.  The training script recursively discovers
`*/*.npz`, deduplicates identical filenames, and selects files by chromosome.
The exact chromosome lists and reported window counts are preserved under
`artifacts/checkpoint_metadata/<species>/split_manifest.json`.

## Canonical candidate definition

Candidates comprise every canonical motif on both strands of the test
chromosome(s): forward-reference GT/AG for plus-strand donor/acceptor and
AC/CT for minus-strand donor/acceptor.  `pos` is the 1-based first nucleotide
of the forward-reference dinucleotide.  Labels are the union of intron
boundaries obtained from adjacent exons across all annotated transcripts,
deduplicated by `(chrom, strand, role, pos)`.  No protein-coding filter is
applied.

Generate candidates with:

```bash
python scripts/prepare_candidates.py --species human
```

## External artifacts

`artifacts/external_artifacts.tsv` records the intended relative path, byte
size, and SHA256 for every reported checkpoint, canonical candidate table,
and final per-site score table.  Populate those paths from the archived
artifact bundle, then run:

```bash
python scripts/verify_external_artifacts.py
```
