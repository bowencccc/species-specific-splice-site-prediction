# Benchmark reproduction

This document separates model reproduction from third-party comparison
scoring. All methods must ultimately write the same six-column per-site table:

```text
chrom  pos  strand  role  label  score
```

## 1. Verify training windows

```bash
python scripts/audit_windows.py --species human
python scripts/audit_windows.py --species chicken
python scripts/audit_windows.py --species arabidopsis
python scripts/audit_windows.py --species drosophila
```

Use `--check-all` to inspect every NPZ array rather than one example per
split. The reported counts are committed in the split manifests.

## 2. Train CNN--Mamba-k7

```bash
for species in human chicken arabidopsis drosophila; do
  python scripts/train.py \
    --species "$species" \
    --output-dir "runs/$species"
done
```

The human checkpoint is also applied zero-shot to chicken, Arabidopsis, and
Drosophila. It is not retrained for those three zero-shot rows.

## 3. Generate candidate tables

```bash
for species in human chicken arabidopsis drosophila; do
  python scripts/prepare_candidates.py --species "$species"
done
```

Candidate audits include input FASTA/annotation hashes, candidate counts,
positive counts, and noncanonical annotated boundary counts.

## 4. CNN--Mamba inference

`scripts/score_genome.py` scores the plus strand directly and the minus strand
through reverse-complement inference. Overlapping windows are partitioned at
their midpoints; the packaged reverse-complement bounds are the corrected
version used for the final score tables. Join raw motif scores to the shared
candidate table with `scripts/join_scores_to_candidates.py`.

## 5. SpliceAI

Download the five official SpliceAI `.h5` models into one directory, then run
for each species. Example:

```bash
python scripts/score_spliceai.py \
  --candidates data/candidates/drosophila.canonical_candidates.tsv.gz \
  --fasta data/raw/drosophila/Release_6_plus_ISO1_MT.fa \
  --chrom-map chrX=NC_004354.4 \
  --model-dir third_party/spliceai/models \
  --output data/site_scores/drosophila.spliceai.tsv.gz
```

The output is the arithmetic mean of the official five models.

## 6. AlphaGenome

Install the official AlphaGenome client and store the API key outside the
repository at `~/.config/alphagenome/api_key`. Example:

```bash
python scripts/score_alphagenome.py --species drosophila
```

Figure 1 uses the raw, tissue-agnostic `SPLICE_SITES` probability. It does not
use SSU or `probability * SSU`. Non-human sequences use the human output head,
which was the applicable official v1 head for custom-sequence zero-shot
prediction. Cached 1-Mb tiles allow interrupted API runs to resume.

## 7. Metrics and assertions

Run `scripts/evaluate_ap.py` for every score table. Candidate counts,
positives, row ordering, and labels must be identical for all models within a
species. `results/fig1_values.tsv` retains donor, acceptor, and arithmetic-mean
AP to six decimal places.

For exact fixed-recall/fixed-precision operating points:

```bash
python scripts/compute_pr_operating_points.py \
  --out-dir results/pr_operating_points_recomputed
```

The script first recomputes all Figure 1 AP values with distinct-threshold tie
handling and aborts if any value differs by more than `1e-6`.
