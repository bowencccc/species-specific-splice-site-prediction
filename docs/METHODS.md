# Reproducible methods

## Architecture

The model embeds each nucleotide into 192 dimensions and applies eight
residual CNN--Mamba blocks.  Each block contains a depthwise 1D convolution
with kernel size 7, a pointwise projection, and a bidirectional Mamba3 layer.
Forward and reverse Mamba outputs are summed at each position.  A layer norm
and two-layer classification head produce three logits per nucleotide:
background, donor, and acceptor.  There is no whole-window pooling.

## Optimization

The reported models used seed 42, FP32, AdamW (`betas=(0.9, 0.95)`), peak
learning rate `3e-4`, weight decay `1e-4`, gradient clipping at 1.0, batch size
4, and gradient accumulation 2.  A linear warmup is followed by cosine decay
over five epochs.  Loss is three-class cross entropy evaluated only at GT/AG
candidate positions and annotated donor/acceptor positions.  The checkpoint
with the largest validation mean of donor and acceptor AP is retained.

## Splits

| Species | Train | Validation | Test |
|---|---|---|---|
| Human | chr2–6, 8, 10–14, 16–22, X, Y | chr7, chr15 | chr1, chr9 |
| Chicken | chr3–39, W, Z | chr2 | chr1 |
| *A. thaliana* | chr2, chr3, chr5 | chr4 | chr1 |
| *D. melanogaster* | chr2L, chr2R, chr3L, chr4 | chr3R | chrX |

## Evaluation

Donor and acceptor average precision are calculated separately using exact
tied-score handling equivalent to `sklearn.metrics.average_precision_score`.
The reported value is their arithmetic mean.  All compared models use the
same per-species canonical candidate table, labels, chromosomes, and strands.
SpliceAI is the mean of its five official models.  AlphaGenome uses the raw,
tissue-agnostic `SPLICE_SITES` probability; missing API candidates receive
zero and no SSU multiplication or score truncation is applied in Figure 1.

The historical April-16 AlphaGenome shared-candidate analysis is a separate
evaluation set and must not be mixed with the genome-wide canonical Figure 1
benchmark.  Its earlier `0.920` SSM result came from a pure bidirectional
Mamba checkpoint, not the CNN--Mamba-k7 checkpoint; see
`docs/RESULT_PROVENANCE.md`.
