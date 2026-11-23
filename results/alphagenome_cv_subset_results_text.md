# Shared-site cross-validation subset Results text

## Results paragraph

On the shared-site cross-validation subset, CNN--Mamba (human-trained) and
AlphaGenome showed comparable overall performance, with mean donor/acceptor AP
values of 0.9340 and 0.9325, respectively, compared with 0.9183 for SpliceAI.
AlphaGenome was slightly better for donors (0.9341 versus 0.9326), whereas
CNN--Mamba (human-trained) was slightly better for acceptors (0.9355 versus
0.9308). At matched recall targets
of 0.50, 0.70, and 0.80, AlphaGenome had the highest precision for both splice
site types. At recall 0.85, its precision was 0.910 for donors and 0.907 for
acceptors, compared with 0.849 and 0.856 for CNN--Mamba (human-trained). On a
0.005 recall grid, the final high-recall crossover occurred at recall 0.885
for both site types. At recall 0.90, CNN--Mamba (human-trained) reached
precision values of 0.757 for
donors and 0.770 for acceptors, compared with 0.715 and 0.717 for AlphaGenome
and 0.707 and 0.733 for SpliceAI. At recall 0.92, the corresponding values were
0.711 and 0.715 for CNN--Mamba (human-trained), 0.627 and 0.615 for AlphaGenome, and 0.655
and 0.664 for SpliceAI. Thus, the models had similar average precision but
different operating characteristics: AlphaGenome was strongest at the
prespecified moderate-recall points, whereas CNN--Mamba (human-trained) retained higher
precision in the high-recall regime. Because the subset was prefiltered using
AlphaGenome's own raw splice-site probability, these results describe a
restricted shared-site comparison rather than an unbiased whole-chromosome
benchmark.

## Figure 6 caption

**Figure 6. Precision at matched recall on the shared-site cross-validation
subset.** Donor and acceptor precision are shown at recall targets of 0.50,
0.70, 0.80, 0.85, 0.90, and 0.92 for AlphaGenome, CNN--Mamba
(human-trained), and the official five-model SpliceAI ensemble. Each point is
the precision at the first distinct-score threshold whose recall reaches or
exceeds the target; tied
scores are included together. Recall targets are shown at equal spacing, and
points are offset slightly within each target for visibility. The comparison contains 42,113
canonical GT/AG candidates on human chromosomes 1 and 9, with labels defined from the union of
CHESS 3.1.3 transcript exon chains. The AlphaGenome input was prefiltered
upstream to raw splice-site probability at least 0.1, so this is a
restricted shared-candidate analysis rather than a whole-chromosome benchmark.

## Precision values plotted in Figure 6

| Role | Recall | AlphaGenome | CNN--Mamba (human-trained) | SpliceAI |
|---|---:|---:|---:|---:|
| Donor | 0.50 | 0.995 | 0.991 | 0.979 |
| Donor | 0.70 | 0.989 | 0.964 | 0.958 |
| Donor | 0.80 | 0.963 | 0.912 | 0.906 |
| Donor | 0.85 | 0.910 | 0.849 | 0.835 |
| Donor | 0.90 | 0.715 | 0.757 | 0.707 |
| Donor | 0.92 | 0.627 | 0.711 | 0.655 |
| Acceptor | 0.50 | 0.995 | 0.991 | 0.976 |
| Acceptor | 0.70 | 0.989 | 0.965 | 0.953 |
| Acceptor | 0.80 | 0.963 | 0.917 | 0.906 |
| Acceptor | 0.85 | 0.907 | 0.856 | 0.838 |
| Acceptor | 0.90 | 0.717 | 0.770 | 0.733 |
| Acceptor | 0.92 | 0.615 | 0.715 | 0.664 |
