# April-16 shared-candidate Results text

## Results paragraph

On the April-16 shared-candidate subset, CNN--Mamba-k7 and AlphaGenome showed
comparable overall performance, with mean donor/acceptor AP values of 0.9340
and 0.9325, respectively, compared with 0.9183 for SpliceAI. AlphaGenome was
slightly better for donors (0.9341 versus 0.9326), whereas CNN--Mamba-k7 was
slightly better for acceptors (0.9355 versus 0.9308). At matched recall targets
of 0.50, 0.70, and 0.80, AlphaGenome had the highest precision for both splice
site types. At recall 0.85, its precision was 0.910 for donors and 0.907 for
acceptors, compared with 0.849 and 0.856 for CNN--Mamba-k7. The final
high-recall crossover occurred between recall 0.880 and 0.885 for both site
types. At recall 0.90, CNN--Mamba-k7 reached precision values of 0.757 for
donors and 0.770 for acceptors, compared with 0.715 and 0.717 for AlphaGenome
and 0.707 and 0.733 for SpliceAI. At recall 0.92, the corresponding values were
0.711 and 0.715 for CNN--Mamba-k7, 0.627 and 0.615 for AlphaGenome, and 0.655
and 0.664 for SpliceAI. Thus, the models had similar average precision but
different operating characteristics: AlphaGenome was strongest at the
prespecified moderate-recall points, whereas CNN--Mamba-k7 retained higher
precision in the high-recall regime.

## Figure 6 caption

**Figure 6. Precision at matched recall on the AlphaGenome shared-candidate
subset.** Donor and acceptor precision are shown at recall targets of 0.50,
0.70, 0.80, and 0.90 for AlphaGenome, CNN--Mamba-k7, and the official
five-model SpliceAI ensemble. Each point is the precision at the first
distinct-score threshold whose recall reaches or exceeds the target; tied
scores are included together. The comparison contains 42,113 canonical GT/AG
candidates on human chromosomes 1 and 9, with labels defined from the union of
CHESS 3.1.3 transcript exon chains. The April-16 AlphaGenome input was
prefiltered upstream to raw splice-site probability at least 0.1, so this is a
restricted shared-candidate analysis rather than a whole-chromosome benchmark.

## Precision values plotted in Figure 6

| Role | Recall | AlphaGenome | CNN--Mamba-k7 | SpliceAI |
|---|---:|---:|---:|---:|
| Donor | 0.50 | 0.995 | 0.991 | 0.979 |
| Donor | 0.70 | 0.989 | 0.964 | 0.958 |
| Donor | 0.80 | 0.963 | 0.912 | 0.906 |
| Donor | 0.90 | 0.715 | 0.757 | 0.707 |
| Acceptor | 0.50 | 0.995 | 0.991 | 0.976 |
| Acceptor | 0.70 | 0.989 | 0.965 | 0.953 |
| Acceptor | 0.80 | 0.963 | 0.917 | 0.906 |
| Acceptor | 0.90 | 0.717 | 0.770 | 0.733 |
