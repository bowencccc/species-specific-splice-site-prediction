# Results

- `fig1_values.tsv`: authoritative donor, acceptor, and mean AP values for the
  four-species unified benchmark.
- `model_summary.tsv`: selected epoch, validation AP, test AP, and checkpoint
  identity for each species-trained CNN--Mamba-k7 model.
- `metrics/`: exact per-model AP JSON files.
- `candidate_audits/`: test-set coordinate definitions, counts, and raw-input
  SHA256 values.
- `pr_operating_points/`: exact precision-at-recall, recall-at-precision, and
  manuscript-claim checks.
- `alphagenome_cv_subset.tsv`: architecture-matched reanalysis of the separate
  April-16 AlphaGenome candidate subset. It supersedes the historical 0.920
  pure-Mamba value when describing the hybrid architecture.
- `alphagenome_cv_subset_matched_precision.tsv`: exact threshold operating
  points at matched recall for the April-16 subset.
- `alphagenome_cv_subset_precision_grid.tsv`: the same calculation on a 0.005
  recall grid for crossover and manuscript-text checks.
- `alphagenome_cv_subset_results_text.md`: a Results paragraph and Figure 6
  caption based on the corrected CNN--Mamba-k7 scores.

The primary Figure 1 and the April-16 subset use different candidate
universes and must not be combined into one ranking or treated as duplicate
estimates.
