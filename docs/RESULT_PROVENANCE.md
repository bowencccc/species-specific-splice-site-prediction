# Result provenance

## Primary Figure 1

`results/fig1_values.tsv` is the authoritative primary result table.  It uses
raw AlphaGenome splice-site probability, the official five-model SpliceAI
mean, the human-trained CNN--Mamba model, and (outside human) a
species-trained CNN--Mamba model.

The four reported CNN--Mamba checkpoints are identified by SHA256 in
`artifacts/external_artifacts.tsv`; their full training configurations,
histories, and split manifests are under `artifacts/checkpoint_metadata/`.

## April-16 human shared-candidate analysis

The draft abstract supplied with this repository contains the values 0.932
(AlphaGenome probability), 0.920 (SSM), and 0.918 (SpliceAI) on an April-16
AlphaGenome fold3/valid candidate subset restricted to chr1 and chr9.  Audit
of the checkpoint shows that the 0.920 model is the older pure bidirectional
Mamba architecture (`d_model=256`, eight layers), with state keys
`norms.*`/`layers.*` and no convolutional parameters.  It therefore cannot be
presented as a direct result of the hybrid CNN--Mamba-k7 architecture without
qualification.

`scripts/evaluate_alphagenome_cv_subset.py` performs the corresponding exact
evaluation using the actual hybrid checkpoint's unified chr1/chr9 scores.
Its output is `results/alphagenome_cv_subset.tsv`. On the 42,113 canonical
candidates, the corrected mean AP values are:

| Model | Donor AP | Acceptor AP | Mean AP |
|---|---:|---:|---:|
| CNN--Mamba (human-trained) | 0.932574 | 0.935476 | **0.934025** |
| AlphaGenome raw probability | 0.934081 | 0.930820 | 0.932451 |
| SpliceAI five-model mean | 0.918309 | 0.918220 | 0.918264 |

The April-16 input had already been filtered upstream to AlphaGenome
probability at least 0.1. These values are therefore a shared-candidate subset
analysis, not a replacement for the complete canonical-candidate Figure 1.

`scripts/plot_alphagenome_cv_subset_matched_precision.py` computes Figure 6
from the same joined rows. At each requested recall it uses the first complete
distinct-score group whose cumulative recall reaches the target; it never
splits tied predictions to force exact recall. The committed operating-point
table records the requested recall, achieved recall, score threshold, TP, FP,
step precision, and interpolated precision. The local joined per-site table is
not distributed in Git, but all three source-file SHA256 values and the exact
coordinate conversion are recorded in
`results/alphagenome_cv_subset_provenance.json`.
