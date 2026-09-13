# Reproducibility notes

The manuscript settings are encoded in `config/analysis.yml`: random seed 221,
90:10 European development/hold-out split, five-fold cross-validation, 100
Optuna trials, 100 repeated LassoNet splits, an 80% stability threshold, and
1,000 nonparametric bootstrap samples.

Raw UK Biobank data, participant identifiers, participant-level predictions,
model checkpoints, TabPFN embeddings, Optuna databases, and licensed GWAS or
pQTL data must remain outside version control. Generated files are written
under `artifacts/` or `results/`, both of which are ignored by Git.

The current environment records Python 3.10.1 as reported in the manuscript.
An exact package lock should be exported from the workstation used for the
final rerun before creating a release tag.

Each fold-specific model is stored together with the scaler and categorical
encoder fitted on that fold's training partition. Prediction and SHAP stages
reload the matching preprocessor; they never refit an encoder on hold-out or
external-ancestry participants.
