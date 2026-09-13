# Plasma proteomics and deep survival learning for cardiovascular risk prediction

This repository contains the analysis code for predicting incident total
cardiovascular disease, atherosclerotic cardiovascular disease, and heart
failure in UK Biobank. It compares PREVENT, refitted Cox regression, XGBoost,
MLP, TabNet, NODE, FT-Transformer, SAINT, and a TabPFN-derived survival model
under clinical-only, proteomic-only, and combined predictor configurations.


## Analysis workflow

1. Select endpoint-specific proteins using 100 repeated 70:30 splits of the
   European development data and an 80% selection-frequency threshold.
2. Reserve 10% of European participants as the final hold-out set.
3. Tune each model with 100 Optuna trials and five-fold cross-validation in the
   remaining European development set.
4. Retain the five fold-specific models and evaluate their averaged predictions
   in the European hold-out, Asian ancestry, and other ancestry cohorts.
5. Evaluate C-index, 10-year time-dependent AUC, calibration, and decision-curve
   net benefit with 1,000 bootstrap samples.
6. Compute direct TreeSHAP explanations for XGBoost and surrogate TreeSHAP
   explanations for non-tree models.
7. Validate consensus interactions with adjusted Cox models and train the
   Kneedle-selected reduced-panel SAINT model.
8. Integrate protein importance with MAPLE and SuSiE results when the workstation
   genetics scripts are added.

## Installation

Create the Python 3.10.1 environment:

```bash
conda env create -f environment.yml
conda activate cvd-survival
```

Install the R packages listed in `requirements-r.txt` using the R package
management approach used by your computing environment. Scripts do not install
packages automatically.

## Local data configuration

Copy `config/paths.example.yml` to `config/paths.local.yml`. The local file is
ignored by Git and can be loaded with `--config config/paths.local.yml`. See
`docs/data_contract.md` for expected columns. Direct CLI values take precedence
over YAML values.

## Command-line usage

Every command runs one explicit outcome, predictor set, and model. There are no
hard-coded experiment loops. For example:

```bash
python run.py select-proteins \
  --disease-type ASCVD \
  --data-type all \
  --model-name XGBoost \
  --path /secure/data/ASCVD/followup_incident_20260122.csv \
  --all-proteins-path /secure/data/all_protein_names.csv

python run.py train \
  --disease-type ASCVD \
  --data-type all \
  --model-name SAINT \
  --path /secure/data/ASCVD/followup_incident_20260122.csv \
  --protein-path /secure/panels/ASCVD/lassonet_protein.csv \
  --n-trials 100

python run.py predict \
  --disease-type ASCVD \
  --data-type all \
  --model-name SAINT \
  --path /secure/data/ASCVD/followup_incident_20260122.csv \
  --protein-path /secure/panels/ASCVD/lassonet_protein.csv
```

Use `FT-Transformer` or `SAINT` explicitly. They have separate public classes:
FT-Transformer uses column attention and SAINT uses combined column-row
attention. Their shared survival-training implementation is internal.

Run `python run.py --help` to list stages, followed by a stage and `--help` to
view modelling arguments.

## Repository layout

- `models/`: manuscript model implementations.
- `utils/`: data contracts, fold-local preprocessing, survival metrics, paths,
  and persistence helpers.
- `scripts/core/`: import-only implementations called by `run.py`.
- `analysis/`: final population, performance, interpretation, genetics,
  interaction, and reduced-panel analyses assembled from the manuscript work.
- `genetics/maple_susie/`: reserved location for workstation-only source code.
- `docs/`: data and reproducibility documentation.

## Data availability

UK Biobank data are available to approved researchers through the UK Biobank
access process. Users must construct endpoint-specific input files under their
own authorization.
