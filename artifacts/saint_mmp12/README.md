# SAINT-MMP12

Eleven clinical predictors plus MMP12, for Total CVD, ASCVD, and HF. Each outcome contains five original best-fold checkpoints, its Optuna database, and metadata with feature order, hyperparameters, and file checksums.

## Predict

```bash
git lfs pull
python -m scripts.core.saint_mmp12 predict --outcome Total_CVD \
  --input /path/to/preprocessed.csv --preprocessed-input \
  --output results/saint_mmp12_log_risk.csv
```

The original files contain weights only: their fitted preprocessing and Breslow estimators were not supplied. Original-weight inference requires continuous predictors standardized with the original fold transformations and the original categorical codes (0/1), in the feature order recorded in `metadata.json`. Use the same batch composition/order as the original analysis for prediction comparisons because SAINT uses row attention. The command returns ensemble log-risk scores, not calibrated event probabilities. A single preprocessed input is suitable only when its transformation is valid for every fold; otherwise score each fold with its own transformed matrix using `make_model` and `log_risk`, then average the scores. Do not fit replacement scaling on prediction data.

## Train

```bash
python -m scripts.core.saint_mmp12 train --outcome Total_CVD \
  --input /path/to/endpoint.csv --output results/saint_mmp12_refit --use-gpu
```

Refit using the saved best hyperparameters, a 10% European holdout, and five-fold training (seed 221). Preprocessing is fitted within each training fold. The new bundle saves fold weights, preprocessors, and Breslow estimators; it does not overwrite the original release. For raw-input log-risk prediction from this bundle, set `--artifact-root results/saint_mmp12_refit` and omit `--preprocessed-input`.

Hyperparameters come from the matching MMP12 Optuna studies and pass strict checkpoint loading. The old `hp_log.txt` files contain inconsistent parameter entries and are not copied. Temporary weights and result figures are excluded.
