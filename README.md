# Cardiovascular survival prediction with plasma proteomics

Code and trained models for proteomics-based prediction of incident Total CVD, ASCVD, and HF in UK Biobank.

中文说明：[README_cn.md](README_cn.md)

## Workflow

1. Select stable proteins using repeated LassoNet.
2. Tune models with Optuna and train five-fold ensembles.
3. Generate ensemble and out-of-fold predictions.
4. Evaluate discrimination, calibration, and clinical utility.
5. Perform model interpretation and reduced-panel analyses.

Models include Cox, XGBoost, MLP, TabNet, NODE, FT-Transformer, SAINT, and TabPFN.

## Setup

```bash
conda env create -f environment.yml
conda activate cvd-survival
python examples/run_synthetic_workflow.py
```

The release environment uses Python 3.10.1. R dependencies are listed in `requirements-r.txt`.

## Input and configuration

Copy `config/paths.example.yml` to `config/paths.local.yml` and set the local input and output paths.

Each endpoint file requires `eid`, `Ethnic`, `Is_Incident`, survival time, and the selected predictors. The predictor set included age, sex, smoking status, diabetes, lipid-lowering medication use, high-density lipoprotein (HDL) cholesterol, non-HDL cholesterol, antihypertensive medication use, systolic blood pressure, estimated glomerular filtration rate (eGFR), and body mass index (BMI).

| Predictor | Input field |
| --- | --- |
| Age | `age` |
| Sex | `sex` |
| Smoking status (ever smoked) | `ever_smoked` |
| Diabetes | `Diabetes_baseline` |
| Lipid-lowering medication use | `Cholesterol_treatment` |
| HDL cholesterol | `hdl_cholesterol` |
| Non-HDL cholesterol | `non_hdl_cholesterol` |
| Antihypertensive medication use | `hypertension_treatment` |
| Systolic blood pressure | `average_SBP` |
| eGFR | `eGFR` |
| BMI | `BMI` |

If `non_hdl_cholesterol` is absent, it is calculated from total and HDL cholesterol. Protein-panel files use the column `Protein_Name`.

## Run

```bash
# Show available stages
python run.py --help

# Train one configuration
python run.py train --config config/paths.local.yml \
  --disease-type ASCVD --data-type all --model-name SAINT \
  --protein-path features/lassonet/ASCVD/lassonet_protein.csv

# Generate ensemble predictions
python run.py predict --config config/paths.local.yml \
  --disease-type ASCVD --data-type all --model-name SAINT \
  --protein-path features/lassonet/ASCVD/lassonet_protein.csv
```

Downstream manuscript analyses are under `analysis/`. [MAPLE/SuSiE scripts](genetics/maple_susie/README.md) cover outcome preparation, sample-structure estimation, MAPLE, colocalization, and sensitivity analyses.

## Released artifacts

- `features/lassonet/`: selection frequencies and final protein panels.
- `artifacts/models/`: five-fold main-analysis models.
- `artifacts/optuna/`: corresponding Optuna studies.
- `artifacts/rp_saint/`: RP-SAINT weights, preprocessors, Breslow estimators, and validation metadata.

The release contains 360 main-analysis fold models, 72 Optuna databases, and 15 RP-SAINT fold models. RP-SAINT predictions reproduce the historical results to floating-point precision. Binary artifacts are managed with Git LFS.

## Validation

```bash
python -m pytest -q
python -m tools.validate_release_runtime --artifact-dir artifacts
```

Citation metadata are provided in `CITATION.cff`. The code is released under the MIT License.
