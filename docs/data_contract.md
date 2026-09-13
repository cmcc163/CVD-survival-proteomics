# Data contract

The code expects one endpoint-specific CSV for each of `Total_CVD`, `ASCVD`,
and `HF`. Participant-level data are not distributed.

Required administrative columns are `eid`, `Ethnic`, and `Is_Incident`. The
file must contain either `time` in days or both `baseline_date` and
`Event_Date`. The 11 PREVENT-aligned clinical predictors are:

`age`, `sex`, `ever_smoked`, `Diabetes_baseline_1`,
`Cholesterol_treatment`, `hdl_cholesterol`, `non_hdl_cholesterol`,
`hypertension_treatment`, `average_SBP`, `eGFR_SCysC`, and `BMI`.

If `non_hdl_cholesterol` is absent, it is calculated from
`total_cholesterol - hdl_cholesterol`. Protein columns must be listed in a
separate CSV. LassoNet panel files use a `Protein_Name` column.

Ancestry codes used by the original analysis are defined in
`utils/load_data.py`. Preprocessing objects are fitted inside each European
training fold. Hold-out and non-European cohorts are transformed only with the
corresponding fitted training-fold transformer.

