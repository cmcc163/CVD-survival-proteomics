# Downstream manuscript analyses

These scripts reproduce the cohort description, discrimination, calibration,
decision-curve, interpretation, genetics-integration, interaction, and reduced-
panel analyses used by the manuscript. They contain no participant data or
generated results.

Python scripts accept input and output locations through their command-line
arguments. R scripts use the documented environment variables near the top of
each file; all defaults are repository-relative. Set `BOOTSTRAP_B` to override
the manuscript default of 1,000 bootstrap replicates.

Run modelling stages through `run.py`. Run downstream scripts only after the
required prediction or attribution files have been generated. The MAPLE/SuSiE
core execution code is intentionally represented by a placeholder until the
workstation implementation can be added.
