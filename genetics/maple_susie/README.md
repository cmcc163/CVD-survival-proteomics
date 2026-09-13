# MAPLE and SuSiE analysis placeholder

The workstation implementation for instrument construction, MAPLE estimation,
LD-matrix preparation, SuSiE fine-mapping, and colocalization is not currently
available in this workspace. Add those scripts here without copying licensed
GWAS, pQTL, or LD-reference data.

Expected public stages are:

1. Select cis-pQTL instruments using the manuscript thresholds.
2. Harmonize exposure and outcome alleles.
3. Build protein-specific European LD matrices.
4. Run MAPLE with 10,000 Gibbs iterations and 20% burn-in.
5. Apply within-outcome Benjamini-Hochberg correction.
6. Run SuSiE fine-mapping and colocalization with the reported priors.
7. Export the source tables consumed by `analysis/genetics/`.

