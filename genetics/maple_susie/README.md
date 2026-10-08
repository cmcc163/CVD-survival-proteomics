# MAPLE and SuSiE colocalization

Core R workflow for protein pQTL and cardiovascular GWAS summary statistics.

| Script | Purpose |
| --- | --- |
| `prepare_outcome.R` | Match local GWAS SNPs to protein exposure SNPs. |
| `estimate_omega.R` | Estimate sample-structure parameters using `MAPLE::est_SS`. |
| `run_maple.R` | Harmonize effects, match LD, run MAPLE, and apply BH/Bonferroni correction. |
| `run_coloc_susie.R` | Fine-map both traits and calculate signal-pair colocalization probabilities. |
| `run_prior_sensitivity.R` | Repeat colocalization with `p12 = 1e-6, 1e-5, 1e-4`. |
| `run_coverage_sensitivity.R` | Retry excluded pairs with expanded SNP/signal limits. |
| `finalize_results.R` | Combine results, retaining the source and failure/skip status. |

## Run

Run from the repository root. Use `--help` for path overrides; no source-file edits are needed.

```bash
Rscript genetics/maple_susie/prepare_outcome.R --data-root /path/to/summary_data --outcome HF --exposure /path/to/pqtl.csv
Rscript genetics/maple_susie/run_maple.R --data-root /path/to/summary_data --outcome HF --exposure /path/to/pqtl.csv --plink /path/to/plink --bfile /path/to/EUR
Rscript genetics/maple_susie/run_coloc_susie.R --outcome HF --plink /path/to/plink --bfile /path/to/EUR
Rscript genetics/maple_susie/run_prior_sensitivity.R --outcome HF
Rscript genetics/maple_susie/run_coverage_sensitivity.R --outcome HF
Rscript genetics/maple_susie/finalize_results.R
```

Repeat outcome-specific steps for CAD, Stroke, and optionally CHD. GWAS IDs are defined in `common.R`; they are not the survival cohort endpoint labels. Outputs default to `results/genetics/`. For existing analyses, set `--maple-root`, `--ld-root`, and `--main-root`. Use `--proteins AGER,MMP12` to run a subset; `--resume true` reuses successful main fits only. A protein subset produces subset-specific MAPLE multiple-testing corrections.

## Inputs

- Exposure: `SNP`, `id.exposure`, `exposure`, `beta.exposure`, `se.exposure`, `effect_allele.exposure`, `other_allele.exposure`, `eaf.exposure`, `pval.exposure`, `samplesize.exposure`.
- Local outcome: `SNP`, `CHR`, `BP`, `REF`, `ALT`, `BETA`, `SE`, `AF`, `LP` (minus log10 P), `SS`. ALT must be the effect allele for BETA. `prepare_outcome.R` converts these to TwoSampleMR fields and retains genomic positions.
- Omega: `Protein`, `t1`, `t2`, `t12`, `t1_se`, `t2_se`, `t12_se`, `exposure_n_snp`, `outcome_n_snp`. Supply existing estimates with `--omega FILE`, or use `estimate_omega.R` with genome-wide exposure/outcome tables (`SNP`, `b`, `se`, `frq_A1`, `A1`, `A2`, `P`, `N`; exposure also requires `Protein`) and `--ld-scores DIR`. Do not estimate Omega from only the selected cis-region SNPs.
- LD: a local PLINK reference panel, or cached headerless `*_Sigma.tsv` matrices and `*_Sigma_snps.txt` SNP lists. LD must be signed correlation, with SNP order and allele coding consistent with the harmonized effects. The original PLINK allele-order convention is retained; harmonization alone does not verify LD allele coding.

Install R dependencies in `requirements-r.txt`. MAPLE and TwoSampleMR require their upstream packages if unavailable from CRAN. Scripts never install packages automatically.

## Analysis settings and provenance

Refactored from the supplied `MAPLE-run-2.R`, `MAPLE-outcome-2.R`, `MAPLE.R`, and the main/prior/coverage SuSiE scripts. No original file is modified. MAPLE uses 10,000 Gibbs iterations, 20% burn-in, and the original prior hyperparameters. SuSiE uses standardized quantitative protein effects (`sdY = 1`), case-control GWAS effects, `p1 = p2 = 1e-4`, and `p12 = 1e-5`. Main analysis: at most 2,500 SNPs, `L = 10`, coverage 0.95, and 1,000 iterations. Expanded-region sensitivity: at most 5,000 SNPs, `L = 20`, coverage 0.95, and 2,000 iterations. The latter expands region/signal limits, not the credible-set coverage probability.

Failures are recorded and return a nonzero exit status; unprocessed pairs are not labelled as timeouts. `finalize_results.R` uses successful expanded-region fits for previously excluded pairs and records `result_source`. Results from changed inputs or settings should use a fresh output directory.

```bash
Rscript genetics/maple_susie/tests/test_workflow.R
```
