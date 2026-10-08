# Harmonize exposure/outcome effects, match signed LD, and run MAPLE per protein.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
opts <- read_options()
require_packages(c("data.table", "TwoSampleMR", "MAPLE"))
library(data.table)
set.seed(opts$seed)
exp <- fread(opts$exposure)
outcome_file <- file.path(opts$output_root, "outcome_data_by_protein", opts$outcome_id,
                          paste0("all_proteins_outcome_", opts$outcome_id, ".csv"))
out <- fread(outcome_file)
omega <- fread(opts$omega)
require_columns(exp, c("SNP", "beta.exposure", "se.exposure", "effect_allele.exposure",
                       "other_allele.exposure", "eaf.exposure", "pval.exposure",
                       "samplesize.exposure", "exposure", "id.exposure"))
require_columns(out, c("SNP", "beta.outcome", "se.outcome", "effect_allele.outcome",
                       "other_allele.outcome", "samplesize.outcome", "id.outcome", "protein_id"))
exp <- exp[, intersect(names(exp), c("SNP", "exposure", grep("\\.exposure$", names(exp), value = TRUE))), with = FALSE]
out <- out[, intersect(names(out), c("SNP", "outcome", "protein_id", grep("\\.outcome$", names(out), value = TRUE))), with = FALSE]
require_columns(omega, c("Protein", "t1", "t2", "t12", "t1_se", "t2_se", "t12_se",
                         "exposure_n_snp", "outcome_n_snp"))
if (anyDuplicated(omega$Protein)) stop("Duplicate Protein in Omega table.")
if (any(!is.finite(as.matrix(omega[, .(t1, t2, t12)])))) stop("Non-finite Omega parameters.")
if (any(as.character(out$id.outcome) != opts$outcome_id)) stop("Outcome ID mismatch.")
proteins <- select_proteins(Reduce(intersect, list(exp$id.exposure, out$protein_id, omega$Protein)), opts)
res_dir <- file.path(opts$maple_root, opts$outcome_id)
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

run_one <- function(p) {
  x <- exp[id.exposure == p]
  y <- out[protein_id == p]
  if (anyDuplicated(x$SNP) || anyDuplicated(y$SNP)) stop("Duplicate SNP within protein.")
  shared <- intersect(x$SNP, y$SNP)
  dat <- as.data.table(TwoSampleMR::harmonise_data(as.data.frame(x[SNP %in% shared]),
                                                 as.data.frame(y[SNP %in% shared]), action = 2))
  dat <- dat[mr_keep == TRUE & is.finite(beta.exposure) & is.finite(beta.outcome) &
               is.finite(se.exposure) & is.finite(se.outcome) & se.exposure > 0 & se.outcome > 0]
  if (nrow(dat) < 3L) stop("Too few harmonized SNPs.")
  matched <- align_ld(dat, load_ld(opts, p, dat$SNP), minimum = 3L)
  dat <- matched$dat
  dat[, `:=`(Zscore_1 = beta.exposure / se.exposure, Zscore_2 = beta.outcome / se.outcome)]
  n1 <- median(dat$samplesize.exposure, na.rm = TRUE)
  n2 <- median(dat$samplesize.outcome, na.rm = TRUE)
  if (!is.finite(n1) || !is.finite(n2) || n1 <= 0 || n2 <= 0) stop("Invalid sample size.")
  w <- omega[Protein == p]
  write_table(dat, file.path(res_dir, paste0(safe_name(p), "_harmonized_for_MAPLE.tsv")))
  # Keep the manuscript Gibbs sampler and prior hyperparameters unchanged.
  fit <- MAPLE::MAPLE(dat$Zscore_1, dat$Zscore_2, matched$Sigma, matched$Sigma, n1, n2,
                      Gibbsnumber = opts$gibbs, burninproportion = 0.2,
                      pi_beta_shape = 0.5, pi_beta_scale = 4.5,
                      pi_c_shape = 0.5, pi_c_scale = 9.5,
                      pi_1_shape = 0.5, pi_1_scale = 1.5,
                      pi_0_shape = 0.05, pi_0_scale = 9.95,
                      t1 = w$t1, t2 = w$t2, t12 = w$t12)
  saveRDS(fit, file.path(res_dir, paste0(safe_name(p), "_MAPLE_result.rds")))
  data.table(protein = p, exposure = dat$exposure[1], outcome_id = opts$outcome_id,
             outcome_name = opts$outcome, nsnp = nrow(dat), samplen1 = n1, samplen2 = n2,
             t1 = w$t1, t2 = w$t2, t12 = w$t12, t1_se = w$t1_se, t2_se = w$t2_se,
             t12_se = w$t12_se, omega_exposure_n_snp = w$exposure_n_snp,
             omega_outcome_n_snp = w$outcome_n_snp,
             causal_effect = fit$causal_effect, causal_pvalue = fit$causal_pvalue,
             cause_se = fit$cause.se, correlated_pleiotropy_effect = fit$correlated_pleiotropy_effect,
             sigmaeta = fit$sigmaeta, sigmabeta = fit$sigmabeta)
}
rows <- list()
errors <- list()
for (p in proteins) {
  message("[MAPLE] ", opts$outcome, " / ", p)
  result <- tryCatch(run_one(p), error = function(e) {
    message("[MAPLE error] ", p, ": ", conditionMessage(e))
    errors[[p]] <<- data.table(protein = p, outcome_id = opts$outcome_id, error = conditionMessage(e))
    NULL
  })
  if (!is.null(result)) rows[[p]] <- result
  if (length(rows)) write_table(rbindlist(rows), file.path(res_dir, "MAPLE_all_results_partial.tsv"))
}
if (length(errors)) write_table(rbindlist(errors), file.path(res_dir, "MAPLE_error_log.tsv"))
if (!length(rows)) stop("No successful MAPLE results; inspect MAPLE_error_log.tsv.")
res <- rbindlist(rows)
res[, `:=`(FDR = p.adjust(causal_pvalue, "BH"), Bonferroni = p.adjust(causal_pvalue, "bonferroni"),
           Direction = fifelse(causal_effect > 0, "Risk", fifelse(causal_effect < 0, "Protective", "Null")))]
res[, `:=`(Significant_FDR = fifelse(FDR < 0.05, "Yes", "No"),
           Significant_Bonferroni = fifelse(Bonferroni < 0.05, "Yes", "No"))]
setorder(res, causal_pvalue)
write_table(res, file.path(res_dir, "MAPLE_all_results.tsv"))
capture.output(sessionInfo(), file = file.path(res_dir, "R_sessionInfo.txt"))
if (length(errors)) stop("MAPLE completed with failed proteins; inspect MAPLE_error_log.tsv.")
