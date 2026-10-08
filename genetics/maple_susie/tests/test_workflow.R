# Synthetic checks only; all generated files remain in an R temporary directory.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
module <- file.path(dirname(script), "..")
source(file.path(module, "common.R"))
source(file.path(module, "coloc_helpers.R"))
require_packages(c("data.table", "coloc", "susieR"))
library(data.table)
for (file in list.files(module, pattern = "\\.R$", full.names = TRUE)) parse(file)
set.seed(10)
n <- 40L
snps <- paste0("rs", seq_len(n))
dat <- data.table(SNP = snps, beta.exposure = c(0.6, rep(0, n - 1)),
                  se.exposure = 0.04, beta.outcome = c(0.3, rep(0, n - 1)), se.outcome = 0.035,
                  samplesize.exposure = 20000, samplesize.outcome = 50000,
                  pos.outcome = seq_len(n))
root <- tempfile("maple_susie_")
opts <- list(maple_root = file.path(root, "MAPLE_results"), ld_root = file.path(root, "ld"),
             outcome = "HF", outcome_id = outcome_map[["HF"]], max_snps = 2500L)
write_table(dat, file.path(opts$maple_root, opts$outcome_id, "DEMO_harmonized_for_MAPLE.tsv"))
ld_dir <- file.path(opts$ld_root, opts$outcome_id)
dir.create(ld_dir, recursive = TRUE)
prefix <- file.path(ld_dir, paste0("DEMO_", opts$outcome_id))
fwrite(as.data.table(diag(n)), paste0(prefix, "_Sigma.tsv"), col.names = FALSE, sep = "\t")
fwrite(data.table(SNP = rev(snps)), paste0(prefix, "_Sigma_snps.txt"), col.names = FALSE)
prepared <- read_harmonized(opts, "DEMO")
matched <- align_ld(prepared, load_ld(opts, "DEMO", prepared$SNP))
stopifnot(identical(matched$dat$SNP, rev(snps)), identical(rownames(matched$Sigma), rev(snps)))
result <- fit_pair(opts, "DEMO", file.path(root, "coloc"))
stopifnot(result$status == "ok", is.finite(result$max_PP.H4), result$max_PP.H4 > 0.8,
          result$top_snp_for_best_H4 == "rs1")
skipped <- fit_pair(opts, "DEMO", file.path(root, "coloc"), max_snps = 20L)
stopifnot(skipped$status == "skipped_large_region")
empty <- summarise_coloc(list(summary = NULL, results = NULL), "HF", opts$outcome_id,
                         "NONE", n, n, list(sets = list(cs = NULL)), list(sets = list(cs = NULL)))
stopifnot(empty$status == "ok_no_signal_pair", is.na(empty$max_PP.H4))
fwrite(data.table(SNP = c(snps[-n], snps[1])), paste0(prefix, "_Sigma_snps.txt"), col.names = FALSE)
stopifnot(inherits(try(load_ld(opts, "DEMO", prepared$SNP), silent = TRUE), "try-error"))
fwrite(data.table(SNP = rev(snps)), paste0(prefix, "_Sigma_snps.txt"), col.names = FALSE)

# Exercise the released command-line stages with synthetic summary statistics.
exposure <- copy(dat)
exposure[, `:=`(id.exposure = "DEMO", exposure = "DEMO",
                effect_allele.exposure = "A", other_allele.exposure = "G",
                eaf.exposure = 0.3, pval.exposure = 2 * pnorm(-abs(beta.exposure / se.exposure)))]
raw <- dat[, .(SNP, CHR = 1L, BP = pos.outcome, REF = "G", ALT = "A",
               BETA = beta.outcome, SE = se.outcome, AF = 0.3,
               LP = -log10(2 * pnorm(-abs(beta.outcome / se.outcome))), SS = samplesize.outcome)]
omega <- data.table(Protein = "DEMO", t1 = 1, t2 = 1, t12 = 0, t1_se = 0.01,
                    t2_se = 0.01, t12_se = 0.01, exposure_n_snp = n, outcome_n_snp = n)
x <- file.path(root, "exposure.csv")
y <- file.path(root, "outcome.tsv")
w <- file.path(root, "omega.tsv")
write_table(exposure, x); write_table(raw, y); write_table(omega, w)
rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
run_stage <- function(stage, args) {
  output <- system2(rscript, c(shQuote(file.path(module, stage)), shQuote(args)), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) stop(paste(output, collapse = "\n"))
}
base_args <- c("--output-root", root, "--outcome", "HF")
run_stage("prepare_outcome.R", c(base_args, "--exposure", x, "--outcome-file", y))
if (all(vapply(c("MAPLE", "TwoSampleMR"), requireNamespace, logical(1), quietly = TRUE))) {
  run_stage("run_maple.R", c(base_args, "--exposure", x, "--omega", w, "--ld-root", opts$ld_root,
                            "--gibbs", "100"))
  maple <- fread(file.path(root, "MAPLE_results", opts$outcome_id, "MAPLE_all_results.tsv"))
  stopifnot(nrow(maple) == 1L, maple$protein == "DEMO", is.finite(maple$causal_effect))
  run_stage("run_coloc_susie.R", c(base_args, "--ld-root", opts$ld_root))
  run_stage("run_coloc_susie.R", c(base_args, "--ld-root", opts$ld_root, "--resume", "true"))
  run_stage("run_prior_sensitivity.R", base_args)
  run_stage("run_coverage_sensitivity.R", c(base_args, "--ld-root", opts$ld_root))
  main_file <- file.path(root, "coloc_susie", "HF", "HF_coloc_susie_summary.csv")
  main <- fread(main_file)
  main[, status := "skipped_large_region"]
  write_table(main, main_file)
  run_stage("run_coverage_sensitivity.R", c(base_args, "--ld-root", opts$ld_root))
  run_stage("finalize_results.R", base_args)
  prior <- fread(file.path(root, "coloc_susie_sensitivity", "prior_sensitivity", "HF", "prior_sensitivity_summary.csv"))
  stopifnot(nrow(prior) == 3L, all(prior$status == "ok"))
  coverage <- fread(file.path(root, "coloc_susie_sensitivity", "coverage_sensitivity", "HF", "coverage_sensitivity_summary.csv"))
  final <- fread(file.path(root, "coloc_susie", "all_outcomes_coloc_susie_summary.csv"))
  stopifnot(coverage$status == "ok", coverage$susie_L == 20L,
            coverage$max_snps == 5000L, final$result_source == "coverage_sensitivity")
  message("PASS: synthetic outcome -> MAPLE -> SuSiE -> resume -> sensitivity -> finalization.")
} else message("SKIP: full MAPLE CLI test requires MAPLE and TwoSampleMR.")
message("PASS: R parsing, cached LD ordering/validation, synthetic SuSiE colocalization, SNP limit, no-signal summary.")
