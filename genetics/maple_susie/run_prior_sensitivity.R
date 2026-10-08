# Reuse saved SuSiE fits, varying only the shared-variant prior p12.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
source(file.path(dirname(script), "coloc_helpers.R"))
opts <- read_options()
require_packages(c("data.table", "coloc"))
library(data.table)
main_dir <- file.path(opts$main_root, opts$outcome)
main <- fread(file.path(main_dir, paste0(opts$outcome, "_coloc_susie_summary.csv")))
main <- main[status %in% c("ok", "ok_no_signal_pair")]
proteins <- select_proteins(main$protein, opts)
rows <- list()
for (p in proteins) {
  base <- main[protein == p][1]
  prefix <- paste0(safe_name(p), "_", opts$outcome)
  for (prior in c(1e-6, 1e-5, 1e-4)) {
    result <- tryCatch({
      s1 <- readRDS(file.path(main_dir, "susie_objects", paste0(prefix, "_exposure_susie.rds")))
      s2 <- readRDS(file.path(main_dir, "susie_objects", paste0(prefix, "_outcome_susie.rds")))
      res <- coloc::coloc.susie(s1, s2, p1 = 1e-4, p2 = 1e-4, p12 = prior)
      summarise_coloc(res, opts$outcome, opts$outcome_id, p, base$n_harmonized, base$nsnps, s1, s2)
    }, error = function(e) pair_error(opts, p, e))
    result[, `:=`(p1 = 1e-4, p2 = 1e-4, p12 = prior, original_status = base$status)]
    rows[[length(rows) + 1L]] <- result
  }
}
summary <- rbindlist(rows, fill = TRUE)
setorder(summary, protein, p12)
root <- file.path(opts$sensitivity_root, "prior_sensitivity", opts$outcome)
write_coloc_summary(summary, root, "prior_sensitivity_summary")
if (any(summary$status == "error")) stop("Some prior comparisons failed; inspect the summary.")
