# Retry excluded/failed regions using the historical expanded SuSiE settings.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
source(file.path(dirname(script), "coloc_helpers.R"))
opts <- read_options()
require_packages(c("data.table", "coloc", "susieR"))
library(data.table)
set.seed(opts$seed)
main <- fread(file.path(opts$main_root, opts$outcome, paste0(opts$outcome, "_coloc_susie_summary.csv")))
targets <- main[status %in% c("skipped_large_region", "skipped_timeout", "not_run", "error")]
if (!nrow(targets)) { message("No excluded pairs to retry."); quit(status = 0) }
proteins <- select_proteins(targets$protein, opts)
root <- file.path(opts$sensitivity_root, "coverage_sensitivity", opts$outcome)
rows <- list()
for (p in proteins) {
  result <- tryCatch(fit_pair(opts, p, root, max_snps = 5000L, L = 20L, coverage = 0.95, maxit = 2000L),
                     error = function(e) pair_error(opts, p, e))
  result[, `:=`(original_status = targets[protein == p, status][1], max_snps = 5000L,
                 susie_L = 20L, susie_coverage = 0.95, susie_maxit = 2000L)]
  rows[[p]] <- result
  write_table(rbindlist(rows, fill = TRUE), file.path(root, "coverage_sensitivity_summary_partial.csv"))
}
summary <- rbindlist(rows, fill = TRUE)
write_coloc_summary(summary, root, "coverage_sensitivity_summary")
if (any(summary$status == "error")) stop("Some coverage comparisons failed; inspect the summary.")
