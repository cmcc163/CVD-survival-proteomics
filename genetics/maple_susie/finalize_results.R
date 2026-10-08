# Combine available main/coverage results without inventing timeout statuses.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
source(file.path(dirname(script), "coloc_helpers.R"))
opts <- read_options()
require_packages("data.table")
library(data.table)
rows <- list()
for (outcome in names(outcome_map)) {
  file <- file.path(opts$main_root, outcome, paste0(outcome, "_coloc_susie_summary.csv"))
  if (!file.exists(file)) next
  main <- fread(file)
  coverage_file <- file.path(opts$sensitivity_root, "coverage_sensitivity", outcome, "coverage_sensitivity_summary.csv")
  main[, result_source := "main"]
  if (file.exists(coverage_file)) {
    coverage <- fread(coverage_file)[status %in% c("ok", "ok_no_signal_pair")]
    coverage[, result_source := "coverage_sensitivity"]
    main <- rbindlist(list(main[!protein %in% coverage$protein], coverage), fill = TRUE)
  }
  rows[[outcome]] <- main
}
if (!length(rows)) stop("No outcome summaries found.")
summary <- rbindlist(rows, fill = TRUE)
setorder(summary, outcome, -max_PP.H4)
write_coloc_summary(summary, opts$main_root, "all_outcomes_coloc_susie_summary")
counts <- summary[, .(n_pairs = .N, n_ok = sum(status == "ok"),
                      n_no_signal_pair = sum(status == "ok_no_signal_pair"),
                      n_skipped = sum(grepl("^skipped|^not_run", status)), n_error = sum(status == "error"),
                      n_h4_ge_0_8 = sum(status == "ok" & max_PP.H4 >= 0.8, na.rm = TRUE)), by = outcome]
write_table(counts, file.path(opts$main_root, "all_outcomes_coloc_susie_counts.csv"))
print(counts)
