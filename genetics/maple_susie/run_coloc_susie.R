# Fine-map both traits and compare signal pairs with coloc.susie.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
source(file.path(dirname(script), "coloc_helpers.R"))
opts <- read_options()
require_packages(c("data.table", "coloc", "susieR"))
library(data.table)
set.seed(opts$seed)
maple <- fread(file.path(opts$maple_root, opts$outcome_id, "MAPLE_all_results.tsv"))
proteins <- select_proteins(maple$protein, opts)
out_dir <- file.path(opts$main_root, opts$outcome)
partial <- file.path(out_dir, paste0(opts$outcome, "_coloc_susie_summary_partial.csv"))
existing <- if (opts$resume && file.exists(partial)) fread(partial) else NULL
rows <- list()
for (p in proteins) {
  message("[coloc.susie] ", opts$outcome, " / ", p)
  # Resume only successful fits; errors/previously skipped regions are retried.
  old <- if (!is.null(existing)) existing[protein == p & status %in% c("ok", "ok_no_signal_pair")] else NULL
  rows[[p]] <- if (!is.null(old) && nrow(old) > 0L) tail(old, 1L) else
    tryCatch(fit_pair(opts, p, out_dir), error = function(e) pair_error(opts, p, e))
  write_table(rbindlist(rows, fill = TRUE), partial)
}
summary <- rbindlist(rows, fill = TRUE)
write_coloc_summary(summary, out_dir, paste0(opts$outcome, "_coloc_susie_summary"))
write_table(data.table(p1 = 1e-4, p2 = 1e-4, p12 = 1e-5, L = 10L, coverage = 0.95,
                       maxit = 1000L, max_snps = opts$max_snps, seed = opts$seed),
             file.path(out_dir, "parameters.csv"))
capture.output(sessionInfo(), file = file.path(out_dir, "R_sessionInfo.txt"))
if (any(summary$status == "error")) stop("Some pairs failed; inspect the summary.")
