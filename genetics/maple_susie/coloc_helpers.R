# SuSiE fitting and posterior summaries shared by main/sensitivity analyses.
make_dataset <- function(dat, Sigma, trait) {
  n <- median(dat[[paste0("samplesize.", trait)]], na.rm = TRUE)
  if (!is.finite(n) || n <= 0) stop("Invalid sample size for ", trait)
  result <- list(beta = dat[[paste0("beta.", trait)]], varbeta = dat[[paste0("varbeta.", trait)]],
                 snp = dat$SNP, position = dat$position_coloc, LD = Sigma, N = n,
                 type = if (trait == "exposure") "quant" else "cc")
  if (trait == "exposure") result$sdY <- 1
  result
}
summarise_coloc <- function(res, outcome, outcome_id, protein, n_harmonized, nsnps, s1, s2) {
  signals <- data.table::as.data.table(res$summary)
  posterior <- data.table::as.data.table(res$results)
  result <- data.table::data.table(outcome = outcome, outcome_id = outcome_id, protein = protein,
                                   status = "ok_no_signal_pair", error_message = NA_character_,
                                   n_harmonized = n_harmonized, nsnps = nsnps,
                                   n_exposure_signals = length(s1$sets$cs), n_outcome_signals = length(s2$sets$cs),
                                   n_signal_pairs = nrow(signals), max_PP.H4 = NA_real_, max_PP.H3 = NA_real_,
                                   best_hit1 = NA_character_, best_hit2 = NA_character_,
                                   top_snp_for_best_H4 = NA_character_, top_snp_pp_h4_for_best_H4 = NA_real_)
  if (!nrow(signals)) return(result)
  require_columns(signals, c("PP.H4.abf", "PP.H3.abf", "hit1", "hit2"))
  best <- which.max(signals$PP.H4.abf)
  result[, `:=`(status = "ok", max_PP.H4 = signals$PP.H4.abf[best],
                 max_PP.H3 = max(signals$PP.H3.abf, na.rm = TRUE),
                 best_hit1 = as.character(signals$hit1[best]), best_hit2 = as.character(signals$hit2[best]))]
  cols <- grep("^SNP\\.PP\\.H4\\.row", names(posterior), value = TRUE)
  # coloc uses an unnumbered posterior column for a single signal pair.
  if (nrow(signals) == 1L && "SNP.PP.H4.abf" %in% names(posterior)) cols <- "SNP.PP.H4.abf"
  if (length(cols) >= best && nrow(posterior)) {
    top <- which.max(posterior[[cols[best]]])
    result[, `:=`(top_snp_for_best_H4 = as.character(posterior$snp[top]),
                   top_snp_pp_h4_for_best_H4 = posterior[[cols[best]]][top])]
  }
  result
}
fit_pair <- function(opts, protein, out_dir, max_snps = opts$max_snps,
                     L = 10L, coverage = 0.95, maxit = 1000L) {
  dat <- read_harmonized(opts, protein)
  n_harmonized <- nrow(dat)
  if (n_harmonized < 10L) stop("Too few harmonized SNPs.")
  if (n_harmonized > max_snps) return(data.table::data.table(outcome = opts$outcome,
      outcome_id = opts$outcome_id, protein = protein, status = "skipped_large_region",
      error_message = paste("SNP count exceeds", max_snps), n_harmonized = n_harmonized,
      nsnps = n_harmonized, max_PP.H4 = NA_real_, max_PP.H3 = NA_real_))
  ld <- load_ld(opts, protein, dat$SNP)
  matched <- align_ld(dat, ld)
  dat <- matched$dat
  d1 <- make_dataset(dat, matched$Sigma, "exposure")
  d2 <- make_dataset(dat, matched$Sigma, "outcome")
  coloc::check_dataset(d1, req = "LD")
  coloc::check_dataset(d2, req = "LD")
  s1 <- coloc::runsusie(d1, L = L, coverage = coverage, maxit = maxit)
  s2 <- coloc::runsusie(d2, L = L, coverage = coverage, maxit = maxit)
  res <- coloc::coloc.susie(s1, s2, p1 = 1e-4, p2 = 1e-4, p12 = 1e-5)
  prefix <- paste0(safe_name(protein), "_", opts$outcome)
  for (folder in c("harmonized", "susie_objects", "snp_posterior"))
    dir.create(file.path(out_dir, folder), recursive = TRUE, showWarnings = FALSE)
  write_table(dat, file.path(out_dir, "harmonized", paste0(prefix, "_susie_input.tsv.gz")))
  saveRDS(s1, file.path(out_dir, "susie_objects", paste0(prefix, "_exposure_susie.rds")))
  saveRDS(s2, file.path(out_dir, "susie_objects", paste0(prefix, "_outcome_susie.rds")))
  write_table(data.table::as.data.table(res$summary), file.path(out_dir, paste0(prefix, "_coloc_susie_signal_summary.csv")))
  if (!is.null(res$results)) write_table(data.table::as.data.table(res$results),
      file.path(out_dir, "snp_posterior", paste0(prefix, "_coloc_susie_snp_posterior.tsv.gz")))
  summary <- summarise_coloc(res, opts$outcome, opts$outcome_id, protein, n_harmonized, nrow(dat), s1, s2)
  summary[, ld_source := ld$source]
  summary
}
pair_error <- function(opts, protein, e) data.table::data.table(outcome = opts$outcome,
    outcome_id = opts$outcome_id, protein = protein, status = "error",
    error_message = conditionMessage(e), max_PP.H4 = NA_real_, max_PP.H3 = NA_real_)
write_coloc_summary <- function(summary, root, stem) {
  write_table(summary, file.path(root, paste0(stem, ".csv")))
  for (threshold in c(0.50, 0.80)) write_table(summary[status == "ok" & max_PP.H4 >= threshold],
      file.path(root, paste0(stem, "_hits_H4_ge_", sprintf("%.2f", threshold), ".csv")))
}
