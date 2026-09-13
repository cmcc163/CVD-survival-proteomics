options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(openxlsx)
  library(parallel)
  library(ggplot2)
})

base_predictions <- Sys.getenv("CVD_PREDICTION_ROOT", unset = "artifacts/predictions")
base_dir <- Sys.getenv("CVD_PERFORMANCE_RESULTS", unset = "results/performance")
eval_dir <- file.path(base_dir, "evaluation_performance")
out_dir <- file.path(eval_dir, "bootstrap_model")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c("Total_CVD" = "Total CVD", "ASCVD" = "ASCVD", "HF" = "HF")

models <- c(
  "PREVENT", "LinearModel", "XGBoost", "MLP", "TabNet",
  "NODE", "FT-Transformer", "SAINT", "TabPFN"
)
model_labels <- c(
  "PREVENT" = "PREVENT equation",
  "LinearModel" = "Refitted Cox model",
  "XGBoost" = "XGBoost",
  "MLP" = "MLP",
  "TabNet" = "TabNet",
  "NODE" = "NODE",
  "FT-Transformer" = "FT-Transformer",
  "SAINT" = "SAINT",
  "TabPFN" = "TabPFN"
)

feature_files <- list(
  "Clinical predictors only" = "classic_val.csv",
  "Protein markers only" = "protein_lassonet_val.csv",
  "Clinical predictors plus protein markers" = "all_lassonet_val.csv"
)
feature_levels <- names(feature_files)

metric_levels <- c("C-index", "OE_low", "OE_high", "ICI_all", "NB_10pct")
target_time <- 3652.5
threshold_nb <- 0.10
B <- as.integer(Sys.getenv("BOOTSTRAP_B", "1000"))
seed0 <- 20260701

clip_prob <- function(p) {
  pmin(pmax(as.numeric(p), 1e-6), 1 - 1e-6)
}

load_prediction <- function(model, outcome, feature_set) {
  if (model == "PREVENT" && feature_set != "Clinical predictors only") {
    return(NULL)
  }
  file_path <- file.path(base_predictions, model, outcome, feature_files[[feature_set]])
  if (!file.exists(file_path)) {
    return(NULL)
  }
  dat <- read.csv(file_path)
  needed <- c("time", "event", "risk_score", "prob_10yr")
  if (!all(needed %in% names(dat))) {
    return(NULL)
  }
  dat <- dat[, needed]
  dat$time <- as.numeric(dat$time)
  dat$event <- as.integer(dat$event)
  dat$risk_score <- as.numeric(dat$risk_score)
  dat$prob_10yr <- clip_prob(dat$prob_10yr)
  dat <- dat[complete.cases(dat), ]
  dat
}

c_index_metric <- function(dat) {
  if (nrow(dat) < 2 || sum(dat$event == 1, na.rm = TRUE) == 0) {
    return(NA_real_)
  }
  out <- tryCatch(
    concordance(Surv(time, event) ~ risk_score, data = dat, reverse = TRUE)$concordance,
    error = function(e) NA_real_
  )
  as.numeric(out)
}

observed_risk_km <- function(dat, t0) {
  if (nrow(dat) == 0 || sum(dat$event == 1, na.rm = TRUE) == 0) {
    return(NA_real_)
  }
  km <- tryCatch(summary(survfit(Surv(time, event) ~ 1, data = dat), times = t0), error = function(e) NULL)
  if (is.null(km) || length(km$surv) == 0) {
    return(NA_real_)
  }
  1 - km$surv[1]
}

oe_metric <- function(dat) {
  expected <- mean(dat$prob_10yr, na.rm = TRUE)
  observed <- observed_risk_km(dat, target_time)
  if (is.na(observed) || is.na(expected) || expected <= 0) {
    return(NA_real_)
  }
  observed / expected
}

split_low_high <- function(dat) {
  if (nrow(dat) < 2) {
    dat$risk_group <- NA_integer_
    return(dat)
  }
  dat <- dat[order(dat$prob_10yr), , drop = FALSE]
  dat$risk_group <- dplyr::ntile(dat$prob_10yr, 2)
  dat
}

ici_metric <- function(dat) {
  if (nrow(dat) < 5 || sum(dat$event == 1, na.rm = TRUE) == 0) {
    return(NA_real_)
  }
  probs <- clip_prob(dat$prob_10yr)
  dat$LP <- log(-log(1 - probs))
  fit <- tryCatch(
    suppressWarnings(coxph(Surv(time, event) ~ LP, data = dat)),
    error = function(e) NULL
  )
  if (is.null(fit) || any(is.na(coef(fit)))) {
    return(NA_real_)
  }
  bh <- tryCatch(basehaz(fit, centered = FALSE), error = function(e) NULL)
  if (is.null(bh) || nrow(bh) == 0) {
    return(NA_real_)
  }
  bh_t <- bh[bh$time <= target_time, , drop = FALSE]
  h0_t <- if (nrow(bh_t) == 0) 0 else tail(bh_t$hazard, 1)
  beta <- unname(coef(fit)[1])
  calibrated <- 1 - exp(-h0_t * exp(beta * dat$LP))
  mean(abs(probs - calibrated), na.rm = TRUE)
}

censoring_weights <- function(dat, t0) {
  censor_event <- 1L - dat$event
  fit <- tryCatch(survfit(Surv(time, censor_event) ~ 1, data = dat), error = function(e) NULL)
  if (is.null(fit)) {
    return(NULL)
  }
  km <- summary(fit)
  get_g <- function(times) {
    if (length(km$time) == 0) {
      return(rep(1, length(times)))
    }
    idx <- findInterval(times, km$time)
    surv <- ifelse(idx == 0, 1, km$surv[idx])
    pmax(surv, 1e-6)
  }
  case <- dat$event == 1 & dat$time <= t0
  control <- dat$time > t0
  weights <- rep(0, nrow(dat))
  weights[case] <- 1 / get_g(pmax(dat$time[case] - 1e-8, 0))
  weights[control] <- 1 / get_g(t0)
  list(weights = weights, case = case, control = control)
}

net_benefit_metric <- function(dat, threshold = threshold_nb) {
  if (nrow(dat) == 0) {
    return(NA_real_)
  }
  cw <- censoring_weights(dat, target_time)
  if (is.null(cw)) {
    return(NA_real_)
  }
  treat <- dat$prob_10yr >= threshold
  n <- nrow(dat)
  tp_rate <- sum(cw$weights[cw$case & treat]) / n
  fp_rate <- sum(cw$weights[cw$control & treat]) / n
  tp_rate - fp_rate * threshold / (1 - threshold)
}

calc_model_metrics <- function(dat) {
  dat <- dat[complete.cases(dat[, c("time", "event", "risk_score", "prob_10yr")]), , drop = FALSE]
  cidx <- c_index_metric(dat)
  grouped <- split_low_high(dat)
  oe_low <- oe_metric(grouped[grouped$risk_group == 1, , drop = FALSE])
  oe_high <- oe_metric(grouped[grouped$risk_group == 2, , drop = FALSE])
  ici <- ici_metric(dat)
  nb <- net_benefit_metric(dat)

  raw <- c(
    "C-index" = cidx,
    "OE_low" = oe_low,
    "OE_high" = oe_high,
    "ICI_all" = ici,
    "NB_10pct" = nb
  )
  score <- c(
    "C-index" = cidx,
    "OE_low" = -abs(log(oe_low)),
    "OE_high" = -abs(log(oe_high)),
    "ICI_all" = -ici,
    "NB_10pct" = nb
  )
  list(raw = raw, score = score)
}

make_task_manifest <- function() {
  expand.grid(
    Outcome = outcomes,
    FeatureSet = feature_levels,
    stringsAsFactors = FALSE
  )
}

check_alignment <- function(data_list) {
  ref <- data_list[["SAINT"]]
  warnings <- c()
  for (model in names(data_list)) {
    dat <- data_list[[model]]
    ok <- nrow(dat) == nrow(ref) &&
      all(dat$time == ref$time, na.rm = TRUE) &&
      all(dat$event == ref$event, na.rm = TRUE)
    if (!ok) {
      warnings <- c(warnings, paste0(model, " is not row-aligned with SAINT"))
    }
  }
  warnings
}

run_one_task <- function(task_index, tasks) {
  set.seed(seed0 + task_index)
  task <- tasks[task_index, ]
  feature_set <- task$FeatureSet
  outcome <- task$Outcome

  available_models <- models
  if (feature_set != "Clinical predictors only") {
    available_models <- setdiff(available_models, "PREVENT")
  }

  data_list <- list()
  file_manifest <- data.frame()
  for (model in available_models) {
    dat <- load_prediction(model, outcome, feature_set)
    if (!is.null(dat) && nrow(dat) > 0) {
      data_list[[model]] <- dat
      file_manifest <- bind_rows(
        file_manifest,
        data.frame(
          outcome = outcome,
          outcome_label = unname(outcome_labels[outcome]),
          feature_set = feature_set,
          model = model,
          model_label = unname(model_labels[model]),
          n = nrow(dat),
          events_total = sum(dat$event == 1, na.rm = TRUE),
          events_10yr = sum(dat$event == 1 & dat$time <= target_time, na.rm = TRUE),
          stringsAsFactors = FALSE
        )
      )
    }
  }

  if (!"SAINT" %in% names(data_list)) {
    stop(paste("SAINT predictions missing for", outcome, feature_set))
  }

  alignment_warnings <- check_alignment(data_list)
  n <- nrow(data_list[["SAINT"]])

  observed_rows <- list()
  observed_scores <- list()
  observed_raw <- list()
  for (model in names(data_list)) {
    metrics <- calc_model_metrics(data_list[[model]])
    observed_scores[[model]] <- metrics$score
    observed_raw[[model]] <- metrics$raw
    observed_rows[[model]] <- data.frame(
      outcome = outcome,
      outcome_label = unname(outcome_labels[outcome]),
      feature_set = feature_set,
      model = model,
      model_label = unname(model_labels[model]),
      metric = metric_levels,
      raw_value = as.numeric(metrics$raw[metric_levels]),
      score = as.numeric(metrics$score[metric_levels]),
      stringsAsFactors = FALSE
    )
  }
  observed_model_metrics <- bind_rows(observed_rows)

  comparator_models <- setdiff(names(data_list), "SAINT")
  diff_array <- array(
    NA_real_,
    dim = c(B, length(comparator_models), length(metric_levels)),
    dimnames = list(NULL, comparator_models, metric_levels)
  )

  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    boot_scores <- list()
    for (model in names(data_list)) {
      boot_scores[[model]] <- calc_model_metrics(data_list[[model]][idx, , drop = FALSE])$score
    }
    saint_score <- boot_scores[["SAINT"]]
    for (comp in comparator_models) {
      diff_array[b, comp, ] <- saint_score[metric_levels] - boot_scores[[comp]][metric_levels]
    }
  }

  summary_rows <- list()
  dist_rows <- list()
  row_id <- 1
  dist_id <- 1
  for (comp in comparator_models) {
    for (metric in metric_levels) {
      diffs <- diff_array[, comp, metric]
      diffs <- diffs[is.finite(diffs)]
      valid_b <- length(diffs)
      observed_diff <- observed_scores[["SAINT"]][metric] - observed_scores[[comp]][metric]
      ci <- if (valid_b > 0) quantile(diffs, probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE) else c(NA_real_, NA_real_)
      p_two <- if (valid_b > 0) {
        2 * min((sum(diffs <= 0, na.rm = TRUE) + 1) / (valid_b + 1), (sum(diffs >= 0, na.rm = TRUE) + 1) / (valid_b + 1))
      } else {
        NA_real_
      }
      p_two <- min(p_two, 1)
      summary_rows[[row_id]] <- data.frame(
        outcome = outcome,
        outcome_label = unname(outcome_labels[outcome]),
        feature_set = feature_set,
        metric = metric,
        saint_model = "SAINT",
        comparator = comp,
        comparator_label = unname(model_labels[comp]),
        observed_saint_raw = observed_raw[["SAINT"]][metric],
        observed_comparator_raw = observed_raw[[comp]][metric],
        observed_saint_score = observed_scores[["SAINT"]][metric],
        observed_comparator_score = observed_scores[[comp]][metric],
        observed_diff = as.numeric(observed_diff),
        boot_mean_diff = mean(diffs, na.rm = TRUE),
        boot_median_diff = median(diffs, na.rm = TRUE),
        ci_low = ci[1],
        ci_high = ci[2],
        p_boot_two_sided = p_two,
        prob_diff_gt0 = mean(diffs > 0, na.rm = TRUE),
        valid_boot = valid_b,
        favors_SAINT = observed_diff > 0,
        ci_excludes_0 = ifelse(is.na(ci[1]) || is.na(ci[2]), NA, ci[1] > 0 || ci[2] < 0),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1

      dist_rows[[dist_id]] <- data.frame(
        outcome = outcome,
        feature_set = feature_set,
        metric = metric,
        comparator = comp,
        bootstrap = seq_along(diffs),
        diff = diffs,
        stringsAsFactors = FALSE
      )
      dist_id <- dist_id + 1
    }
  }

  list(
    summary = bind_rows(summary_rows),
    distribution = bind_rows(dist_rows),
    observed_model_metrics = observed_model_metrics,
    manifest = file_manifest,
    warnings = data.frame(
      outcome = outcome,
      feature_set = feature_set,
      warning = if (length(alignment_warnings) == 0) "All model rows aligned with SAINT" else alignment_warnings,
      stringsAsFactors = FALSE
    )
  )
}

write_outputs <- function(results) {
  pairwise <- bind_rows(lapply(results, `[[`, "summary")) %>%
    mutate(
      metric = factor(metric, levels = metric_levels),
      p_boot_two_sided = pmin(p_boot_two_sided, 1),
      interpretation = case_when(
        observed_diff > 0 & ci_low > 0 ~ "SAINT significantly better by bootstrap CI",
        observed_diff > 0 ~ "SAINT numerically better",
        observed_diff < 0 & ci_high < 0 ~ "Comparator significantly better by bootstrap CI",
        observed_diff < 0 ~ "Comparator numerically better",
        TRUE ~ "No observed difference"
      )
    ) %>%
    arrange(metric, outcome, feature_set, comparator)

  distribution <- bind_rows(lapply(results, `[[`, "distribution"))
  observed <- bind_rows(lapply(results, `[[`, "observed_model_metrics")) %>%
    arrange(metric, outcome, feature_set, model)
  manifest <- bind_rows(lapply(results, `[[`, "manifest")) %>%
    arrange(outcome, feature_set, model)
  warnings <- bind_rows(lapply(results, `[[`, "warnings"))

  metric_summary <- pairwise %>%
    group_by(metric) %>%
    summarise(
      n_comparisons = n(),
      saint_numerically_better = sum(observed_diff > 0, na.rm = TRUE),
      saint_ci_significant = sum(observed_diff > 0 & ci_low > 0, na.rm = TRUE),
      comparator_ci_significant = sum(observed_diff < 0 & ci_high < 0, na.rm = TRUE),
      median_observed_diff = median(observed_diff, na.rm = TRUE),
      .groups = "drop"
    )

  task_summary <- pairwise %>%
    group_by(outcome, feature_set, metric) %>%
    summarise(
      n_comparisons = n(),
      saint_numerically_better = sum(observed_diff > 0, na.rm = TRUE),
      saint_ci_significant = sum(observed_diff > 0 & ci_low > 0, na.rm = TRUE),
      comparator_ci_significant = sum(observed_diff < 0 & ci_high < 0, na.rm = TRUE),
      .groups = "drop"
    )

  csv_path <- file.path(out_dir, "bootstrap_pairwise_differences_long.csv.gz")
  gz <- gzfile(csv_path, open = "wt")
  write.csv(distribution, gz, row.names = FALSE)
  close(gz)

  write.csv(pairwise, file.path(out_dir, "bootstrap_pairwise_summary.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(observed, file.path(out_dir, "observed_model_metrics.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(manifest, file.path(out_dir, "input_manifest.csv"), row.names = FALSE, fileEncoding = "UTF-8")

  wb <- createWorkbook()
  sheets <- list(
    metric_summary = metric_summary,
    pairwise_summary = pairwise,
    task_summary = task_summary,
    observed_model_metrics = observed,
    input_manifest = manifest,
    alignment_checks = warnings
  )
  for (nm in names(sheets)) {
    addWorksheet(wb, nm)
    writeData(wb, nm, sheets[[nm]])
    freezePane(wb, nm, firstRow = TRUE)
    setColWidths(wb, nm, cols = seq_len(ncol(sheets[[nm]])), widths = "auto")
  }
  saveWorkbook(wb, file.path(out_dir, "paired_bootstrap_model_results.xlsx"), overwrite = TRUE)

  plot_df <- pairwise %>%
    mutate(
      metric = factor(metric, levels = metric_levels),
      comparator_label = factor(comparator_label, levels = rev(unname(model_labels[setdiff(models, "SAINT")]))),
      significant = ci_low > 0 | ci_high < 0
    )

  p <- ggplot(plot_df, aes(x = observed_diff, y = comparator_label, color = significant)) +
    geom_vline(xintercept = 0, linetype = 2, color = "grey45") +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.18, linewidth = 0.45) +
    geom_point(size = 1.9) +
    facet_grid(metric ~ outcome, scales = "free_x") +
    scale_color_manual(values = c("FALSE" = "#6B7A90", "TRUE" = "#D55E00"), guide = "none") +
    labs(
      title = "Paired bootstrap differences: SAINT minus comparator",
      subtitle = "Positive values favor SAINT; error bars show percentile 95% bootstrap intervals",
      x = "SAINT - comparator",
      y = "Comparator"
    ) +
    theme_bw(base_size = 9) +
    theme(strip.background = element_rect(fill = "#EAF2F8"), plot.title = element_text(face = "bold"))
  ggsave(file.path(out_dir, "paired_bootstrap_forest_by_metric_outcome.png"), p, width = 13, height = 10, dpi = 300)

  heat_df <- metric_summary %>%
    mutate(metric = factor(metric, levels = metric_levels))
  p2 <- ggplot(heat_df, aes(x = metric, y = "SAINT", fill = saint_ci_significant / n_comparisons)) +
    geom_tile(color = "white") +
    geom_text(aes(label = paste0(saint_ci_significant, "/", n_comparisons)), size = 4) +
    scale_fill_gradient(low = "#F8F1D5", high = "#0B7A53", limits = c(0, 1)) +
    labs(
      title = "Bootstrap-supported SAINT advantages by metric",
      subtitle = "Cell text: number of comparisons where the bootstrap 95% CI for SAINT - comparator excludes 0 in favor of SAINT",
      x = "Metric",
      y = NULL,
      fill = "Proportion"
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.title = element_text(face = "bold"))
  ggsave(file.path(out_dir, "bootstrap_significance_summary_heatmap.png"), p2, width = 8, height = 2.8, dpi = 300)

  notes <- c(
    "Paired individual-level bootstrap model comparison for European holdout only.",
    "",
    paste0("Bootstrap replicates: ", B),
    paste0("Target time: ", target_time, " days"),
    paste0("Net benefit threshold: ", threshold_nb),
    "",
    "For each outcome x feature-set task, the same bootstrap sample indices were used across SAINT and comparator models.",
    "Difference definition: SAINT score - comparator score.",
    "Positive differences favor SAINT.",
    "",
    "Direction-aligned scores:",
    "  C-index: C-index",
    "  OE_low and OE_high: -abs(log(O/E))",
    "  ICI_all: -ICI",
    "  NB_10pct: net benefit at the 10% threshold",
    "",
    "Risk strata for O/E were recalculated within each bootstrap sample using model-specific predicted 10-year risk halves.",
    "Bootstrap p-values are two-sided percentile/sign p-values with a +1 continuity correction.",
    "",
    "Important note: prediction files do not contain participant IDs; paired alignment is based on row order after checking time/event equality against SAINT."
  )
  writeLines(notes, file.path(out_dir, "analysis_notes.txt"), useBytes = TRUE)

  invisible(list(pairwise = pairwise, metric_summary = metric_summary))
}

tasks <- make_task_manifest()
requested_cores <- as.integer(Sys.getenv("BOOTSTRAP_CORES", "5"))
n_cores <- max(1, min(requested_cores, detectCores() - 2, nrow(tasks)))

cat("Paired bootstrap model comparison\n")
cat("European holdout only\n")
cat("Tasks:", nrow(tasks), "\n")
cat("Bootstrap replicates:", B, "\n")
cat("Cores:", n_cores, "\n")
cat("Output directory:", out_dir, "\n")

start_time <- Sys.time()

if (n_cores > 1) {
  cl <- makeCluster(n_cores)
  clusterExport(
    cl,
    varlist = c(
      "base_predictions", "outcomes", "outcome_labels", "models", "model_labels",
      "feature_files", "feature_levels", "metric_levels", "target_time",
      "threshold_nb", "B", "seed0", "clip_prob", "load_prediction",
      "c_index_metric", "observed_risk_km", "oe_metric", "split_low_high",
      "ici_metric", "censoring_weights", "net_benefit_metric",
      "calc_model_metrics", "check_alignment", "run_one_task", "tasks"
    ),
    envir = environment()
  )
  clusterEvalQ(cl, {
    library(survival)
    library(dplyr)
  })
  results <- parLapplyLB(cl, seq_len(nrow(tasks)), run_one_task, tasks = tasks)
  stopCluster(cl)
} else {
  results <- lapply(seq_len(nrow(tasks)), run_one_task, tasks = tasks)
}

outputs <- write_outputs(results)
cat("Elapsed minutes:", round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2), "\n")
cat("Metric summary:\n")
print(outputs$metric_summary)
