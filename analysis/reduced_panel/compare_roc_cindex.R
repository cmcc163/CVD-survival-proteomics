options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(survival)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(openxlsx)
  library(parallel)
})

project_dir <- Sys.getenv("CVD_REDUCED_RESULTS", unset = "results/reduced_panel")
prediction_root <- Sys.getenv("CVD_PREDICTION_ROOT", unset = "artifacts/predictions")
output_dir <- file.path(project_dir, "output")
figure_dir <- file.path(project_dir, "figures")
auc_ci_source_csv <- file.path(project_dir, "data", "auc_10yr_ci_source.csv")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(Total_CVD = "Total CVD", ASCVD = "ASCVD", HF = "HF")
horizon_days <- 365.25 * 10
B <- as.integer(Sys.getenv("BOOTSTRAP_B", "1000"))
seed0 <- 20260904L

roc_specs <- data.frame(
  model_key = c("PREVENT", "LinearModel", "SAINT", "RP_SAINT"),
  model_label = c("PREVENT equation", "Refitted Cox model", "SAINT", "RP-SAINT"),
  predictor_set = c("Clinical predictors", "Clinical predictors", "Clinical predictors", "Reduced predictor panel"),
  file_name = c("classic_val.csv", "classic_val.csv", "classic_val.csv", "all_lassonet_val.csv"),
  stringsAsFactors = FALSE
)

cindex_specs <- data.frame(
  comparator_key = c(
    "PREVENT_clinical", "Cox_clinical", "Cox_protein", "Cox_clinical_protein",
    "SAINT_clinical", "SAINT_protein", "SAINT_clinical_protein"
  ),
  model_key = c("PREVENT", "LinearModel", "LinearModel", "LinearModel", "SAINT", "SAINT", "SAINT"),
  comparator_model = c(
    "PREVENT equation", "Refitted Cox model", "Refitted Cox model", "Refitted Cox model",
    "SAINT", "SAINT", "SAINT"
  ),
  predictor_set = c(
    "Clinical predictors", "Clinical predictors", "Protein markers", "Clinical + protein markers",
    "Clinical predictors", "Protein markers", "Clinical + protein markers"
  ),
  file_name = c(
    "classic_val.csv", "classic_val.csv", "protein_lassonet_val.csv", "all_lassonet_val.csv",
    "classic_val.csv", "protein_lassonet_val.csv", "all_lassonet_val.csv"
  ),
  stringsAsFactors = FALSE
)

prediction_path <- function(model_key, outcome, file_name) {
  if (model_key == "RP_SAINT") {
    file.path(prediction_root, "Kneedle", "SAINT", outcome, file_name)
  } else {
    file.path(prediction_root, model_key, outcome, file_name)
  }
}

read_prediction <- function(model_key, outcome, file_name) {
  path <- prediction_path(model_key, outcome, file_name)
  if (!file.exists(path)) stop("Prediction file not found: ", path)
  dat <- read.csv(path)
  required <- c("time", "event", "risk_score", "prob_10yr")
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0) stop("Missing columns in ", path, ": ", paste(missing, collapse = ", "))
  dat <- dat[, required]
  dat$time <- as.numeric(dat$time)
  dat$event <- as.integer(dat$event)
  dat$risk_score <- as.numeric(dat$risk_score)
  dat$prob_10yr <- as.numeric(dat$prob_10yr)
  if (any(!complete.cases(dat))) stop("Missing values found in required columns: ", path)
  attr(dat, "source_file") <- path
  dat
}

check_alignment <- function(reference, candidate, label) {
  ok <- nrow(reference) == nrow(candidate) &&
    identical(reference$time, candidate$time) &&
    identical(reference$event, candidate$event)
  if (!ok) stop("Participant rows are not aligned for ", label)
  TRUE
}

censoring_survival <- function(time, event, query_times) {
  fit <- survfit(Surv(time, 1L - event) ~ 1)
  pmax(summary(fit, times = query_times, extend = TRUE)$surv, 1e-8)
}

time_dependent_roc <- function(dat, horizon = horizon_days) {
  cases <- dat[dat$event == 1L & dat$time <= horizon, , drop = FALSE]
  controls <- dat[dat$time > horizon, , drop = FALSE]
  if (nrow(cases) == 0 || nrow(controls) == 0) stop("No cases or controls at 10 years")
  case_weights <- 1 / censoring_survival(dat$time, dat$event, pmax(cases$time - 1e-8, 0))
  case_scores <- cases$prob_10yr
  control_scores <- controls$prob_10yr
  thresholds <- c(Inf, sort(unique(c(case_scores, control_scores)), decreasing = TRUE), -Inf)
  tpr <- vapply(thresholds, function(z) sum(case_weights[case_scores >= z]) / sum(case_weights), numeric(1))
  fpr <- vapply(thresholds, function(z) mean(control_scores >= z), numeric(1))
  roc <- data.frame(fpr = fpr, tpr = tpr)
  roc <- roc[order(roc$fpr, roc$tpr), , drop = FALSE]
  roc[!duplicated(paste(roc$fpr, roc$tpr)), , drop = FALSE]
}

auc_from_roc <- function(roc) {
  roc <- roc[order(roc$fpr, roc$tpr), , drop = FALSE]
  sum(diff(roc$fpr) * (head(roc$tpr, -1) + tail(roc$tpr, -1)) / 2)
}

smooth_roc_curve <- function(roc, n = 501) {
  roc <- aggregate(tpr ~ fpr, data = roc, FUN = max)
  if (min(roc$fpr) > 0) roc <- rbind(data.frame(fpr = 0, tpr = 0), roc)
  if (max(roc$fpr) < 1) roc <- rbind(roc, data.frame(fpr = 1, tpr = 1))
  grid_x <- seq(0, 1, length.out = n)
  eps <- 1e-4
  fit_data <- roc[roc$fpr > eps & roc$fpr < 1 - eps & roc$tpr > eps & roc$tpr < 1 - eps, ]
  if (nrow(fit_data) >= 8) {
    fit_data$xf <- qnorm(pmin(pmax(fit_data$fpr, eps), 1 - eps))
    fit_data$yt <- qnorm(pmin(pmax(fit_data$tpr, eps), 1 - eps))
    fit <- lm(yt ~ xf, data = fit_data)
    smooth_y <- pnorm(coef(fit)[1] + coef(fit)[2] * qnorm(pmin(pmax(grid_x, eps), 1 - eps)))
  } else {
    smooth_y <- approx(roc$fpr, roc$tpr, xout = grid_x, ties = "ordered", rule = 2)$y
  }
  smooth_y <- cummax(pmin(pmax(smooth_y, 0), 1))
  smooth_y[1] <- 0
  smooth_y[length(smooth_y)] <- 1
  data.frame(fpr = grid_x, tpr = cummax(smooth_y))
}

c_index <- function(dat) {
  if (nrow(dat) < 2 || sum(dat$event == 1L) == 0) return(NA_real_)
  tryCatch(
    as.numeric(concordance(Surv(time, event) ~ risk_score, data = dat, reverse = TRUE)$concordance),
    error = function(e) NA_real_
  )
}

attach_auc_confidence_intervals <- function(auc_data) {
  source <- read.csv(auc_ci_source_csv, check.names = FALSE)
  source_key <- paste(source$outcome_label, source$model, sep = "||")
  target_key <- paste(auc_data$outcome_label, auc_data$model, sep = "||")
  idx <- match(target_key, source_key)
  if (anyNA(idx)) stop("AUC confidence interval lookup failed for one or more plotted models")
  auc_data$auc_10yr_ci_low <- source$auc_10yr_ci_low[idx]
  auc_data$auc_10yr_ci_high <- source$auc_10yr_ci_high[idx]
  auc_data$auc_10yr_formatted <- sprintf(
    "%.4f (%.4f-%.4f)", auc_data$auc_10yr,
    auc_data$auc_10yr_ci_low, auc_data$auc_10yr_ci_high
  )
  auc_data
}

run_roc <- function() {
  auc_rows <- list()
  curve_rows <- list()
  k <- 1L
  for (outcome in outcomes) {
    reference <- read_prediction("RP_SAINT", outcome, "all_lassonet_val.csv")
    for (i in seq_len(nrow(roc_specs))) {
      sp <- roc_specs[i, ]
      dat <- read_prediction(sp$model_key, outcome, sp$file_name)
      check_alignment(reference, dat, paste(outcome, sp$model_label))
      raw_roc <- time_dependent_roc(dat)
      auc <- auc_from_roc(raw_roc)
      curve <- smooth_roc_curve(raw_roc)
      auc_rows[[k]] <- data.frame(
        outcome = outcome, outcome_label = unname(outcome_labels[outcome]),
        model = sp$model_label, predictor_set = sp$predictor_set,
        auc_10yr = auc, n = nrow(dat), events_total = sum(dat$event),
        events_10yr = sum(dat$event == 1L & dat$time <= horizon_days),
        source_file = attr(dat, "source_file"), stringsAsFactors = FALSE
      )
      curve_rows[[k]] <- data.frame(
        outcome = outcome, outcome_label = unname(outcome_labels[outcome]),
        model = sp$model_label, predictor_set = sp$predictor_set,
        auc_10yr = auc, fpr = curve$fpr, tpr = curve$tpr,
        stringsAsFactors = FALSE
      )
      k <- k + 1L
    }
  }
  auc_data <- attach_auc_confidence_intervals(do.call(rbind, auc_rows))
  curve_data <- do.call(rbind, curve_rows)
  write.csv(auc_data, file.path(output_dir, "RP_SAINT_10yr_AUC_summary.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(curve_data, file.path(output_dir, "RP_SAINT_10yr_ROC_curve_data.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.xlsx(list(`10-year AUC` = auc_data, `ROC curve data` = curve_data),
             file.path(output_dir, "RP_SAINT_10yr_AUC_and_ROC.xlsx"), overwrite = TRUE)

  model_levels <- roc_specs$model_label
  curve_data$model <- factor(curve_data$model, levels = model_levels)
  curve_data$outcome_label <- factor(curve_data$outcome_label, levels = unname(outcome_labels))
  colors <- c("PREVENT equation" = "#111111", "Refitted Cox model" = "#6B6B6B",
              "SAINT" = "#3B5BA9", "RP-SAINT" = "#D7191C")
  line_widths <- setNames(rep(0.90, length(model_levels)), model_levels)

  panels <- lapply(seq_along(outcomes), function(j) {
    outcome <- outcomes[j]
    d <- curve_data[curve_data$outcome == outcome, , drop = FALSE]
    a <- auc_data[auc_data$outcome == outcome, , drop = FALSE]
    a <- a[match(model_levels, a$model), , drop = FALSE]
    legend_labels <- sprintf(
      "%s: AUC = %.4f (%.4f-%.4f)",
      model_levels, a$auc_10yr, a$auc_10yr_ci_low, a$auc_10yr_ci_high
    )
    ggplot(d, aes(fpr, tpr, color = model, group = model)) +
      geom_abline(intercept = 0, slope = 1, color = "grey75", linewidth = 0.38, linetype = "dashed") +
      geom_line(aes(linewidth = model), lineend = "round") +
      scale_color_manual(values = colors, breaks = model_levels, labels = legend_labels) +
      scale_linewidth_manual(values = line_widths, breaks = model_levels, guide = "none") +
      coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      scale_x_continuous(breaks = seq(0, 1, 0.2), labels = sprintf("%.1f", seq(0, 1, 0.2))) +
      scale_y_continuous(breaks = seq(0, 1, 0.2), labels = sprintf("%.1f", seq(0, 1, 0.2))) +
      labs(title = NULL,
           x = "1 - Specificity", y = "Sensitivity", color = NULL) +
      guides(color = guide_legend(ncol = 1, override.aes = list(linewidth = rep(1.05, 4)))) +
      theme_classic(base_family = "Arial", base_size = 10.5) +
      theme(
        plot.title = element_text(face = "bold", size = 11.5, hjust = 0, margin = margin(b = 5)),
        axis.title = element_text(face = "bold", size = 10.5),
        axis.text = element_text(color = "black", size = 9.5),
        axis.line = element_line(color = "black", linewidth = 0.55),
        axis.ticks = element_line(color = "black", linewidth = 0.45),
        legend.position = c(0.97, 0.03), legend.justification = c(1, 0),
        legend.background = element_rect(fill = scales::alpha("white", 0.88), color = NA),
        legend.text = element_text(size = 7.0), legend.key.width = unit(0.48, "cm"),
        legend.key.height = unit(0.28, "cm"), plot.margin = margin(7, 7, 7, 7)
      )
  })
  combined <- arrangeGrob(grobs = panels, nrow = 1)
  png_path <- file.path(figure_dir, "Figure_RP_SAINT_10yr_ROC_combined.png")
  ggsave(png_path, combined, width = 12.6, height = 4.15, units = "in", dpi = 600, bg = "white")
  # Optional PDF export (intentionally disabled; uncomment if needed later):
  # ggsave(file.path(figure_dir, "Figure_RP_SAINT_10yr_ROC_combined.pdf"),
  #        combined, width = 12.6, height = 4.15, units = "in", device = cairo_pdf, bg = "white")
  list(auc = auc_data, curves = curve_data, png = png_path)
}

run_cindex_outcome <- function(outcome_index) {
  outcome <- outcomes[outcome_index]
  set.seed(seed0 + outcome_index)
  rp <- read_prediction("RP_SAINT", outcome, "all_lassonet_val.csv")
  comparators <- lapply(seq_len(nrow(cindex_specs)), function(i) {
    sp <- cindex_specs[i, ]
    dat <- read_prediction(sp$model_key, outcome, sp$file_name)
    check_alignment(rp, dat, paste(outcome, sp$comparator_key))
    dat
  })
  names(comparators) <- cindex_specs$comparator_key
  n <- nrow(rp)
  observed_rp <- c_index(rp)
  observed_comp <- vapply(comparators, c_index, numeric(1))
  boot_rp <- rep(NA_real_, B)
  boot_comp <- matrix(NA_real_, nrow = B, ncol = nrow(cindex_specs),
                      dimnames = list(NULL, cindex_specs$comparator_key))
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    boot_rp[b] <- c_index(rp[idx, , drop = FALSE])
    for (i in seq_along(comparators)) {
      boot_comp[b, i] <- c_index(comparators[[i]][idx, , drop = FALSE])
    }
  }
  rows <- list()
  dist <- list()
  rp_ci <- quantile(boot_rp, c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  for (i in seq_len(nrow(cindex_specs))) {
    diffs <- boot_rp - boot_comp[, i]
    valid <- is.finite(diffs)
    d <- diffs[valid]
    comp_b <- boot_comp[valid, i]
    delta_ci <- quantile(d, c(0.025, 0.975), na.rm = TRUE, names = FALSE)
    comp_ci <- quantile(comp_b, c(0.025, 0.975), na.rm = TRUE, names = FALSE)
    p_two <- 2 * min((sum(d <= 0) + 1) / (length(d) + 1),
                     (sum(d >= 0) + 1) / (length(d) + 1))
    p_two <- min(p_two, 1)
    rows[[i]] <- data.frame(
      outcome = outcome, outcome_label = unname(outcome_labels[outcome]),
      comparator_key = cindex_specs$comparator_key[i],
      comparator_model = cindex_specs$comparator_model[i],
      predictor_set = cindex_specs$predictor_set[i],
      n = n, events = sum(rp$event), bootstrap_replicates = length(d),
      rp_saint_cindex = observed_rp, rp_saint_ci_low = rp_ci[1], rp_saint_ci_high = rp_ci[2],
      comparator_cindex = observed_comp[i], comparator_ci_low = comp_ci[1], comparator_ci_high = comp_ci[2],
      delta_cindex = observed_rp - observed_comp[i], delta_ci_low = delta_ci[1], delta_ci_high = delta_ci[2],
      p_boot_two_sided = p_two,
      interpretation = if (delta_ci[1] > 0) "RP-SAINT significantly higher" else if (delta_ci[2] < 0) "Comparator significantly higher" else "No statistically significant difference",
      stringsAsFactors = FALSE
    )
    dist[[i]] <- data.frame(
      outcome = outcome, comparator_key = cindex_specs$comparator_key[i],
      bootstrap = seq_along(d), rp_saint_cindex = boot_rp[valid],
      comparator_cindex = comp_b, delta_cindex = d, stringsAsFactors = FALSE
    )
  }
  list(summary = do.call(rbind, rows), distribution = do.call(rbind, dist))
}

run_cindex <- function() {
  cores <- max(1L, min(3L, detectCores() - 1L))
  if (.Platform$OS.type == "windows" && cores > 1L) {
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    clusterEvalQ(cl, library(survival))
    clusterExport(cl, varlist = c(
      "outcomes", "outcome_labels", "horizon_days", "B", "seed0", "prediction_root",
      "cindex_specs", "prediction_path", "read_prediction", "check_alignment", "c_index"
    ), envir = environment())
    res <- parLapply(cl, seq_along(outcomes), run_cindex_outcome)
  } else {
    res <- lapply(seq_along(outcomes), run_cindex_outcome)
  }
  summary <- do.call(rbind, lapply(res, `[[`, "summary"))
  distribution <- do.call(rbind, lapply(res, `[[`, "distribution"))
  write.csv(summary, file.path(output_dir, "RP_SAINT_Cindex_paired_bootstrap_summary.csv"), row.names = FALSE, fileEncoding = "UTF-8")
  write.csv(
    distribution,
    file.path(output_dir, "RP_SAINT_Cindex_bootstrap_distributions.csv"),
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  write.xlsx(
    list(`Paired comparisons` = summary),
    file.path(output_dir, "RP_SAINT_Cindex_paired_bootstrap_results.xlsx"), overwrite = TRUE
  )
  summary
}

start_time <- Sys.time()
roc_results <- run_roc()
if (identical(Sys.getenv("ROC_ONLY"), "1")) {
  capture.output(sessionInfo(), file = file.path(output_dir, "session_info.txt"))
  cat("Completed ROC-only update in", project_dir, "\n")
  quit(save = "no", status = 0)
}
cindex_results <- run_cindex()
cat("Completed ROC and paired C-index bootstrap analysis in", project_dir, "\n")
