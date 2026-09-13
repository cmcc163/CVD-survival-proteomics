rm(list = ls())

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(readr)
  library(openxlsx)
})

base_path <- Sys.getenv("CVD_REDUCED_PREDICTION_ROOT", unset = "artifacts/predictions/Kneedle/SAINT")
out_dir <- getwd()

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c("Total_CVD" = "Total CVD", "ASCVD" = "ASCVD", "HF" = "HF")

target_time <- 3652.5
report_thresholds <- c(0.05, 0.075, 0.10)
curve_thresholds <- seq(0.005, 0.15, by = 0.001)
bootstrap_n <- as.integer(Sys.getenv("BOOTSTRAP_B", unset = "1000"))
set.seed(221)

load_prediction <- function(outcome) {
  file_path <- file.path(base_path, outcome, "all_lassonet_val.csv")
  if (!file.exists(file_path)) return(NULL)
  data <- read.csv(file_path)
  needed <- c("time", "event", "prob_10yr")
  if (!all(needed %in% names(data))) return(NULL)
  data <- na.omit(data[, needed])
  data$time <- as.numeric(data$time)
  data$event <- as.integer(data$event)
  data$prob_10yr <- pmin(pmax(as.numeric(data$prob_10yr), 0), 1)
  attr(data, "file_path") <- file_path
  data
}

censoring_weights <- function(data, t0) {
  censor_event <- 1L - data$event
  km <- survfit(Surv(time, censor_event) ~ 1, data = data)
  km_summary <- summary(km)

  get_g <- function(times) {
    if (length(km_summary$time) == 0) return(rep(1, length(times)))
    idx <- findInterval(times, km_summary$time)
    surv <- ifelse(idx == 0, 1, km_summary$surv[idx])
    pmax(surv, 1e-6)
  }

  case <- data$event == 1 & data$time <= t0
  control <- data$time > t0
  weights <- rep(0, nrow(data))
  weights[case] <- 1 / get_g(pmax(data$time[case] - 1e-8, 0))
  weights[control] <- 1 / get_g(t0)
  list(weights = weights, case = case, control = control)
}

net_benefit_ipcw <- function(data, thresholds, t0) {
  cw <- censoring_weights(data, t0)
  weights <- cw$weights
  case <- cw$case
  control <- cw$control
  n <- nrow(data)

  event_risk <- sum(weights[case]) / n
  non_event_risk <- sum(weights[control]) / n

  rows <- lapply(thresholds, function(pt) {
    treat <- data$prob_10yr >= pt
    tp_rate <- sum(weights[case & treat]) / n
    fp_rate <- sum(weights[control & treat]) / n
    nb <- tp_rate - fp_rate * pt / (1 - pt)
    data.frame(
      Threshold = pt,
      NetBenefit = nb,
      TreatAll = event_risk - non_event_risk * pt / (1 - pt),
      TreatNone = 0
    )
  })
  bind_rows(rows)
}

bootstrap_ci <- function(data, thresholds, t0, b = bootstrap_n) {
  n <- nrow(data)
  boot_mat <- matrix(NA_real_, nrow = b, ncol = length(thresholds))
  for (i in seq_len(b)) {
    idx <- sample.int(n, n, replace = TRUE)
    boot_data <- data[idx, , drop = FALSE]
    boot_mat[i, ] <- net_benefit_ipcw(boot_data, thresholds, t0)$NetBenefit
  }
  data.frame(
    Threshold = thresholds,
    NetBenefit_L = apply(boot_mat, 2, quantile, probs = 0.025, na.rm = TRUE),
    NetBenefit_U = apply(boot_mat, 2, quantile, probs = 0.975, na.rm = TRUE)
  )
}

run_task <- function(i) {
  set.seed(221 + i)
  outcome <- outcomes[i]
  data <- load_prediction(outcome)
  if (is.null(data) || nrow(data) == 0) return(NULL)
  file_path <- attr(data, "file_path")

  report_nb <- net_benefit_ipcw(data, report_thresholds, target_time)
  report_ci <- bootstrap_ci(data, report_thresholds, target_time, bootstrap_n)
  report <- report_nb %>%
    left_join(report_ci, by = "Threshold") %>%
    mutate(
      Outcome = outcome,
      OutcomeLabel = unname(outcome_labels[[outcome]]),
      ValidationSet = "European hold-out set",
      PredictorSet = "Kneedle-selected clinical plus protein markers",
      Model = "SAINT_reduced",
      ModelLabel = "SAINT reduced predictor panel",
      N = nrow(data),
      Events = sum(data$event == 1 & data$time <= target_time),
      File = file_path
    )

  curve <- net_benefit_ipcw(data, curve_thresholds, target_time) %>%
    mutate(
      Outcome = outcome,
      OutcomeLabel = unname(outcome_labels[[outcome]]),
      ValidationSet = "European hold-out set",
      PredictorSet = "Kneedle-selected clinical plus protein markers",
      Model = "SAINT_reduced",
      ModelLabel = "SAINT reduced predictor panel",
      File = file_path
    )

  list(report = report, curve = curve)
}

cat("Reduced SAINT clinical utility Net Benefit calculation in R\n")
cat("Bootstrap replicates:", bootstrap_n, "\n")
cat("Tasks:", length(outcomes), "\n")
start_time <- Sys.time()

results <- lapply(seq_along(outcomes), run_task)
results <- results[!vapply(results, is.null, logical(1))]
report_all <- bind_rows(lapply(results, `[[`, "report"))
curve_all <- bind_rows(lapply(results, `[[`, "curve"))

supp_table <- report_all %>%
  mutate(
    ThresholdLabel = ifelse(Threshold == 0.05, "5%", ifelse(Threshold == 0.075, "7.5%", "10%")),
    Value = sprintf("%.4f (%.4f-%.4f)", NetBenefit, NetBenefit_L, NetBenefit_U)
  ) %>%
  select(Outcome = OutcomeLabel, `Validation set` = ValidationSet, `Predictor set` = PredictorSet,
         Model = ModelLabel, ThresholdLabel, Value) %>%
  tidyr::pivot_wider(names_from = ThresholdLabel, values_from = Value) %>%
  select(Outcome, `Validation set`, `Predictor set`, Model, `5%`, `7.5%`, `10%`)

wb <- createWorkbook()
addWorksheet(wb, "Net Benefit long")
writeData(wb, "Net Benefit long", report_all)
addWorksheet(wb, "Supplementary table")
writeData(wb, "Supplementary table", supp_table)
addWorksheet(wb, "DCA curve data")
writeData(wb, "DCA curve data", curve_all)

excel_file <- file.path(out_dir, "reduced_saint_net_benefit_R.xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)
write.csv(report_all, file.path(out_dir, "reduced_saint_net_benefit_R_long.csv"), row.names = FALSE)

cat("Done.\n")
cat("Elapsed minutes:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "\n")
cat("Saved:", excel_file, "\n")
