rm(list = ls())

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(openxlsx)
  library(parallel)
})

base_path <- Sys.getenv("CVD_PREDICTION_ROOT", unset = "artifacts/predictions")
out_dir <- Sys.getenv("CVD_DCA_OUTPUT", unset = "results/performance/clinical_utility")

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(
  "Total_CVD" = "Total CVD",
  "ASCVD" = "ASCVD",
  "HF" = "HF"
)

models <- c(
  "PREVENT",
  "LinearModel",
  "XGBoost",
  "MLP",
  "TabNet",
  "NODE",
  "FT-Transformer",
  "SAINT",
  "TabPFN"
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
  "Clinical predictors only" = "classic_%s.csv",
  "Protein markers only" = "protein_lassonet_%s.csv",
  "Clinical predictors plus protein markers" = "all_lassonet_%s.csv"
)
feature_levels <- names(feature_files)

validations <- c(
  "European hold-out set" = "val",
  "Asian ancestry" = "asian",
  "Other ancestry" = "other"
)
validation_levels <- names(validations)

target_time <- 3652.5
report_thresholds <- c(0.05, 0.075, 0.10)
curve_thresholds <- seq(0.005, 0.15, by = 0.001)
bootstrap_n <- as.integer(Sys.getenv("BOOTSTRAP_B", unset = "1000"))
set.seed(221)

load_prediction <- function(model, outcome, feature_set, suffix) {
  if (model == "PREVENT" && feature_set != "Clinical predictors only") {
    return(NULL)
  }
  file_name <- sprintf(feature_files[[feature_set]], suffix)
  file_path <- file.path(base_path, model, outcome, file_name)
  if (!file.exists(file_path)) return(NULL)

  data <- read.csv(file_path)
  needed <- c("time", "event", "prob_10yr")
  if (!all(needed %in% names(data))) return(NULL)
  data <- na.omit(data[, needed])
  data$time <- as.numeric(data$time)
  data$event <- as.integer(data$event)
  data$prob_10yr <- pmin(pmax(as.numeric(data$prob_10yr), 0), 1)
  data
}

censoring_weights <- function(data, t0) {
  censor_event <- 1L - data$event
  km <- survfit(Surv(time, censor_event) ~ 1, data = data)
  km_summary <- summary(km)

  get_g <- function(times) {
    if (length(km_summary$time) == 0) {
      return(rep(1, length(times)))
    }
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

tasks <- expand.grid(
  Outcome = outcomes,
  ValidationSet = validation_levels,
  PredictorSet = feature_levels,
  Model = models,
  stringsAsFactors = FALSE
)
tasks <- tasks[!(tasks$Model == "PREVENT" & tasks$PredictorSet != "Clinical predictors only"), ]

run_task <- function(i) {
  set.seed(221 + i)
  task <- tasks[i, ]
  suffix <- unname(validations[[task$ValidationSet]])
  data <- load_prediction(task$Model, task$Outcome, task$PredictorSet, suffix)
  if (is.null(data) || nrow(data) == 0) return(NULL)

  report_nb <- net_benefit_ipcw(data, report_thresholds, target_time)
  report_ci <- bootstrap_ci(data, report_thresholds, target_time, bootstrap_n)
  report <- report_nb %>%
    left_join(report_ci, by = "Threshold") %>%
    mutate(
      Outcome = task$Outcome,
      OutcomeLabel = unname(outcome_labels[[task$Outcome]]),
      ValidationSet = task$ValidationSet,
      PredictorSet = task$PredictorSet,
      Model = task$Model,
      ModelLabel = unname(model_labels[[task$Model]]),
      N = nrow(data),
      Events = sum(data$event == 1 & data$time <= target_time)
    )

  curve <- net_benefit_ipcw(data, curve_thresholds, target_time) %>%
    mutate(
      Outcome = task$Outcome,
      OutcomeLabel = unname(outcome_labels[[task$Outcome]]),
      ValidationSet = task$ValidationSet,
      PredictorSet = task$PredictorSet,
      Model = task$Model,
      ModelLabel = unname(model_labels[[task$Model]])
    )

  list(report = report, curve = curve)
}

cat("Clinical utility Net Benefit calculation\n")
cat("Tasks:", nrow(tasks), "\n")
cat("Bootstrap replicates:", bootstrap_n, "\n")

n_cores <- max(1, min(8, detectCores() - 2, nrow(tasks)))
cat("Cores:", n_cores, "\n")
start_time <- Sys.time()

if (n_cores > 1) {
  cl <- makeCluster(n_cores)
  clusterExport(
    cl,
    varlist = c(
      "base_path", "outcomes", "outcome_labels", "models", "model_labels",
      "feature_files", "feature_levels", "validations", "validation_levels",
      "target_time", "report_thresholds",
      "curve_thresholds", "bootstrap_n", "tasks", "load_prediction",
      "censoring_weights", "net_benefit_ipcw", "bootstrap_ci", "run_task"
    ),
    envir = environment()
  )
  clusterEvalQ(cl, {
    library(survival)
    library(dplyr)
  })
  results <- parLapplyLB(cl, seq_len(nrow(tasks)), run_task)
  stopCluster(cl)
} else {
  results <- lapply(seq_len(nrow(tasks)), run_task)
}

results <- results[!vapply(results, is.null, logical(1))]
report_long <- bind_rows(lapply(results, `[[`, "report")) %>%
  select(Outcome, OutcomeLabel, ValidationSet, PredictorSet, Model, ModelLabel, Threshold, N, Events,
         NetBenefit, NetBenefit_L, NetBenefit_U, TreatAll, TreatNone)
curve_long <- bind_rows(lapply(results, `[[`, "curve")) %>%
  select(Outcome, OutcomeLabel, ValidationSet, PredictorSet, Model, ModelLabel, Threshold,
         NetBenefit, TreatAll, TreatNone)

report_formatted <- report_long %>%
  mutate(
    ThresholdLabel = case_when(
      abs(Threshold - 0.075) < 1e-10 ~ "7.5%",
      TRUE ~ paste0(formatC(Threshold * 100, format = "f", digits = 0), "%")
    ),
    NetBenefit_95CI = sprintf("%.4f (%.4f-%.4f)", NetBenefit, NetBenefit_L, NetBenefit_U)
  )

wide_table <- report_formatted %>%
  mutate(
    Outcome = factor(Outcome, levels = outcomes),
    ValidationSet = factor(ValidationSet, levels = validation_levels),
    PredictorSet = factor(PredictorSet, levels = feature_levels),
    Model = factor(Model, levels = models)
  ) %>%
  arrange(Outcome, ValidationSet, PredictorSet, Model, Threshold) %>%
  select(OutcomeLabel, ValidationSet, PredictorSet, ModelLabel, ThresholdLabel, NetBenefit_95CI) %>%
  pivot_wider(
    id_cols = c(OutcomeLabel, ValidationSet, PredictorSet, ModelLabel),
    names_from = ThresholdLabel,
    values_from = NetBenefit_95CI
  ) %>%
  rename(
    Outcome = OutcomeLabel,
    `Validation set` = ValidationSet,
    `Predictor set` = PredictorSet,
    Model = ModelLabel
  )

wb <- createWorkbook()
addWorksheet(wb, "Net Benefit long")
writeData(wb, "Net Benefit long", report_long)
addWorksheet(wb, "Supplementary table")
writeData(wb, "Supplementary table", wide_table)
addWorksheet(wb, "DCA curve data")
writeData(wb, "DCA curve data", curve_long)

for (sheet in names(wb)) {
  setColWidths(wb, sheet, cols = 1:20, widths = "auto")
}

excel_out <- file.path(out_dir, "Supplementary_Table_net_benefit_10yr.xlsx")
saveWorkbook(wb, excel_out, overwrite = TRUE)

write.csv(report_long, file.path(out_dir, "net_benefit_10yr_thresholds_long.csv"), row.names = FALSE, fileEncoding = "UTF-8")
write.csv(curve_long, file.path(out_dir, "dca_curve_10yr_data.csv"), row.names = FALSE, fileEncoding = "UTF-8")

cat("Done.\n")
cat("Elapsed minutes:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "\n")
cat("Saved:", excel_out, "\n")
