rm(list = ls())

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(tidyr)
  library(openxlsx)
  library(parallel)
})

base_path <- Sys.getenv("CVD_REDUCED_PREDICTION_ROOT", unset = "artifacts/predictions/Kneedle/SAINT")
out_dir <- getwd()

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c("Total_CVD" = "Total CVD", "ASCVD" = "ASCVD", "HF" = "HF")
pred_col_name <- "prob_10yr"
target_time <- 3652.5
B_boot <- as.integer(Sys.getenv("BOOTSTRAP_B", unset = "1000"))
CALCULATE_BOOTSTRAP_CI <- TRUE
set.seed(221)

calc_ici_robust <- function(df, t_time, p_col) {
  probs <- df[[p_col]]
  p_c <- pmin(pmax(probs, 1e-6), 1 - 1e-6)
  df$LP <- log(-log(1 - p_c))

  fit <- tryCatch(
    suppressWarnings(coxph(Surv(time, event) ~ LP, data = df)),
    error = function(e) NULL
  )
  if (is.null(fit)) return(NA_real_)

  bh <- tryCatch(basehaz(fit, centered = FALSE), error = function(e) NULL)
  if (is.null(bh) || nrow(bh) == 0) return(NA_real_)

  bh_t <- bh %>% filter(time <= t_time)
  h0_t <- if (nrow(bh_t) == 0) 0 else tail(bh_t$hazard, 1)
  beta <- unname(coef(fit)[1])
  calib_risk <- 1 - exp(-h0_t * exp(beta * df$LP))
  mean(abs(probs - calib_risk), na.rm = TRUE)
}

calc_metrics_subset <- function(sub_data, t_time, n_boot, group_name, p_col) {
  current_probs <- sub_data[[p_col]]
  n_size <- nrow(sub_data)

  if (n_size == 0 || sum(sub_data$event, na.rm = TRUE) == 0) {
    return(data.frame(
      Group = group_name, N = n_size, Events = sum(sub_data$event, na.rm = TRUE),
      oe = NA_real_, oe_L = NA_real_, oe_U = NA_real_,
      slope = NA_real_, slope_L = NA_real_, slope_U = NA_real_,
      ici = NA_real_, ici_L = NA_real_, ici_U = NA_real_
    ))
  }

  exp_risk <- mean(current_probs, na.rm = TRUE)
  km_sum <- tryCatch(
    summary(survfit(Surv(time, event) ~ 1, data = sub_data), times = t_time),
    error = function(e) NULL
  )
  obs_risk <- if (!is.null(km_sum) && length(km_sum$surv) > 0) 1 - km_sum$surv else NA_real_
  oe_est <- obs_risk / exp_risk

  p_c <- pmin(pmax(current_probs, 1e-6), 1 - 1e-6)
  sub_data$LP_cloglog <- log(-log(1 - p_c))
  fit_slope <- tryCatch(
    suppressWarnings(coxph(Surv(time, event) ~ LP_cloglog, data = sub_data)),
    error = function(e) NULL
  )
  slope_est <- if (!is.null(fit_slope)) unname(coef(fit_slope)[1]) else NA_real_
  ici_est <- calc_ici_robust(sub_data, t_time, p_col)

  if (!is.na(obs_risk) && obs_risk > 0 && !is.null(km_sum$std.err)) {
    se_log_oe <- km_sum$std.err / obs_risk
    oe_L <- exp(log(oe_est) - 1.96 * se_log_oe)
    oe_U <- exp(log(oe_est) + 1.96 * se_log_oe)
  } else {
    oe_L <- NA_real_
    oe_U <- NA_real_
  }

  if (!is.null(fit_slope)) {
    se_slope <- sqrt(diag(vcov(fit_slope))[1])
    slope_L <- slope_est - 1.96 * se_slope
    slope_U <- slope_est + 1.96 * se_slope
  } else {
    slope_L <- NA_real_
    slope_U <- NA_real_
  }

  if (!CALCULATE_BOOTSTRAP_CI || is.na(oe_est) || is.na(slope_est) || is.na(ici_est) || n_boot <= 0) {
    return(data.frame(
      Group = group_name, N = n_size, Events = sum(sub_data$event),
      oe = oe_est, oe_L = oe_L, oe_U = oe_U,
      slope = slope_est, slope_L = slope_L, slope_U = slope_U,
      ici = ici_est, ici_L = NA_real_, ici_U = NA_real_
    ))
  }

  boot_metrics <- replicate(n_boot, {
    idx <- sample.int(n_size, size = n_size, replace = TRUE)
    boot_df <- sub_data[idx, , drop = FALSE]

    e_b <- mean(boot_df[[p_col]], na.rm = TRUE)
    km_b <- tryCatch(
      summary(survfit(Surv(time, event) ~ 1, data = boot_df), times = t_time),
      error = function(e) NULL
    )
    o_b <- if (!is.null(km_b) && length(km_b$surv) > 0) 1 - km_b$surv else NA_real_
    oe_b <- o_b / e_b

    p_c_b <- pmin(pmax(boot_df[[p_col]], 1e-6), 1 - 1e-6)
    boot_df$LP_cloglog <- log(-log(1 - p_c_b))
    fit_s_b <- tryCatch(
      suppressWarnings(coxph(Surv(time, event) ~ LP_cloglog, data = boot_df)),
      error = function(e) NULL
    )
    slope_b <- if (!is.null(fit_s_b)) unname(coef(fit_s_b)[1]) else NA_real_
    ici_b <- tryCatch(calc_ici_robust(boot_df, t_time, p_col), error = function(e) NA_real_)

    c(oe = oe_b, slope = slope_b, ici = ici_b)
  })

  if (is.null(dim(boot_metrics))) boot_metrics <- matrix(boot_metrics, nrow = 3)

  oe_ci <- quantile(boot_metrics[1, ], probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  slope_ci <- quantile(boot_metrics[2, ], probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  ici_ci <- quantile(boot_metrics[3, ], probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE)

  data.frame(
    Group = group_name, N = n_size, Events = sum(sub_data$event),
    oe = oe_est, oe_L = oe_ci[1], oe_U = oe_ci[2],
    slope = slope_est, slope_L = slope_ci[1], slope_U = slope_ci[2],
    ici = ici_est, ici_L = ici_ci[1], ici_U = ici_ci[2]
  )
}

make_curve_data <- function(data, t_time, p_col) {
  data <- data %>% mutate(decile = ntile(.data[[p_col]], 10))
  bind_rows(lapply(1:10, function(d) {
    sub_d <- data %>% filter(decile == d)
    km <- tryCatch(
      summary(survfit(Surv(time, event) ~ 1, data = sub_d), times = t_time),
      error = function(e) NULL
    )
    if (!is.null(km) && length(km$surv) > 0) {
      data.frame(
        decile = d, N = nrow(sub_d), Events = sum(sub_d$event),
        pred = mean(sub_d[[p_col]], na.rm = TRUE), obs = 1 - km$surv,
        obs_lower = if (!is.null(km$upper)) 1 - km$upper else NA_real_,
        obs_upper = if (!is.null(km$lower)) 1 - km$lower else NA_real_
      )
    } else {
      data.frame(
        decile = d, N = nrow(sub_d), Events = sum(sub_d$event),
        pred = mean(sub_d[[p_col]], na.rm = TRUE),
        obs = NA_real_, obs_lower = NA_real_, obs_upper = NA_real_
      )
    }
  }))
}

tasks <- data.frame(Outcome = outcomes, stringsAsFactors = FALSE)

run_one_task <- function(task_index) {
  set.seed(221 + task_index)
  curr <- tasks[task_index, , drop = FALSE]
  file_path <- file.path(base_path, curr$Outcome, "all_lassonet_val.csv")
  if (!file.exists(file_path)) return(NULL)

  data_raw <- read.csv(file_path)
  if (!all(c("time", "event", pred_col_name) %in% names(data_raw))) return(NULL)
  data <- data_raw %>%
    select(time, event, all_of(pred_col_name)) %>%
    mutate(
      time = as.numeric(time),
      event = as.integer(event),
      !!pred_col_name := pmin(pmax(as.numeric(.data[[pred_col_name]]), 1e-6), 1 - 1e-6)
    ) %>%
    na.omit()

  data <- data %>%
    arrange(.data[[pred_col_name]]) %>%
    mutate(risk_group = ntile(.data[[pred_col_name]], 2))

  metrics <- bind_rows(
    calc_metrics_subset(data, target_time, B_boot, "Overall", pred_col_name),
    calc_metrics_subset(data %>% filter(risk_group == 1), target_time, B_boot, "Low", pred_col_name),
    calc_metrics_subset(data %>% filter(risk_group == 2), target_time, B_boot, "High", pred_col_name)
  )
  metrics$Outcome <- curr$Outcome
  metrics$OutcomeLabel <- unname(outcome_labels[curr$Outcome])
  metrics$ValidationSet <- "European hold-out set"
  metrics$Model <- "SAINT_reduced"
  metrics$ModelLabel <- "SAINT reduced predictor panel"
  metrics$FeatureSet <- "Kneedle-selected clinical plus protein markers"
  metrics$File <- file_path

  curve <- make_curve_data(data, target_time, pred_col_name)
  curve$Outcome <- curr$Outcome
  curve$OutcomeLabel <- unname(outcome_labels[curr$Outcome])
  curve$ValidationSet <- "European hold-out set"
  curve$Model <- "SAINT_reduced"
  curve$ModelLabel <- "SAINT reduced predictor panel"
  curve$FeatureSet <- "Kneedle-selected clinical plus protein markers"
  curve$File <- file_path

  list(metrics = metrics, curve = curve)
}

cat("Reduced SAINT calibration metric calculation in R\n")
cat("Bootstrap replicates:", B_boot, "\n")
cat("Tasks:", nrow(tasks), "\n")

n_cores <- max(1, min(3, nrow(tasks)))
start_time <- Sys.time()

if (n_cores > 1) {
  cl <- makeCluster(n_cores)
  clusterExport(
    cl,
    varlist = c(
      "tasks", "base_path", "pred_col_name", "target_time", "B_boot",
      "CALCULATE_BOOTSTRAP_CI", "outcome_labels", "calc_ici_robust",
      "calc_metrics_subset", "make_curve_data", "run_one_task"
    ),
    envir = environment()
  )
  clusterEvalQ(cl, {
    library(survival)
    library(dplyr)
  })
  results <- parLapplyLB(cl, seq_len(nrow(tasks)), run_one_task)
  stopCluster(cl)
} else {
  results <- lapply(seq_len(nrow(tasks)), run_one_task)
}

results <- results[!vapply(results, is.null, logical(1))]
metrics_all <- bind_rows(lapply(results, `[[`, "metrics")) %>%
  select(Outcome, OutcomeLabel, ValidationSet, Model, ModelLabel, FeatureSet, Group, N, Events,
         oe, oe_L, oe_U, slope, slope_L, slope_U, ici, ici_L, ici_U, File)
curve_all <- bind_rows(lapply(results, `[[`, "curve")) %>%
  select(Outcome, OutcomeLabel, ValidationSet, Model, ModelLabel, FeatureSet, decile, N, Events,
         pred, obs, obs_lower, obs_upper, File)

metrics_formatted <- metrics_all %>%
  mutate(
    OE_Ratio_95CI = ifelse(is.na(oe), "/", sprintf("%.4f (%.4f-%.4f)", oe, oe_L, oe_U)),
    Calibration_Slope_95CI = ifelse(is.na(slope), "/", sprintf("%.4f (%.4f-%.4f)", slope, slope_L, slope_U)),
    ICI_95CI = ifelse(is.na(ici), "/", sprintf("%.4f (%.4f-%.4f)", ici, ici_L, ici_U))
  )

supp_oe <- metrics_formatted %>%
  select(Outcome = OutcomeLabel, `Validation set` = ValidationSet, `Predictor set` = FeatureSet,
         Model = ModelLabel, Group, OE_Ratio_95CI) %>%
  pivot_wider(names_from = Group, values_from = OE_Ratio_95CI)

supp_slope <- metrics_formatted %>%
  select(Outcome = OutcomeLabel, `Validation set` = ValidationSet, `Predictor set` = FeatureSet,
         Model = ModelLabel, Group, Calibration_Slope_95CI) %>%
  pivot_wider(names_from = Group, values_from = Calibration_Slope_95CI)

supp_ici <- metrics_formatted %>%
  select(Outcome = OutcomeLabel, `Validation set` = ValidationSet, `Predictor set` = FeatureSet,
         Model = ModelLabel, Group, ICI_95CI) %>%
  pivot_wider(names_from = Group, values_from = ICI_95CI)

wb <- createWorkbook()
addWorksheet(wb, "Metrics_long")
writeData(wb, "Metrics_long", metrics_all)
addWorksheet(wb, "Metrics_formatted")
writeData(wb, "Metrics_formatted", metrics_formatted)
addWorksheet(wb, "Supplementary Table 5")
writeData(wb, "Supplementary Table 5", supp_oe)
addWorksheet(wb, "Supplementary Table 6")
writeData(wb, "Supplementary Table 6", supp_slope)
addWorksheet(wb, "Supplementary Table 7")
writeData(wb, "Supplementary Table 7", supp_ici)
addWorksheet(wb, "Calibration_curve_deciles")
writeData(wb, "Calibration_curve_deciles", curve_all)

excel_file <- file.path(out_dir, "reduced_saint_calibration_R.xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)
write.csv(metrics_all, file.path(out_dir, "reduced_saint_calibration_R_metrics_long.csv"), row.names = FALSE)

cat("Done.\n")
cat("Elapsed minutes:", round(difftime(Sys.time(), start_time, units = "mins"), 2), "\n")
cat("Saved:", excel_file, "\n")
