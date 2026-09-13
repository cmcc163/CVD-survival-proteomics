options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
  library(survival)
  library(ggplot2)
  library(grid)
})

suppressPackageStartupMessages(library(gridExtra))

prediction_root <- Sys.getenv("CVD_PREDICTION_ROOT", unset = "artifacts/predictions")

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  output_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else {
  output_dir <- getwd()
}

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(Total_CVD = "Total CVD", ASCVD = "ASCVD", HF = "HF")

model_order <- c(
  "PREVENT", "LinearModel", "XGBoost", "MLP", "TabNet",
  "NODE", "FT-Transformer", "SAINT", "TabPFN"
)
ml_model_order <- setdiff(model_order, "PREVENT")

model_labels <- c(
  PREVENT = "PREVENT equation",
  LinearModel = "Refitted Cox model",
  XGBoost = "XGBoost",
  MLP = "MLP",
  TabNet = "TabNet",
  NODE = "NODE",
  `FT-Transformer` = "FT-Transformer",
  SAINT = "SAINT",
  TabPFN = "TabPFN"
)

predictor_sets <- list(
  list(
    id = "classic",
    label = "Clinical predictors",
    file = "classic_val.csv",
    models = model_order
  ),
  list(
    id = "protein_lassonet",
    label = "Protein markers",
    file = "protein_lassonet_val.csv",
    models = c("PREVENT", ml_model_order)
  ),
  list(
    id = "clinical_protein_lassonet",
    label = "Clinical + protein predictors",
    file = "all_lassonet_val.csv",
    models = c("PREVENT", ml_model_order)
  )
)

horizon_days <- 365.25 * 10

model_colors <- c(
  PREVENT = "#111111",
  LinearModel = "grey45",
  XGBoost = "#00468BFF",
  MLP = "#FDAF91FF",
  NODE = "#42B540FF",
  TabNet = "#0099B4FF",
  `FT-Transformer` = "#925E9FFF",
  SAINT = "#ED0000FF",
  TabPFN = "#AD002AFF"
)

model_linetypes <- c(
  PREVENT = "solid",
  LinearModel = "solid",
  XGBoost = "solid",
  MLP = "solid",
  TabNet = "solid",
  NODE = "solid",
  `FT-Transformer` = "solid",
  SAINT = "solid",
  TabPFN = "solid"
)

safe_read_prediction <- function(model, outcome, file_name) {
  path <- file.path(prediction_root, model, outcome, file_name)
  if (!file.exists(path)) {
    stop("Prediction file not found: ", path)
  }
  data <- read.csv(path)
  required <- c("time", "event", "prob_10yr")
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    stop("Missing columns in ", path, ": ", paste(missing, collapse = ", "))
  }
  data <- data[, required]
  data$time <- as.numeric(data$time)
  data$event <- as.numeric(data$event)
  data$prob_10yr <- as.numeric(data$prob_10yr)
  data <- data[complete.cases(data), ]
  data
}

prediction_file_for_model <- function(model, set_spec) {
  if (model == "PREVENT") {
    "classic_val.csv"
  } else {
    set_spec$file
  }
}

censoring_survival <- function(time, event, query_times) {
  censor_event <- 1 - event
  fit <- survfit(Surv(time, censor_event) ~ 1)
  survival <- summary(fit, times = query_times, extend = TRUE)$surv
  pmax(survival, 1e-8)
}

time_dependent_roc <- function(data, horizon = horizon_days) {
  cases <- data[data$event == 1 & data$time <= horizon, ]
  controls <- data[data$time > horizon, ]
  if (nrow(cases) == 0 || nrow(controls) == 0) {
    stop("No cases or controls at the requested horizon.")
  }

  case_weights <- 1 / censoring_survival(data$time, data$event, cases$time)
  control_weights <- rep(1, nrow(controls))
  case_scores <- cases$prob_10yr
  control_scores <- controls$prob_10yr

  thresholds <- sort(unique(c(case_scores, control_scores)), decreasing = TRUE)
  thresholds <- c(Inf, thresholds, -Inf)
  tpr <- numeric(length(thresholds))
  fpr <- numeric(length(thresholds))

  for (i in seq_along(thresholds)) {
    threshold <- thresholds[i]
    tpr[i] <- sum(case_weights[case_scores >= threshold]) / sum(case_weights)
    fpr[i] <- sum(control_weights[control_scores >= threshold]) / sum(control_weights)
  }

  ord <- order(fpr, tpr)
  roc <- data.frame(fpr = fpr[ord], tpr = tpr[ord])
  roc <- roc[!duplicated(paste(roc$fpr, roc$tpr)), ]
  roc
}

auc_from_roc <- function(roc) {
  roc <- roc[order(roc$fpr, roc$tpr), ]
  sum(diff(roc$fpr) * (head(roc$tpr, -1) + tail(roc$tpr, -1)) / 2)
}

smooth_roc_curve <- function(roc, n = 501) {
  roc <- roc[order(roc$fpr, roc$tpr), ]
  roc <- aggregate(tpr ~ fpr, data = roc, FUN = max)
  if (min(roc$fpr) > 0) {
    roc <- rbind(data.frame(fpr = 0, tpr = 0), roc)
  }
  if (max(roc$fpr) < 1) {
    roc <- rbind(roc, data.frame(fpr = 1, tpr = 1))
  }

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

  smooth_y <- pmin(pmax(smooth_y, 0), 1)
  smooth_y <- cummax(smooth_y)
  smooth_y[1] <- 0
  smooth_y[length(smooth_y)] <- 1
  smooth_y <- cummax(smooth_y)
  data.frame(fpr = grid_x, tpr = smooth_y)
}

build_results_for_set <- function(set_spec) {
  rows <- list()
  idx <- 1
  for (outcome in outcomes) {
    for (model in set_spec$models) {
      file_name <- prediction_file_for_model(model, set_spec)
      data <- safe_read_prediction(model, outcome, file_name)
      roc <- time_dependent_roc(data)
      smooth_roc <- smooth_roc_curve(roc)
      auc <- auc_from_roc(roc)
      curve_label <- sprintf("%s: AUC = %.3f", model_labels[[model]], auc)

      rows[[idx]] <- list(
        outcome = outcome,
        outcome_label = outcome_labels[[outcome]],
        predictor_set = set_spec$label,
        predictor_set_id = set_spec$id,
        model = model,
        model_label = model_labels[[model]],
        curve_label = curve_label,
        auc_10yr = auc,
        roc = smooth_roc
      )
      idx <- idx + 1
    }
  }
  rows
}

make_plot_data <- function(results) {
  do.call(rbind, lapply(results, function(x) {
    data.frame(
      outcome = x$outcome,
      outcome_label = x$outcome_label,
      predictor_set = x$predictor_set,
      predictor_set_id = x$predictor_set_id,
      model = x$model,
      model_label = x$model_label,
      curve_label = x$curve_label,
      auc_10yr = x$auc_10yr,
      fpr = x$roc$fpr,
      tpr = x$roc$tpr
    )
  }))
}

make_auc_table <- function(results) {
  do.call(rbind, lapply(results, function(x) {
    data.frame(
      outcome = x$outcome_label,
      predictor_set = x$predictor_set,
      model = x$model_label,
      auc_10yr = x$auc_10yr
    )
  }))
}

theme_roc <- function(base_size = 8.5) {
  theme_classic(base_family = "sans", base_size = base_size) +
    theme(
      plot.title = element_blank(),
      axis.title = element_text(face = "bold", size = base_size),
      axis.text = element_text(color = "black", size = base_size - 0.4),
      axis.line = element_line(color = "black", linewidth = 0.45),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      legend.position = c(1.05, 0.05),
      legend.justification = c(1, 0),
      legend.direction = "vertical",
      legend.title = element_blank(),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.key.width = unit(0.50, "cm"),
      legend.key.height = unit(0.26, "cm"),
      legend.text = element_text(size = base_size - 2.1),
      plot.margin = margin(7, 6, 7, 7)
    )
}

build_outcome_plot <- function(plot_data, outcome, set_spec) {
  panel_data <- plot_data[plot_data$outcome == outcome, ]
  panel_models <- set_spec$models
  curve_levels <- vapply(panel_models, function(model) {
    panel_data$curve_label[panel_data$model == model][1]
  }, character(1))
  panel_data$model <- factor(panel_data$model, levels = panel_models)
  panel_data$curve_label <- factor(panel_data$curve_label, levels = curve_levels)

  ggplot(panel_data, aes(x = fpr, y = tpr, color = model, linetype = model, group = model)) +
    geom_abline(intercept = 0, slope = 1, color = "grey76", linewidth = 0.35, linetype = "dashed") +
    geom_line(aes(alpha = model), linewidth = 0.86, lineend = "round") +
    scale_color_manual(
      values = model_colors[panel_models],
      breaks = panel_models,
      labels = curve_levels
    ) +
    scale_linetype_manual(
      values = model_linetypes[panel_models],
      breaks = panel_models,
      labels = curve_levels
    ) +
    scale_alpha_manual(
      values = c(PREVENT = 1, LinearModel = 0.72, XGBoost = 0.72, MLP = 0.72, TabNet = 0.72,
                 NODE = 0.72, `FT-Transformer` = 0.72, SAINT = 0.72, TabPFN = 0.72)[panel_models],
      breaks = panel_models,
      labels = curve_levels
    ) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), labels = sprintf("%.1f", seq(0, 1, 0.2))) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), labels = sprintf("%.1f", seq(0, 1, 0.2))) +
    labs(
      title = NULL,
      x = "1 - Specificity",
      y = "Sensitivity"
    ) +
    guides(
      color = guide_legend(
        ncol = 1,
        override.aes = list(
          linewidth = 1.20,
          linetype = unname(model_linetypes[panel_models]),
          alpha = unname(c(PREVENT = 1, LinearModel = 0.72, XGBoost = 0.72, MLP = 0.72, TabNet = 0.72,
                           NODE = 0.72, `FT-Transformer` = 0.72, SAINT = 0.72, TabPFN = 0.72)[panel_models])
        )
      ),
      linetype = "none",
      alpha = "none"
    ) +
    theme_roc()
}

save_figure <- function(plot, filename_base, width = 12.8, height = 3.25, dpi = 600) {
  png_file <- paste0(filename_base, ".png")
  pdf_file <- paste0(filename_base, ".pdf")

  ggsave(png_file, plot = plot, width = width, height = height, units = "in", dpi = dpi, bg = "white")
  ggsave(pdf_file, plot = plot, width = width, height = height, units = "in", device = cairo_pdf, bg = "white")
}

build_set_output <- function(set_spec) {
  results <- build_results_for_set(set_spec)
  plot_data <- make_plot_data(results)
  auc_table <- make_auc_table(results)

  panels <- list(
    build_outcome_plot(plot_data, "Total_CVD", set_spec),
    build_outcome_plot(plot_data, "ASCVD", set_spec),
    build_outcome_plot(plot_data, "HF", set_spec)
  )
  combined <- arrangeGrob(
    grobs = panels,
    nrow = 1,
    widths = unit(c(1, 1, 1), "null")
  )

  write.csv(
    auc_table,
    file = file.path(output_dir, paste0("ROC_10yr_AUC_all_models_", set_spec$id, ".csv")),
    row.names = FALSE
  )
  write.csv(
    plot_data,
    file = file.path(output_dir, paste0("ROC_10yr_smoothed_curve_data_", set_spec$id, ".csv")),
    row.names = FALSE
  )

  list(auc = auc_table, curves = plot_data, panels = panels)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

set_outputs <- lapply(predictor_sets, build_set_output)
all_auc <- do.call(rbind, lapply(set_outputs, `[[`, "auc"))
all_curves <- do.call(rbind, lapply(set_outputs, `[[`, "curves"))

write.csv(
  all_auc,
  file = file.path(output_dir, "ROC_10yr_AUC_all_models_all_predictor_sets.csv"),
  row.names = FALSE
)
write.csv(
  all_curves,
  file = file.path(output_dir, "ROC_10yr_smoothed_curve_data_all_predictor_sets.csv"),
  row.names = FALSE
)

all_panels <- unlist(lapply(set_outputs, `[[`, "panels"), recursive = FALSE)
combined_3x3 <- arrangeGrob(
  grobs = all_panels,
  nrow = 3,
  ncol = 3,
  widths = unit(c(1, 1, 1), "null"),
  heights = unit(c(1, 1, 1), "null")
)
save_figure(
  combined_3x3,
  file.path(output_dir, "Figure_ROC_10yr_all_models_3x3"),
  width = 12.8,
  height = 9.4
)

message("All-model 10-year ROC figures written to: ", output_dir)
