rm(list = ls())

suppressPackageStartupMessages({
  library(survival)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(readr)
  library(tidyr)
})

base_path <- Sys.getenv("CVD_PREDICTION_ROOT", unset = "artifacts/predictions")
project_dir <- Sys.getenv("CVD_PERFORMANCE_RESULTS", unset = "results/performance")
output_root <- file.path(project_dir, "figure_clinical_utility")
data_out <- file.path(project_dir, "clinical_utility_evaluation", "dca_curve_10yr_all_validation_sets.csv")
annotation_out <- file.path(project_dir, "clinical_utility_evaluation", "net_benefit_10yr_thresholds_long.csv")

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(
  "Total_CVD" = "Total CVD",
  "ASCVD" = "ASCVD",
  "HF" = "HF"
)

validations <- c(
  "European hold-out set" = "val",
  "Asian ancestry" = "asian",
  "Other ancestry" = "other"
)
validation_folder <- c(
  "European hold-out set" = "European_holdout",
  "Asian ancestry" = "Asian_ancestry",
  "Other ancestry" = "Other_ancestry"
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
  "Clinical predictors" = "classic_%s.csv",
  "Protein markers" = "protein_lassonet_%s.csv",
  "Clinical + protein markers" = "all_lassonet_%s.csv"
)
feature_levels <- names(feature_files)

curve_colors <- c(
  "Clinical predictors" = "#2166AC",
  "Protein markers" = "#1B9E77",
  "Clinical + protein markers" = "#D73027",
  "PREVENT equation" = "#7B3294",
  "Treat all" = "#6B6B6B",
  "Treat none" = "#222222"
)
curve_linetypes <- c(
  "Clinical predictors" = "solid",
  "Protein markers" = "solid",
  "Clinical + protein markers" = "solid",
  "PREVENT equation" = "solid",
  "Treat all" = "22",
  "Treat none" = "solid"
)
legend_labels <- c(
  "Clinical predictors" = "Clinical-only",
  "Protein markers" = "Proteomic-only",
  "Clinical + protein markers" = "Combined clinical-plus-proteomic",
  "PREVENT equation" = "PREVENT equation",
  "Treat all" = "Treat all",
  "Treat none" = "Treat none"
)

target_time <- 3652.5
thresholds <- seq(0, 0.15, by = 0.001)
annotation_threshold <- 0.10
bootstrap_n <- as.integer(Sys.getenv("BOOTSTRAP_B", unset = "1000"))

load_prediction <- function(model, outcome, feature_set, suffix) {
  if (model == "PREVENT" && feature_set != "Clinical predictors") {
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

  bind_rows(lapply(thresholds, function(pt) {
    treat <- data$prob_10yr >= pt
    tp_rate <- sum(weights[case & treat]) / n
    fp_rate <- sum(weights[control & treat]) / n
    data.frame(
      Threshold = pt,
      NetBenefit = tp_rate - fp_rate * pt / (1 - pt),
      TreatAll = event_risk - non_event_risk * pt / (1 - pt),
      TreatNone = 0
    )
  }))
}

build_dca_data <- function() {
  rows <- list()
  idx <- 1
  for (validation_name in names(validations)) {
    suffix <- validations[[validation_name]]
    for (outcome in outcomes) {
      for (feature_set in feature_levels) {
        for (model in models) {
          data <- load_prediction(model, outcome, feature_set, suffix)
          if (is.null(data)) next
          nb <- net_benefit_ipcw(data, thresholds, target_time) %>%
            mutate(
              Outcome = outcome,
              OutcomeLabel = unname(outcome_labels[[outcome]]),
              ValidationSet = validation_name,
              PredictorSet = feature_set,
              Model = model,
              ModelLabel = unname(model_labels[[model]])
            )
          rows[[idx]] <- nb
          idx <- idx + 1
        }
      }
    }
  }
  bind_rows(rows)
}

dca_data <- build_dca_data()
write_csv(dca_data, data_out)

bootstrap_ci <- function(data, threshold, t0, b = bootstrap_n) {
  n <- nrow(data)
  boot_nb <- rep(NA_real_, b)
  for (i in seq_len(b)) {
    idx <- sample.int(n, n, replace = TRUE)
    boot_data <- data[idx, , drop = FALSE]
    boot_nb[i] <- net_benefit_ipcw(boot_data, threshold, t0)$NetBenefit
  }
  c(
    NetBenefit_L = unname(quantile(boot_nb, probs = 0.025, na.rm = TRUE)),
    NetBenefit_U = unname(quantile(boot_nb, probs = 0.975, na.rm = TRUE))
  )
}

build_annotation_data <- function() {
  rows <- list()
  idx <- 1
  for (validation_name in names(validations)) {
    suffix <- validations[[validation_name]]
    for (outcome in outcomes) {
      for (feature_set in feature_levels) {
        for (model in models) {
          data <- load_prediction(model, outcome, feature_set, suffix)
          if (is.null(data)) next
          set.seed(221 + idx)
          nb <- net_benefit_ipcw(data, annotation_threshold, target_time)
          ci <- bootstrap_ci(data, annotation_threshold, target_time, bootstrap_n)
          rows[[idx]] <- data.frame(
            Outcome = outcome,
            OutcomeLabel = unname(outcome_labels[[outcome]]),
            ValidationSet = validation_name,
            PredictorSet = feature_set,
            Model = model,
            ModelLabel = unname(model_labels[[model]]),
            Threshold = annotation_threshold,
            N = nrow(data),
            Events = sum(data$event == 1 & data$time <= target_time),
            NetBenefit = nb$NetBenefit,
            NetBenefit_L = ci[["NetBenefit_L"]],
            NetBenefit_U = ci[["NetBenefit_U"]]
          )
          idx <- idx + 1
        }
      }
    }
  }
  bind_rows(rows)
}

if (file.exists(annotation_out)) {
  annotation_data <- read_csv(annotation_out, show_col_types = FALSE)
} else {
  annotation_data <- build_annotation_data()
  write_csv(annotation_data, annotation_out)
}

make_dca_matrix <- function(data, outcome, validation_name) {
  model_level_labels <- unname(model_labels[models])
  legend_breaks <- c(feature_levels, "PREVENT equation", "Treat all", "Treat none")

  sub <- data %>%
    filter(Outcome == outcome, ValidationSet == validation_name) %>%
    mutate(
      PredictorSet = factor(PredictorSet, levels = feature_levels),
      ModelLabel = factor(ModelLabel, levels = model_level_labels),
      Curve = as.character(PredictorSet)
    )

  ref_base <- sub %>%
    filter(PredictorSet == "Clinical predictors", Model == "PREVENT") %>%
    select(Threshold, TreatAll, TreatNone) %>%
    distinct() %>%
    pivot_longer(cols = c(TreatAll, TreatNone), names_to = "Curve", values_to = "NetBenefit") %>%
    mutate(Curve = recode(Curve, TreatAll = "Treat all", TreatNone = "Treat none"))

  ref <- tidyr::crossing(
    ref_base,
    ModelLabel = factor(model_level_labels, levels = model_level_labels)
  )

  prevent_base <- sub %>%
    filter(PredictorSet == "Clinical predictors", Model == "PREVENT") %>%
    transmute(
      Threshold,
      NetBenefit,
      Curve = "PREVENT equation"
    ) %>%
    distinct()

  prevent_ref <- tidyr::crossing(
    prevent_base,
    ModelLabel = factor(model_level_labels, levels = model_level_labels)
  )

  plot_data <- sub %>%
    filter(Model != "PREVENT") %>%
    select(Threshold, NetBenefit, Curve, ModelLabel)

  max_y <- max(c(plot_data$NetBenefit, prevent_ref$NetBenefit, ref$NetBenefit, 0), na.rm = TRUE)
  upper_y <- max(0.02, ceiling(max_y * 100) / 100)

  annot <- annotation_data %>%
    filter(Outcome == outcome, ValidationSet == validation_name, abs(Threshold - annotation_threshold) < 1e-10) %>%
    mutate(
      PredictorSet = recode(
        PredictorSet,
        "Clinical predictors only" = "Clinical predictors",
        "Protein markers only" = "Protein markers",
        "Clinical predictors plus protein markers" = "Clinical + protein markers"
      ),
      ModelLabel = factor(ModelLabel, levels = model_level_labels),
      Curve = ifelse(Model == "PREVENT", "PREVENT equation", PredictorSet),
      Curve = factor(Curve, levels = legend_breaks),
      ShortLabel = recode(
        PredictorSet,
        "Clinical predictors" = "Clinical",
        "Protein markers" = "Proteomic",
        "Clinical + protein markers" = "Combined"
      ),
      ShortLabel = ifelse(Model == "PREVENT", "PREVENT", ShortLabel),
      Label = sprintf("%s %.4f (%.4f-%.4f)", ShortLabel, NetBenefit, NetBenefit_L, NetBenefit_U)
    ) %>%
    filter(
      (Model == "PREVENT" & PredictorSet == "Clinical predictors") |
        (Model != "PREVENT" & PredictorSet %in% feature_levels)
    ) %>%
    arrange(ModelLabel, Curve) %>%
    group_by(ModelLabel) %>%
    mutate(
      x = 0.148,
      y = upper_y - 0.002 - (row_number() - 1) * max(0.0025, (upper_y + 0.01) * 0.065)
    ) %>%
    ungroup()

  ggplot() +
    geom_hline(yintercept = -0.01, color = "black", linewidth = 0.55) +
    geom_vline(xintercept = 0, color = "black", linewidth = 0.55) +
    geom_hline(yintercept = 0, color = "gray60", linewidth = 0.55) +
    geom_line(
      data = ref,
      aes(x = Threshold, y = NetBenefit, color = Curve, linetype = Curve, group = Curve),
      linewidth = 0.75,
      alpha = 0.95
    ) +
    geom_line(
      data = prevent_ref,
      aes(x = Threshold, y = NetBenefit, color = Curve, linetype = Curve, group = Curve),
      linewidth = 0.90,
      alpha = 0.95
    ) +
    geom_line(
      data = plot_data,
      aes(x = Threshold, y = NetBenefit, color = Curve, linetype = Curve, group = Curve),
      linewidth = 0.82,
      alpha = 0.98
    ) +
    geom_vline(xintercept = c(0.05, 0.075, 0.10), linetype = "dotted", color = "gray55", linewidth = 0.40) +
    geom_text(
      data = annot,
      aes(x = x, y = y, label = Label, color = Curve),
      hjust = 1,
      vjust = 1,
      size = 2.35,
      lineheight = 0.9,
      show.legend = FALSE
    ) +
    facet_wrap(~ ModelLabel, ncol = 3, drop = FALSE) +
    coord_cartesian(ylim = c(-0.01, upper_y), expand = FALSE) +
    scale_x_continuous(
      limits = c(0, 0.16),
      breaks = c(0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15),
      labels = c("0%", "2.5%", "5%", "7.5%", "10%", "12.5%", "15%")
    ) +
    scale_y_continuous(labels = number_format(accuracy = 0.01)) +
    scale_color_manual(
      values = curve_colors,
      breaks = legend_breaks,
      labels = unname(legend_labels[legend_breaks]),
      name = NULL
    ) +
    scale_linetype_manual(
      values = curve_linetypes,
      breaks = legend_breaks,
      name = NULL
    ) +
    guides(
      color = guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(linewidth = 1.2, linetype = unname(curve_linetypes[legend_breaks]))
      ),
      linetype = "none"
    ) +
    labs(x = "10-year risk threshold", y = "Net benefit") +
    theme_bw(base_family = "sans", base_size = 12) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.border = element_blank(),
      panel.grid.major = element_line(color = "#E7E7E7", linewidth = 0.45),
      panel.grid.minor = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 12.5, color = "gray10"),
      panel.spacing = unit(0.55, "lines"),
      axis.title = element_text(face = "bold", size = 14, color = "black"),
      axis.text = element_text(size = 10.2, color = "gray20"),
      axis.text.x = element_text(angle = 35, hjust = 1),
      axis.line = element_line(color = "black", linewidth = 0.55),
      axis.ticks = element_line(color = "black", linewidth = 0.5),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.text = element_text(size = 10.5),
      legend.key.width = unit(1.8, "lines"),
      legend.spacing.x = unit(0.45, "lines"),
      plot.margin = margin(8, 8, 8, 8)
    )
}

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
old_png <- list.files(output_root, pattern = "\\.png$", recursive = TRUE, full.names = TRUE)
if (length(old_png) > 0) file.remove(old_png)

for (validation_name in names(validations)) {
  sub_dir <- file.path(output_root, validation_folder[[validation_name]])
  dir.create(sub_dir, recursive = TRUE, showWarnings = FALSE)
  for (outcome in outcomes) {
    p <- make_dca_matrix(dca_data, outcome, validation_name)
    file_stub <- paste0("DCA_10yr_model_matrix_", outcome, "_", validation_folder[[validation_name]])
    ggsave(
      file.path(sub_dir, paste0(file_stub, ".png")),
      plot = p,
      width = 12,
      height = 10.5,
      units = "in",
      dpi = 600,
      bg = "white"
    )
  }
}

writeLines(
  c(
    "临床效用DCA图输出说明",
    "",
    "1. plot_dca_curves.R",
    "   使用saved_predictions中的prob_10yr、time和event计算并绘制10年decision curve analysis曲线。",
    "",
    "2. 输出结构",
    "   European_holdout、Asian_ancestry、Other_ancestry三个子文件夹分别保存三个结局的3×3模型矩阵图。",
    "   每个子图对应一个模型；子图内展示Treat all、Treat none、临床预测变量、蛋白标志物、临床+蛋白标志物的net benefit曲线。",
    "   PREVENT equation仅展示临床预测变量曲线。",
    "   每个子图右上角标注10%阈值下的net benefit及其bootstrap 95% CI。",
    "",
    "3. 图形口径",
    "   阈值范围为0.5%-15%；虚线垂直线标记5%、7.5%和10%阈值。",
    "   y轴下限固定为-0.01，以突出主要临床相关阈值区间内的net benefit差异。",
    "   Net benefit采用10年IPCW删失处理；Treat all和Treat none作为参考策略。"
  ),
  file.path(output_root, "文件说明.txt"),
  useBytes = TRUE
)

cat("DCA 3x3 model-matrix figures written to:", output_root, "\n")
