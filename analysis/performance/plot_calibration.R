rm(list = ls())

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(stringr)
})

base_dir <- Sys.getenv("CVD_PERFORMANCE_RESULTS", unset = "results/performance")
calibration_file <- file.path(base_dir, "calibration_evaluation", "Calibration_metrics_R_bootstrap_validation_sets.xlsx")
output_dir <- file.path(base_dir, "figure_calibration")

outcome_levels <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(
  "Total_CVD" = "Total CVD",
  "ASCVD" = "ASCVD",
  "HF" = "HF"
)
validation_levels <- c("European hold-out set", "Asian ancestry", "Other ancestry")
validation_folder <- c(
  "European hold-out set" = "European_holdout",
  "Asian ancestry" = "Asian_ancestry",
  "Other ancestry" = "Other_ancestry"
)
validation_file <- c(
  "European hold-out set" = "European_holdout",
  "Asian ancestry" = "Asian_ancestry",
  "Other ancestry" = "Other_ancestry"
)

model_levels <- c(
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

feature_levels <- c("Classic", "Protein", "Combined")
feature_labels <- c(
  "Classic" = "Clinical-only",
  "Protein" = "Proteomic-only",
  "Combined" = "Combined clinical-plus-proteomic"
)


feature_table_labels <- c(
  "Classic" = "Clinical",
  "Protein" = "Proteomic",
  "Combined" = "Combined"
)
feature_colors <- c(
  "Classic" = "#2166AC",
  "Protein" = "#1B9E77",
  "Combined" = "#D73027"
)
feature_shapes <- c(
  "Classic" = 16,
  "Protein" = 17,
  "Combined" = 15
)

log_tick_base <- 10

format_metric <- function(x, digits) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

format_log_percent <- function(x) {
  ifelse(
    x < 0.001,
    paste0(formatC(x * 100, format = "f", digits = 3), "%"),
    ifelse(
    x < 0.01,
    paste0(formatC(x * 100, format = "f", digits = 2), "%"),
    paste0(formatC(x * 100, format = "f", digits = 0), "%")
    )
  )
}

build_log_breaks <- function(lower, upper, base = log_tick_base) {
  exponents <- ceiling(log(lower, base = base)):floor(log(upper, base = base))
  base^exponents
}

build_log_limits <- function(x, base = log_tick_base, pad = 0.08) {
  x <- x[is.finite(x) & x > 0]
  if (length(x) == 0) {
    return(c(0.001, 0.30))
  }
  log_x <- log(x, base = base)
  span <- diff(range(log_x))
  if (!is.finite(span) || span == 0) {
    span <- 1
  }
  base^c(min(log_x) - span * pad, max(log_x) + span * pad)
}

build_model_label <- function(sub_df) {
  sub_df <- sub_df %>%
    filter(!is.na(FeatureSet)) %>%
    mutate(FeatureSet = factor(FeatureSet, levels = feature_levels)) %>%
    arrange(FeatureSet)

  if (nrow(sub_df) == 0) {
    return("")
  }

  lines <- c(
    "ICI by group",
    "Feature      Overall    Low   High",
    "--------------------------------"
  )
  for (feature in feature_levels) {
    df_f <- sub_df %>% filter(FeatureSet == feature)
    if (nrow(df_f) == 0) next
    feature_txt <- str_pad(feature_table_labels[[feature]], 9, side = "right")
    row_all <- df_f %>% filter(Group == "Overall")
    row_low <- df_f %>% filter(Group == "Low")
    row_high <- df_f %>% filter(Group == "High")

    ici_all <- if (nrow(row_all) > 0) format_metric(row_all$ici[1], 3) else "NA"
    ici_low <- if (nrow(row_low) > 0) format_metric(row_low$ici[1], 3) else "NA"
    ici_high <- if (nrow(row_high) > 0) format_metric(row_high$ici[1], 3) else "NA"

    lines <- c(
      lines,
      paste0(
        feature_txt, "  ",
        str_pad(ici_all, 7, side = "left"), "  ",
        str_pad(ici_low, 5, side = "left"), "  ",
        str_pad(ici_high, 5, side = "left")
      )
    )
  }
  paste(lines, collapse = "\n")
}

smooth_one_curve <- function(df) {
  df <- df %>%
    arrange(pred) %>%
    filter(
      is.finite(pred),
      is.finite(obs),
      is.finite(obs_lower),
      is.finite(obs_upper)
    )

  if (nrow(df) < 4) {
    return(data.frame())
  }

  x_grid <- seq(min(df$pred, na.rm = TRUE), max(df$pred, na.rm = TRUE), length.out = 120)

  fit_loess <- function(y) {
    fitted <- tryCatch(
      loess(y ~ pred, data = df, weights = pmax(df$N, 1), span = 0.95, degree = 1, family = "symmetric"),
      error = function(e) NULL
    )
    fallback <- approx(df$pred, y, xout = x_grid, ties = "ordered", rule = 2)$y
    if (is.null(fitted)) {
      return(fallback)
    }
    predicted <- predict(fitted, newdata = data.frame(pred = x_grid))
    if (length(predicted) != length(x_grid)) {
      fallback
    } else {
      ifelse(is.na(predicted), fallback, predicted)
    }
  }

  y <- fit_loess(df$obs)
  lower <- fit_loess(df$obs_lower)
  upper <- fit_loess(df$obs_upper)

  y <- pmin(pmax(y, 0), 1)
  lower <- pmin(pmax(lower, 0), y)
  upper <- pmax(pmin(pmax(upper, 0), 1), y)

  data.frame(
    pred = x_grid,
    obs = y,
    obs_lower = lower,
    obs_upper = upper,
    Outcome = df$Outcome[1],
    ValidationSet = df$ValidationSet[1],
    Model = df$Model[1],
    ModelLabel = df$ModelLabel[1],
    FeatureSet = df$FeatureSet[1]
  )
}

build_smooth_curves <- function(plot_df, axis_limit) {
  grouped <- split(plot_df, list(plot_df$ModelLabel, plot_df$FeatureSet), drop = TRUE)
  do.call(rbind, lapply(grouped, smooth_one_curve)) %>%
    mutate(
      ModelLabel = factor(ModelLabel, levels = model_labels[model_levels]),
      FeatureSet = factor(FeatureSet, levels = feature_levels)
    )
}

make_calibration_plot <- function(curve_df, metric_df, outcome_name, validation_name) {
  plot_df <- curve_df %>%
    filter(Outcome == outcome_name, ValidationSet == validation_name) %>%
    mutate(
      Model = factor(Model, levels = model_levels),
      ModelLabel = factor(model_labels[as.character(Model)], levels = model_labels[model_levels]),
      FeatureSet = factor(FeatureSet, levels = feature_levels)
    )

  metric_sub <- metric_df %>%
    filter(Outcome == outcome_name, ValidationSet == validation_name) %>%
    mutate(
      Model = factor(Model, levels = model_levels),
      ModelLabel = factor(model_labels[as.character(Model)], levels = model_labels[model_levels]),
      FeatureSet = factor(FeatureSet, levels = feature_levels)
    )

  text_labels <- metric_sub %>%
    group_by(ModelLabel) %>%
    group_modify(~ data.frame(label_text = build_model_label(.x))) %>%
    ungroup()

  positive_axis_values <- c(plot_df$pred, plot_df$obs, 0.01, 0.10)
  positive_axis_values <- positive_axis_values[is.finite(positive_axis_values) & positive_axis_values > 0]
  axis_limits <- build_log_limits(positive_axis_values)
  axis_lower <- axis_limits[1]
  axis_upper <- axis_limits[2]
  zero_break <- axis_lower
  axis_plot_lower <- log_tick_base^(log(zero_break, base = log_tick_base) - 0.10)
  positive_breaks <- c(0.01, 0.10)
  axis_breaks <- c(zero_break, positive_breaks)
  axis_labels <- function(x) {
    ifelse(
      abs(log(x / zero_break, base = log_tick_base)) < 1e-8,
      "0",
      ifelse(
        abs(x - 0.01) < 1e-10,
        "1%",
        ifelse(abs(x - 0.10) < 1e-10, "10%", format_log_percent(x))
      )
    )
  }
  log_span <- log(axis_upper, base = log_tick_base) - log(axis_lower, base = log_tick_base)
  plot_df <- plot_df %>%
    mutate(
      pred_plot = pmax(pred, zero_break),
      obs_plot = pmax(obs, zero_break)
    )
  text_labels$x_pos <- log_tick_base^(log(axis_lower, base = log_tick_base) + log_span * 0.035)
  text_labels$y_pos <- log_tick_base^(log(axis_upper, base = log_tick_base) - log_span * 0.035)
  ggplot(plot_df, aes(x = pred_plot, y = obs_plot, color = FeatureSet, shape = FeatureSet)) +
    geom_hline(yintercept = axis_plot_lower, color = "black", linewidth = 0.55) +
    geom_vline(xintercept = axis_plot_lower, color = "black", linewidth = 0.55) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray45", linewidth = 0.7) +
    geom_smooth(
      data = function(d) d %>% filter(obs > 0),
      aes(fill = FeatureSet, group = FeatureSet),
      method = "loess",
      formula = y ~ x,
      se = TRUE,
      span = 2.0,
      linewidth = 0.8,
      alpha = 0.05,
      lineend = "round",
      show.legend = c(fill = FALSE, color = TRUE)
    ) +
    geom_point(size = 2.5, alpha = 1, stroke = 0) +
    geom_label(
      data = text_labels,
      aes(x = x_pos, y = y_pos, label = label_text),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 1,
      size = 2.7,
      lineheight = 0.92,
      family = "mono",
      color = "gray10",
      fill = "white",
      linewidth = 0,
      label.padding = unit(0.04, "lines"),
      label.r = unit(0, "lines")
    ) +
    facet_wrap(~ ModelLabel, ncol = 3, drop = FALSE) +
    coord_cartesian(xlim = c(axis_plot_lower, axis_upper), ylim = c(axis_plot_lower, axis_upper), expand = FALSE) +
    scale_x_continuous(
      trans = log_trans(base = log_tick_base),
      labels = axis_labels,
      breaks = axis_breaks,
      guide = guide_axis(check.overlap = TRUE)
    ) +
    scale_y_continuous(
      trans = log_trans(base = log_tick_base),
      labels = axis_labels,
      breaks = axis_breaks
    ) +
    scale_color_manual(values = feature_colors, labels = feature_labels, name = "Predictor set") +
    scale_fill_manual(values = feature_colors, labels = feature_labels, name = "Predictor set") +
    scale_shape_manual(values = feature_shapes, labels = feature_labels, name = "Predictor set") +
    guides(
      color = guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(linewidth = 1.4, alpha = 1, fill = NA)
      ),
      fill = "none",
      shape = guide_legend(nrow = 1, byrow = TRUE)
    ) +
    labs(x = "Predicted 10-year risk", y = "Observed 10-year risk") +
    theme_bw(base_family = "Arial", base_size = 12) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      panel.border = element_blank(),
      panel.grid.major = element_line(color = "#E6E6E6", linewidth = 0.55),
      panel.grid.minor = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 13.5, color = "gray10"),
      axis.title = element_text(face = "bold", size = 15, color = "black"),
      axis.text = element_text(size = 11.2, color = "gray20"),
      axis.text.x = element_text(size = 10.2, color = "gray20"),
      axis.line = element_line(color = "black", linewidth = 0.55),
      axis.ticks = element_line(color = "black", linewidth = 0.55),
      panel.spacing.x = unit(7, "pt"),
      panel.spacing.y = unit(7, "pt"),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_text(face = "bold", size = 13),
      legend.text = element_text(size = 12.5),
      legend.key.width = unit(1.0, "cm"),
      legend.key = element_rect(fill = "white", color = NA),
      legend.box.margin = margin(t = 4, r = 0, b = 0, l = 0),
      plot.margin = margin(8, 8, 8, 8)
    )
}

curve_df <- read_excel(calibration_file, sheet = "Calibration_curve_deciles")
metric_df <- read_excel(calibration_file, sheet = "Metrics_long")

for (validation_name in validation_levels) {
  sub_dir <- file.path(output_dir, validation_folder[[validation_name]])
  dir.create(sub_dir, recursive = TRUE, showWarnings = FALSE)

  for (outcome_name in outcome_levels) {
    p <- make_calibration_plot(curve_df, metric_df, outcome_name, validation_name)
    file_stub <- paste0(
      "Calibration_3x3_",
      outcome_name,
      "_",
      validation_file[[validation_name]]
    )
    ggsave(
      filename = file.path(sub_dir, paste0(file_stub, ".png")),
      plot = p,
      width = 12,
      height = 12,
      units = "in",
      dpi = 1000,
      bg = "white"
    )
  }
}

readme_path <- file.path(output_dir, "文件说明.txt")
writeLines(
  c(
    "校准曲线图输出说明",
    "",
    "1. plot_calibration_3x3.R",
    "   使用Calibration_metrics_R_bootstrap_validation_sets.xlsx中的校准曲线十分位数据和校准指标绘制3x3模型校准曲线图。",
    "",
    "2. 子文件夹",
    "   European_holdout：欧洲hold-out验证集。",
    "   Asian_ancestry：亚洲祖源验证集。",
    "   Other_ancestry：其他祖源验证集。",
    "",
    "3. 图形内容",
    "   每个验证集文件夹包含Total CVD、ASCVD和HF三个结局的3x3校准曲线图。",
    "   模型顺序为PREVENT equation、Refitted Cox model、XGBoost、MLP、TabNet、NODE、FT-Transformer、SAINT、TabPFN。",
    "   每个分面内的点为10等分风险组，虚线为理想校准线；左上角显示O:E ratio、calibration slope和ICI点估计。"
  ),
  readme_path,
  useBytes = TRUE
)

cat("Calibration figures written to:", output_dir, "\n")
