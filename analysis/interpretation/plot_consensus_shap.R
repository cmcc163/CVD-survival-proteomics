rm(list = ls())

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(ragg)
})

base_dir <- Sys.getenv("CVD_INTERPRETATION_OUTPUT", unset = "results/interpretation")
output_dir <- file.path(base_dir, "combined_circular_components")
input_root <- Sys.getenv("CVD_SHAP_ROOT", unset = "artifacts/shap/final")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

unlink(
  list.files(output_dir, pattern = "^Consensus_SHAP_combined_circular_bar_.*\\.png$", full.names = TRUE),
  force = TRUE
)

outcomes <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(
  "Total_CVD" = "Total CVD",
  "ASCVD" = "ASCVD",
  "HF" = "HF"
)

clinical_label_map <- c(
  "age" = "Age",
  "sex" = "Sex",
  "ever_smoked" = "Smoking status",
  "Diabetes_baseline_1" = "Diabetes",
  "Cholesterol_treatment" = "Lipid-lowering medication",
  "hdl_cholesterol" = "HDL-C",
  "non_hdl_cholesterol" = "Non-HDL-C",
  "hypertension_treatment" = "Antihypertensive medication",
  "average_SBP" = "SBP",
  "eGFR_SCysC" = "eGFR",
  "BMI" = "BMI"
)

feature_type_cols <- c(
  "Clinical variable" = "#0baf64",
  "Protein marker" = "gray12"
)

shap_cols <- c("#1D5FA8", "#5967B8", "#8C4D95", "#C83E70", "#EF3B2C")

read_rank <- function(outcome, set_name) {
  path <- file.path(
    input_root,
    outcome,
    set_name,
    sprintf("Selection_Results_%s_%s.xlsx", outcome, set_name)
  )
  if (!file.exists(path)) {
    stop("Missing SHAP ranking file: ", path)
  }

  read_excel(path, sheet = "1_Ensemble_SHAP_Rank") %>%
    transmute(
      Feature = as.character(Feature),
      Rank = as.integer(Rank),
      SHAP_pct = as.numeric(Ensemble_Mean_SHAP_pct),
      Cumulative_SHAP_pct = as.numeric(Cumulative_SHAP_pct)
    )
}

prepare_outcome <- function(outcome) {
  classic <- read_rank(outcome, "Classic")
  combined <- read_rank(outcome, "Combined")
  clinical_features <- classic %>% arrange(Rank) %>% pull(Feature)

  combined %>%
    arrange(Rank) %>%
    mutate(
      Outcome = outcome,
      OutcomeLabel = unname(outcome_labels[[outcome]]),
      FeatureType = if_else(Feature %in% clinical_features, "Clinical variable", "Protein marker"),
      FeatureType = factor(FeatureType, levels = names(feature_type_cols)),
      DisplayFeature = if_else(
        Feature %in% names(clinical_label_map),
        unname(clinical_label_map[Feature]),
        Feature
      ),
      id = row_number()
    )
}

label_geometry <- function(df) {
  n <- nrow(df)
  df %>%
    mutate(
      label_angle_raw = 90 - 360 * (id - 0.5) / n,
      label_angle = if_else(label_angle_raw < -90, label_angle_raw + 180, label_angle_raw),
      label_hjust = if_else(label_angle_raw < -90, 1, 0)
    )
}

make_circular_bar <- function(df, global_max) {
  df <- label_geometry(df)
  inner_radius <- 3.8
  bar_height_max <- 7.2

  df <- df %>%
    mutate(
      y_inner = inner_radius,
      y_outer = inner_radius + bar_height_max * SHAP_pct / global_max,
      label_radius = y_outer + 0.55
    )

  ggplot(df, aes(x = id)) +
    geom_rect(
      aes(
        xmin = id - 0.47,
        xmax = id + 0.47,
        ymin = y_inner,
        ymax = y_outer,
        fill = SHAP_pct
      ),
      color = "white",
      linewidth = 0.16
    ) +
    geom_text(
      aes(
        y = label_radius,
        label = DisplayFeature,
        angle = label_angle,
        hjust = label_hjust,
        color = FeatureType
      ),
      family = "Arial",
      fontface = "bold",
      size = 2.45,
      lineheight = 0.84
    ) +
    scale_color_manual(values = feature_type_cols, guide = "none") +
    scale_fill_gradientn(
      colours = shap_cols,
      limits = c(0, global_max),
      breaks = pretty_breaks(n = 5)(c(0, global_max)),
      name = "Normalized mean |SHAP| (%)"
    ) +
    coord_polar(start = 0, clip = "off") +
    scale_x_continuous(limits = c(0.5, nrow(df) + 0.5), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, inner_radius + bar_height_max + 0.95), expand = c(0, 0)) +
    theme_void(base_family = "Arial") +
    theme(
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.background = element_rect(fill = "transparent", color = NA),
      legend.box.background = element_rect(fill = "transparent", color = NA),
      legend.key = element_rect(fill = "transparent", color = NA),
      # legend.position = "right",
      legend.position = c(1.90, 0.50),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10.5),
      legend.key.height = unit(6, "mm"),
      legend.key.width = unit(7, "mm"),
      legend.box = "vertical",
      plot.margin = margin(28, 46, 28, 46)
    )
}

prepared <- lapply(outcomes, prepare_outcome)
names(prepared) <- outcomes
global_max <- ceiling(max(bind_rows(prepared)$SHAP_pct, na.rm = TRUE))

for (outcome in outcomes) {
  ggsave(
    filename = file.path(output_dir, sprintf("Consensus_SHAP_combined_circular_bar_%s.png", outcome)),
    plot = make_circular_bar(prepared[[outcome]], global_max),
    width = 9.0,
    height = 9.0,
    units = "in",
    dpi = 600,
    bg = "transparent",
    device = ragg::agg_png
  )
}

cat("Wrote circular bar PNGs to:", output_dir, "\n")
