rm(list = ls())
options(stringsAsFactors = FALSE)

required_packages <- c("readxl", "dplyr", "tidyr", "ggplot2", "ggpubr", "ggsci", "scales", "stringr", "ragg")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) stop("Install required packages: ", paste(missing_packages, collapse = ", "))

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(ggsci)
  library(scales)
  library(stringr)
  library(ragg)
})

script_arg <- commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))]
output_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/"))
} else {
  getwd()
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_file <- file.path(output_dir, "Candidate_predictor_interactions_all_lassonet.xlsx")
if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

outcome_levels <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(Total_CVD = "Total CVD", ASCVD = "ASCVD", HF = "HF")
model_order <- c("XGBoost", "MLP", "TabNet", "NODE", "FT-Transformer", "SAINT", "TabPFN")

lancet_cols <- ggsci::pal_lancet()(7)
model_colors <- c(
  "XGBoost" = lancet_cols[1],
  "MLP" = lancet_cols[6],
  "TabNet" = lancet_cols[4],
  "NODE" = lancet_cols[3],
  "FT-Transformer" = lancet_cols[5],
  "SAINT" = lancet_cols[2],
  "TabPFN" = lancet_cols[7]
)

pair_type_palette <- c(
  "Clinical-clinical" = "#4D4D4D",
  "Clinical-protein" = "#B55246",
  "Protein-protein" = "#0F4D92",
  "Other" = "#8A8A8A"
)

heat_palette <- c("#F8F8F8", "#F3D6D0", "#DE8F83", "#B55246", "#5E1F1A")

theme_interaction <- function(base_size = 8.2) {
  theme_classic(base_family = "Arial", base_size = base_size) +
    theme(
      axis.line = element_line(linewidth = 0.45, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.text = element_text(colour = "black"),
      axis.title = element_text(face = "bold"),
      strip.background = element_rect(fill = "#F1F3F6", colour = "black", linewidth = 0.6),
      strip.text = element_text(face = "bold", size = base_size + 1.4),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_text(face = "bold"),
      legend.key.width = grid::unit(0.56, "cm"),
      legend.key.height = grid::unit(0.32, "cm"),
      panel.grid.major = element_line(linewidth = 0.18, colour = "grey90"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(5, 6, 5, 6)
    )
}

save_all_formats <- function(plot, name, width, height) {
  png_file <- file.path(output_dir, paste0(name, ".png"))
  ragg::agg_png(png_file, width = width, height = height, units = "in", res = 600, background = "white")
  print(plot)
  dev.off()

  ggsave(
    filename = file.path(output_dir, paste0(name, ".pdf")),
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
}

wrap_pair <- function(x, width = 28) {
  x <- stringr::str_replace_all(x, " x ", "\n x ")
  stringr::str_wrap(x, width = width)
}

details <- readxl::read_excel(input_file, sheet = "model_pair_details") %>%
  mutate(
    Outcome = factor(Outcome, levels = outcome_levels),
    OutcomeLabel = factor(OutcomeLabel, levels = outcome_labels[outcome_levels]),
    Model = factor(Model, levels = model_order),
    PairType = factor(PairType, levels = names(pair_type_palette)),
    Top10InModel = ModelRank <= 10,
    Top20InModel = ModelRank <= 20
  )

consensus <- readxl::read_excel(input_file, sheet = "consensus_interactions") %>%
  mutate(
    Outcome = factor(Outcome, levels = outcome_levels),
    OutcomeLabel = factor(OutcomeLabel, levels = outcome_labels[outcome_levels]),
    PairType = factor(PairType, levels = names(pair_type_palette))
  )

build_top_figures <- function(top_n) {
  top_col <- paste0("Top", top_n, "InModel")
  support_label <- paste0("Top ", top_n, " model support")

  top_pairs <- consensus %>%
    arrange(Outcome, ConsensusRank) %>%
    group_by(Outcome) %>%
    slice_head(n = top_n) %>%
    ungroup() %>%
    select(Outcome, OutcomeLabel, PairKey, Pair, PairType, ConsensusRank, MeanInteractionSharePct) %>%
    left_join(
      details %>%
        group_by(Outcome, PairKey) %>%
        summarise(ModelSupport = sum(.data[[top_col]], na.rm = TRUE), .groups = "drop"),
      by = c("Outcome", "PairKey")
    ) %>%
    mutate(
      PairPlot = paste0(as.character(Outcome), "__", sprintf("%02d", ConsensusRank), "__", Pair),
      PairLabel = wrap_pair(Pair, ifelse(top_n <= 10, 36, 33))
    ) %>%
    arrange(Outcome, desc(ConsensusRank)) %>%
    mutate(PairPlot = factor(PairPlot, levels = unique(PairPlot)))

  heat_data <- details %>%
    inner_join(top_pairs %>% select(Outcome, PairKey, Pair, ConsensusRank, PairPlot, PairLabel), by = c("Outcome", "PairKey", "Pair")) %>%
    group_by(Outcome, OutcomeLabel, PairPlot, PairLabel, ConsensusRank, Model) %>%
    summarise(InteractionSharePct = mean(InteractionSharePct, na.rm = TRUE), .groups = "drop") %>%
    mutate(PairPlot = factor(PairPlot, levels = levels(top_pairs$PairPlot)))

  heat_breaks <- pretty(c(0, quantile(heat_data$InteractionSharePct, 0.98, na.rm = TRUE)), n = 5)

  p_heat <- ggplot(heat_data, aes(x = Model, y = PairPlot, fill = InteractionSharePct)) +
    geom_tile(colour = "white", linewidth = 0.18) +
    facet_wrap(~ OutcomeLabel, nrow = 1, scales = "free_y") +
    scale_x_discrete(drop = FALSE) +
    scale_y_discrete(labels = setNames(top_pairs$PairLabel, top_pairs$PairPlot)) +
    scale_fill_gradientn(
      colours = heat_palette,
      breaks = heat_breaks,
      labels = number_format(accuracy = 0.1),
      name = "Interaction share (%)"
    ) +
    labs(x = NULL, y = NULL) +
    theme_interaction(base_size = ifelse(top_n <= 10, 8.2, 7.6)) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 7.5),
      axis.text.y = element_text(size = ifelse(top_n <= 10, 7.0, 6.0), lineheight = 0.88),
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.spacing.x = grid::unit(1.05, "lines"),
      legend.position = "bottom",
      legend.key.height = grid::unit(0.24, "cm"),
      legend.key.width = grid::unit(4.2, "cm"),
      plot.margin = margin(5, 5, 5, 5)
    ) +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5))

  p_bubble <- ggplot(
    top_pairs,
    aes(
      x = MeanInteractionSharePct,
      y = PairPlot,
      colour = PairType,
      size = ModelSupport
    )
  ) +
    geom_point(alpha = 0.88, stroke = 0.8) +
    facet_wrap(~ OutcomeLabel, nrow = 1, scales = "free_y") +
    scale_y_discrete(labels = setNames(top_pairs$PairLabel, top_pairs$PairPlot)) +
    scale_colour_manual(values = pair_type_palette, name = NULL, drop = TRUE) +
    scale_size_area(max_size = 5.2, breaks = c(2, 4, 6, 7), limits = c(0, 7), name = support_label) +
    scale_x_continuous(
      labels = number_format(accuracy = 0.1),
      expand = expansion(mult = c(0.14, 0.10))
    ) +
    labs(x = "Mean interaction share across seven models (%)", y = NULL) +
    theme_interaction(base_size = 7.9) +
    theme(
      axis.text.y = element_text(size = ifelse(top_n <= 10, 7.0, 6.3), lineheight = 0.88),
      axis.title.x = element_text(size = 8.8),
      panel.spacing.x = grid::unit(1.35, "lines"),
      legend.box = "horizontal",
      legend.box.just = "center",
      legend.text = element_text(size = 7.2),
      legend.title = element_text(size = 7.2),
      plot.margin = margin(5, 8, 5, 5)
    ) +
    guides(
      colour = guide_legend(order = 1, override.aes = list(size = 3.2)),
      size = guide_legend(order = 2, title.position = "left", title.hjust = 0.5)
    )

  top_model <- details %>%
    semi_join(top_pairs %>% select(Outcome, PairKey), by = c("Outcome", "PairKey")) %>%
    left_join(top_pairs %>% select(Outcome, PairKey, PairPlot, PairLabel, ConsensusRank), by = c("Outcome", "PairKey")) %>%
    mutate(PairPlot = factor(PairPlot, levels = levels(top_pairs$PairPlot)))

  p_model <- ggplot(
    top_model,
    aes(x = Model, y = PairPlot, colour = Model, shape = Model)
  ) +
    geom_point(aes(alpha = .data[[top_col]]), size = 2.0, stroke = 0.8) +
    facet_wrap(~ OutcomeLabel, nrow = 1, scales = "free_y") +
    scale_y_discrete(labels = setNames(top_pairs$PairLabel, top_pairs$PairPlot)) +
    scale_colour_manual(values = model_colors, breaks = model_order, drop = FALSE, name = NULL) +
    scale_shape_manual(
      values = c("XGBoost" = 17, "MLP" = 15, "TabNet" = 3, "NODE" = 0, "FT-Transformer" = 8, "SAINT" = 4, "TabPFN" = 19),
      breaks = model_order,
      drop = FALSE,
      name = NULL
    ) +
    scale_alpha_manual(values = c(`FALSE` = 0.18, `TRUE` = 0.95), guide = "none") +
    labs(x = NULL, y = NULL) +
    theme_interaction(base_size = 7.9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 7.3),
      axis.text.y = element_text(size = ifelse(top_n <= 10, 6.8, 6.1), lineheight = 0.88),
      panel.grid.major.x = element_line(linewidth = 0.18, colour = "grey90"),
      panel.grid.major.y = element_blank(),
      panel.spacing.x = grid::unit(1.35, "lines"),
      legend.position = "bottom",
      legend.text = element_text(size = 7.4)
    ) +
    guides(
      colour = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 2.6, alpha = 1)),
      shape = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(size = 2.6, alpha = 1))
    )

  suffix <- paste0("_top", top_n)
  heat_height <- ifelse(top_n <= 10, 4.5, 6.2)
  panel_height <- ifelse(top_n <= 10, 4.1, 5.6)
  save_all_formats(p_heat, paste0("Figure_candidate_interaction_heatmap", suffix), width = 10.8, height = heat_height)
  save_all_formats(p_bubble, paste0("Figure_candidate_interaction_bubble", suffix), width = 10.8, height = panel_height)
  save_all_formats(p_model, paste0("Figure_candidate_interaction_model_support", suffix), width = 10.8, height = panel_height)
}

build_top_figures(10)
build_top_figures(20)

cat("Wrote R interaction figures to:", output_dir, "\n")
