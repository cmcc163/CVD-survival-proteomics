rm(list = ls())
options(stringsAsFactors = FALSE)

required_packages <- c("readxl", "dplyr", "ggplot2", "ggrepel", "scales", "stringr", "ragg")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) stop("Install required packages: ", paste(missing_packages, collapse = ", "))

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(scales)
  library(stringr)
  library(ragg)
})

script_arg <- commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))]
default_output_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/"))
} else {
  getwd()
}
output_dir <- Sys.getenv("COX_INTERACTION_OUTPUT_DIR", unset = default_output_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_file <- file.path(output_dir, "Cox_interaction_top20_results.xlsx")
if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

outcome_levels <- c("Total_CVD", "ASCVD", "HF")
outcome_labels <- c(Total_CVD = "Total CVD", ASCVD = "ASCVD", HF = "HF")

pair_type_palette <- c(
  "Clinical-clinical" = "#4D4D4D",
  "Clinical-protein" = "#B55246",
  "Protein-protein" = "#0F4D92"
)

en_dash <- intToUtf8(0x2013)
multiply_sign <- intToUtf8(0x00D7)
pair_type_legend_labels <- c(
  "Clinical-clinical" = paste0("Clinical", en_dash, "clinical"),
  "Clinical-protein" = paste0("Clinical", en_dash, "proteomic"),
  "Protein-protein" = paste0("Proteomic", en_dash, "proteomic")
)

# Standardize interaction labels without changing the fitted interaction terms:
# - clinical-protein: protein first, clinical variable second;
# - protein-protein: alphabetical order;
# - clinical-clinical: alphabetical order, except Age is always last.
clinical_variables <- c(
  "Age", "Sex", "SBP", "BMI",
  "Antihypertensive medication", "Lipid-lowering medication"
)

canonical_pair_label <- function(pair, pair_type) {
  parts <- stringr::str_split(pair, "\\s+[xX\u00D7]\\s+", simplify = FALSE)[[1]]
  parts <- stringr::str_trim(parts)

  if (length(parts) != 2) {
    return(stringr::str_replace_all(pair, "\\s+[xX]\\s+", paste0(" ", multiply_sign, " ")))
  }

  if (pair_type == "Clinical-protein") {
    is_clinical <- parts %in% clinical_variables
    if (sum(is_clinical) == 1) {
      parts <- c(parts[!is_clinical], parts[is_clinical])
    } else {
      parts <- parts[order(tolower(parts))]
    }
  } else if (pair_type == "Clinical-clinical") {
    sort_key <- tolower(parts)
    sort_key[parts == "Age"] <- "zzzzzz_age"
    parts <- parts[order(sort_key)]
  } else if (pair_type == "Protein-protein") {
    parts <- parts[order(tolower(parts))]
  }

  paste(parts, collapse = paste0(" ", multiply_sign, " "))
}

wrap_pair <- function(x, width = 31) {
  stringr::str_wrap(x, width = width)
}

theme_cox_interaction <- function(base_size = 7.8) {
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
      legend.title = element_text(face = "bold", size = base_size),
      legend.text = element_text(size = base_size),
      legend.key.width = grid::unit(0.56, "cm"),
      legend.key.height = grid::unit(0.32, "cm"),
      panel.grid.major = element_line(linewidth = 0.18, colour = "grey90"),
      panel.grid.minor = element_blank(),
      panel.spacing.x = grid::unit(1.25, "lines"),
      plot.margin = margin(5, 6, 5, 6)
    )
}

save_png_pdf <- function(plot, name, width, height) {
  png_file <- file.path(output_dir, paste0(name, ".png"))
  ragg::agg_png(png_file, width = width, height = height, units = "in", res = 600, background = "white")
  print(plot)
  dev.off()

  ggplot2::ggsave(
    filename = file.path(output_dir, paste0(name, ".pdf")),
    plot = plot,
    width = width,
    height = height,
    units = "in",
    device = grDevices::cairo_pdf,
    bg = "white"
  )
}

results <- readxl::read_excel(input_file, sheet = "interaction_term_results") %>%
  mutate(
    Outcome = factor(Outcome, levels = outcome_levels),
    OutcomeLabel = factor(outcome_labels[as.character(Outcome)], levels = outcome_labels[outcome_levels]),
    PairType = factor(PairType, levels = names(pair_type_palette)),
    FDRStatus = if_else(PValue_BH_FDR_within_outcome_top20 <= 0.05, "FDR <= 0.05", "FDR >= 0.05"),
    FDRSignificant = PValue_BH_FDR_within_outcome_top20 <= 0.05,
    PairCanonical = mapply(
      canonical_pair_label,
      Pair,
      as.character(PairType),
      USE.NAMES = FALSE
    ),
    PairPlot = paste0(as.character(Outcome), "__", sprintf("%02d", ConsensusRank), "__", Pair),
    PairLabel = wrap_pair(PairCanonical, 32),
    NegLog10FDR = -log10(pmax(PValue_BH_FDR_within_outcome_top20, .Machine$double.xmin)),
    NegLog10P = -log10(pmax(PValue, .Machine$double.xmin)),
    LabelPriority = case_when(
      stringr::str_detect(Pair, "MMP12") ~ TRUE,
      PValue_BH_FDR_within_outcome_top20 < 0.001 ~ TRUE,
      ConsensusRank <= 3 & PValue < 0.05 ~ TRUE,
      TRUE ~ FALSE
    ),
    PlotLabel = if_else(LabelPriority, PairCanonical, NA_character_)
  ) %>%
  arrange(Outcome, desc(ConsensusRank)) %>%
  mutate(PairPlot = factor(PairPlot, levels = unique(PairPlot)))

forest_data <- results %>%
  mutate(
    HRLowerPlot = pmax(HRLower95, 0.65),
    HRUpperPlot = pmin(HRUpper95, 1.20)
  )

p_forest <- ggplot(forest_data, aes(y = PairPlot, x = HR)) +
  geom_vline(xintercept = 1, linetype = "dashed", linewidth = 0.35, colour = "grey35") +
  geom_errorbar(
    aes(xmin = HRLowerPlot, xmax = HRUpperPlot, colour = PairType),
    orientation = "y",
    width = 0,
    linewidth = 0.55
  ) +
  geom_point(
    aes(colour = PairType),
    shape = 16,
    size = 2.25,
    stroke = 0
  ) +
  geom_text(
    data = forest_data %>% filter(FDRSignificant),
    aes(x = HR, label = "*"),
    colour = "black",
    family = "Arial",
    fontface = "bold",
    size = 3.0,
    hjust = 1,
    vjust = 0.5,
    nudge_x = -0.007,
    nudge_y = 0.17,
    show.legend = FALSE
  ) +
  facet_wrap(~ OutcomeLabel, nrow = 1, scales = "free_y") +
  scale_y_discrete(labels = setNames(results$PairLabel, results$PairPlot)) +
  scale_x_continuous(
    limits = c(0.65, 1.20),
    breaks = c(0.70, 0.80, 0.90, 1.00, 1.10, 1.20),
    labels = number_format(accuracy = 0.01),
    expand = expansion(mult = c(0.01, 0.03))
  ) +
  scale_colour_manual(
    values = pair_type_palette,
    breaks = names(pair_type_palette),
    labels = pair_type_legend_labels,
    name = NULL,
    drop = TRUE
  ) +
  labs(
    x = "Cox multiplicative interaction HR (95% CI)",
    y = NULL
  ) +
  theme_cox_interaction(base_size = 7.7) +
  theme(
    axis.text.y = element_text(size = 6.1, lineheight = 0.88),
    axis.title.x = element_text(size = 8.8),
    strip.background = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.box = "horizontal"
  ) +
  guides(
    colour = guide_legend(order = 1, override.aes = list(size = 2.6, alpha = 1))
  )

p_priority <- ggplot(
  results,
  aes(
    x = MeanInteractionSharePct,
    y = NegLog10FDR,
    colour = PairType,
    size = Top20ModelSupport
  )
) +
  geom_hline(
    yintercept = -log10(0.05),
    linetype = "dashed",
    linewidth = 0.35,
    colour = "grey35"
  ) +
  geom_point(alpha = 0.86, stroke = 0.75) +
  ggrepel::geom_text_repel(
    data = results %>% filter(!is.na(PlotLabel)),
    aes(label = PlotLabel),
    size = 2.2,
    family = "Arial",
    min.segment.length = 0,
    box.padding = 0.26,
    point.padding = 0.18,
    max.overlaps = Inf,
    segment.size = 0.22,
    seed = 20260702,
    show.legend = FALSE
  ) +
  facet_wrap(~ OutcomeLabel, nrow = 1) +
  scale_colour_manual(values = pair_type_palette, name = NULL, drop = TRUE) +
  scale_size_area(
    max_size = 4.5,
    breaks = c(4, 6, 7),
    limits = c(0, 7),
    name = "Top 20 model support"
  ) +
  scale_x_continuous(
    labels = number_format(accuracy = 0.1),
    expand = expansion(mult = c(0.04, 0.14))
  ) +
  scale_y_continuous(
    breaks = pretty_breaks(n = 5),
    expand = expansion(mult = c(0.02, 0.12))
  ) +
  labs(
    x = "Mean TreeSHAP interaction share across seven models (%)",
    y = "-log10(Cox interaction FDR)"
  ) +
  theme_cox_interaction(base_size = 7.8) +
  theme(
    axis.title = element_text(size = 8.7),
    legend.box = "horizontal"
  ) +
  guides(
    colour = guide_legend(order = 1, override.aes = list(size = 2.8)),
    size = guide_legend(order = 2, title.position = "top", title.hjust = 0.5)
  )

summary_table <- results %>%
  group_by(Outcome, OutcomeLabel) %>%
  summarise(
    Top20_interactions = n(),
    Cox_P_lt_0_05 = sum(PValue < 0.05, na.rm = TRUE),
    Cox_FDR_lt_0_05 = sum(PValue_BH_FDR_within_outcome_top20 < 0.05, na.rm = TRUE),
    Strongest_Cox_FDR_pair = Pair[which.min(PValue_BH_FDR_within_outcome_top20)],
    Min_Cox_FDR = min(PValue_BH_FDR_within_outcome_top20, na.rm = TRUE),
    .groups = "drop"
  )

openxlsx_available <- requireNamespace("openxlsx", quietly = TRUE)
if (openxlsx_available) {
  openxlsx::write.xlsx(
    list(plot_summary = summary_table),
    file = file.path(output_dir, "Cox_interaction_top20_plot_summary.xlsx"),
    overwrite = TRUE
  )
} else {
  write.csv(summary_table, file.path(output_dir, "Cox_interaction_top20_plot_summary.csv"), row.names = FALSE)
}

save_png_pdf(p_forest, "Figure_cox_interaction_top20_forest", width = 10.8, height = 6.1)
save_png_pdf(p_priority, "Figure_cox_interaction_priority_scatter", width = 10.8, height = 4.4)

plot_note <- c(
  "Cox interaction Top20 plot notes",
  "",
  "1. Figure_cox_interaction_top20_forest shows the Cox multiplicative interaction HR and 95% CI for each Top20 TreeSHAP consensus interaction.",
  "2. Point and confidence-interval colours indicate pair type; an asterisk marks within-outcome BH-FDR <= 0.05.",
  "3. Figure_cox_interaction_priority_scatter maps TreeSHAP consensus interaction strength against Cox interaction FDR.",
  "4. MMP12-related pairs and selected strong Cox-FDR pairs are directly labelled.",
  "5. Only PNG and PDF outputs are generated."
)
writeLines(plot_note, file.path(output_dir, "cox_interaction_top20_plot_notes.txt"), useBytes = TRUE)

cat("Wrote Cox interaction plots to:", output_dir, "\n")
