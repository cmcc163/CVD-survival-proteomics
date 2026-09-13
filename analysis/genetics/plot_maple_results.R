rm(list = ls())

suppressPackageStartupMessages({
  library(openxlsx)
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(ragg)
  library(scales)
})

script_arg <- commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))]
default_output_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/"))
} else {
  getwd()
}
output_dir <- Sys.getenv("MAPLE_FIGURE_OUTPUT_DIR", unset = default_output_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

maple_source <- Sys.getenv("CVD_MAPLE_SOURCE", unset = "artifacts/genetics/maple_source_data.xlsx")

outcome_levels <- c("CAD", "Stroke", "HF")
fdr_threshold <- 0.05
x_axis_limits <- c(-2, 2)

theme_pub <- function(base_size = 7.5, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.45, colour = "black"),
      axis.ticks = element_line(linewidth = 0.40, colour = "black"),
      axis.text = element_text(colour = "black", size = 7.2),
      axis.title = element_text(face = "bold", size = 8.5),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 9),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.text = element_text(size = 7.2),
      legend.title = element_blank(),
      panel.grid.major = element_line(linewidth = 0.18, colour = "grey90"),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(1.0, "lines"),
      plot.margin = margin(6, 10, 6, 8)
    )
}

maple <- read.xlsx(maple_source, sheet = "all_results_normalized") %>%
  filter(analysis == "MAPLE", outcome_group %in% outcome_levels) %>%
  transmute(
    outcome = factor(outcome_group, levels = outcome_levels),
    protein = as.character(exposure),
    method = method,
    nsnp = as.integer(nsnp),
    maple_effect = as.numeric(effect),
    maple_se = as.numeric(se),
    maple_ci_low = as.numeric(ci_low),
    maple_ci_high = as.numeric(ci_high),
    maple_p_original = as.numeric(p_value),
    maple_fdr_original = as.numeric(fdr),
    maple_z = maple_effect / maple_se,
    maple_log10_p = log10(2) + pnorm(-abs(maple_z), log.p = TRUE) / log(10),
    maple_p = 10^maple_log10_p
  ) %>%
  group_by(outcome) %>%
  mutate(
    maple_fdr = p.adjust(maple_p, method = "BH"),
    maple_fdr_log10 = log10(maple_fdr),
    maple_p_scientific = formatC(maple_p, format = "e", digits = 6),
    maple_fdr_scientific = formatC(maple_fdr, format = "e", digits = 6)
  ) %>%
  ungroup() %>%
  mutate(
    neg_log10_fdr = -log10(maple_fdr),
    neg_log10_fdr_plot = neg_log10_fdr,
    maple_effect_out_of_range = maple_effect < x_axis_limits[1] |
      maple_effect > x_axis_limits[2],
    maple_effect_plot = pmax(
      x_axis_limits[1],
      pmin(x_axis_limits[2], maple_effect)
    ),
    maple_sig = !is.na(maple_fdr) & maple_fdr <= fdr_threshold,
    maple_direction = case_when(
      maple_sig & maple_effect > 0 ~ "Positive effect",
      maple_sig & maple_effect < 0 ~ "Negative effect",
      TRUE ~ "Not significant"
    )
  )

plot_data <- maple %>%
  mutate(
    label_priority = case_when(
      maple_sig ~ 3L,
      TRUE ~ 9L
    )
  )

label_data <- plot_data %>%
  filter(maple_sig) %>%
  group_by(outcome) %>%
  arrange(label_priority, maple_fdr, desc(abs(maple_effect)), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

summary_table <- plot_data %>%
  group_by(outcome) %>%
  summarise(
    n_maple_results = n(),
    n_maple_fdr_le_0_05 = sum(maple_sig),
    n_positive_maple_fdr_le_0_05 = sum(maple_direction == "Positive effect"),
    n_negative_maple_fdr_le_0_05 = sum(maple_direction == "Negative effect"),
    proteins_maple_fdr_le_0_05 = paste(protein[maple_sig], collapse = ", "),
    .groups = "drop"
  )

write.xlsx(
  list(
    volcano_source_data = plot_data,
    labeled_points = label_data,
    summary = summary_table
  ),
  file.path(output_dir, "MAPLE_volcano_source_data_CAD_Stroke_HF.xlsx"),
  overwrite = TRUE
)

direction_palette <- c(
  "Not significant" = "#B9B9B9",
  "Negative effect" = "#3A6EA5",
  "Positive effect" = "#B55246"
)

y_max <- ceiling(max(plot_data$neg_log10_fdr_plot, na.rm = TRUE) / 10) * 10
y_breaks <- sort(unique(c(0, -log10(fdr_threshold), 2, 5, 10, 20, y_max)))
y_breaks <- y_breaks[y_breaks <= y_max]
y_labels <- ifelse(abs(y_breaks - -log10(fdr_threshold)) < 1e-8, "FDR=0.05", formatC(y_breaks, format = "fg", digits = 3))
x_breaks <- seq(x_axis_limits[1], x_axis_limits[2], by = 1)

p <- ggplot(plot_data, aes(x = maple_effect_plot, y = neg_log10_fdr_plot)) +
  geom_hline(
    yintercept = -log10(fdr_threshold),
    linetype = "dashed",
    linewidth = 0.35,
    colour = "grey35"
  ) +
  geom_vline(xintercept = 0, linewidth = 0.30, colour = "grey55") +
  geom_point(
    data = filter(plot_data, !maple_effect_out_of_range),
    aes(colour = maple_direction),
    size = 2.25,
    alpha = 0.88,
    stroke = 0
  ) +
  geom_point(
    data = filter(plot_data, maple_effect_out_of_range),
    aes(colour = maple_direction),
    shape = 17,
    size = 2.25,
    alpha = 0.88,
    stroke = 0,
    show.legend = FALSE
  ) +
  ggrepel::geom_text_repel(
    data = label_data,
    aes(label = protein),
    size = 2.25,
    family = "Arial",
    colour = "black",
    min.segment.length = 0,
    box.padding = 0.30,
    point.padding = 0.22,
    max.overlaps = Inf,
    segment.size = 0.22,
    seed = 20260628
  ) +
  facet_wrap(~ outcome, nrow = 1) +
  scale_colour_manual(
    values = direction_palette,
    breaks = names(direction_palette),
    name = NULL
  ) +
  scale_y_continuous(
    breaks = y_breaks,
    labels = y_labels,
    trans = scales::pseudo_log_trans(sigma = 0.4, base = 10),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  scale_x_continuous(
    breaks = x_breaks,
    labels = formatC(x_breaks, format = "fg", digits = 2),
    expand = expansion(mult = c(0.025, 0.025))
  ) +
  labs(
    x = "Causal effect by MAPLE",
    y = "-log10(FDR)"
  ) +
  coord_cartesian(
    xlim = x_axis_limits,
    ylim = c(0, y_max),
    clip = "off"
  ) +
  theme_pub() +
  guides(
    colour = guide_legend(order = 1, override.aes = list(size = 3.0, alpha = 1))
  )

base_file <- file.path(output_dir, "Figure_MAPLE_volcano_CAD_Stroke_HF")

ragg::agg_png(paste0(base_file, ".png"), width = 183 / 25.4, height = 98 / 25.4, units = "in", res = 600, background = "white")
print(p)
dev.off()

cat("Wrote:", paste0(base_file, ".png"), "\n")
print(summary_table)
