rm(list = ls())

suppressPackageStartupMessages({
  library(readxl)
  library(openxlsx)
  library(dplyr)
  library(tidyr)
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
output_dir <- Sys.getenv("SHAP_MAPLE_FIGURE_OUTPUT_DIR", unset = default_output_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

shap_root <- Sys.getenv("CVD_SHAP_ROOT", unset = "artifacts/shap/final")
ascvd_combined_file <- file.path(shap_root, "ASCVD", "Combined", "Selection_Results_ASCVD_Combined.xlsx")
hf_combined_file <- file.path(shap_root, "HF", "Combined", "Selection_Results_HF_Combined.xlsx")
maple_source <- Sys.getenv("CVD_MAPLE_SOURCE", unset = "artifacts/genetics/maple_source_data.xlsx")
coloc_source <- Sys.getenv("CVD_COLOC_FILE", unset = "artifacts/genetics/coloc_susie_results.xlsx")

clinical_vars <- c(
  "age", "BMI", "hypertension_treatment", "Cholesterol_treatment", "sex",
  "average_SBP", "eGFR", "hdl_cholesterol", "smoking status",
  "Diabetes_baseline", "non_hdl_cholesterol"
)

outcome_levels <- c("CAD", "Stroke", "HF")

read_ranked_shap_proteins <- function(path, shap_group) {
  read_excel(path, sheet = "1_Ensemble_SHAP_Rank") %>%
    transmute(
      shap_group = shap_group,
      protein = as.character(Feature),
      combined_model_rank = as.integer(Rank),
      shap_pct = as.numeric(Ensemble_Mean_SHAP_pct),
      cumulative_shap_pct = as.numeric(Cumulative_SHAP_pct),
      final_80pct_lrt = Final_80pct_LRT,
      final_kneedle_lrt = Final_Kneedle_LRT
    ) %>%
    filter(!protein %in% clinical_vars) %>%
    arrange(desc(shap_pct), combined_model_rank) %>%
    mutate(
      protein_shap_rank = row_number(),
      selected_lrt = final_80pct_lrt == "Yes" | final_kneedle_lrt == "Yes"
    )
}

ascvd_ranked <- read_ranked_shap_proteins(ascvd_combined_file, "ASCVD combined model, ranked proteins")
hf_ranked <- read_ranked_shap_proteins(hf_combined_file, "HF combined model, ranked proteins")

maple <- read.xlsx(maple_source, sheet = "all_results_normalized") %>%
  filter(analysis == "MAPLE", outcome_group %in% outcome_levels) %>%
  transmute(
    outcome = outcome_group,
    protein = as.character(exposure),
    maple_method = method,
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
    maple_fdr_scientific = formatC(maple_fdr, format = "e", digits = 6),
    maple_sig = maple_fdr <= 0.05,
    maple_direction = case_when(
      maple_sig & maple_effect > 0 ~ "Positive MAPLE effect",
      maple_sig & maple_effect < 0 ~ "Negative MAPLE effect",
      TRUE ~ "FDR >= 0.05"
    )
  ) %>%
  ungroup()

coloc_primary_sheet <- c(CAD = 5L, Stroke = 6L, HF = 7L)

coloc_primary <- bind_rows(lapply(outcome_levels, function(current_outcome) {
  read.xlsx(coloc_source, sheet = coloc_primary_sheet[[current_outcome]]) %>%
    transmute(
      outcome = current_outcome,
      protein = as.character(protein),
      coloc_status = status,
      coloc_evidence_class = evidence_class,
      coloc_final_use = final_use,
      coloc_max_pp_h4 = as.numeric(max_PP.H4),
      coloc_max_pp_h3 = as.numeric(max_PP.H3),
      coloc_best_hit1 = as.character(best_hit1),
      coloc_best_hit2 = as.character(best_hit2),
      coloc_top_snp = as.character(top_snp_for_best_H4)
    ) %>%
    filter(!is.na(coloc_max_pp_h4), coloc_max_pp_h4 >= 0.8)
}))

comparison <- bind_rows(
  tidyr::crossing(outcome = c("CAD", "Stroke"), ascvd_ranked),
  hf_ranked %>% mutate(outcome = "HF")
) %>%
  left_join(maple, by = c("outcome", "protein")) %>%
  left_join(coloc_primary, by = c("outcome", "protein")) %>%
  mutate(
    outcome = factor(outcome, levels = outcome_levels),
    has_maple_result = !is.na(maple_fdr),
    maple_sig = if_else(has_maple_result & !is.na(maple_fdr) & maple_fdr <= 0.05, TRUE, FALSE),
    strong_susie_coloc = !is.na(coloc_max_pp_h4),
    evidence_status = case_when(
      maple_sig ~ "Significant by MAPLE",
      TRUE ~ "Not significant"
    ),
    neg_log10_fdr = if_else(has_maple_result, -log10(maple_fdr), NA_real_),
    neg_log10_fdr_plot = neg_log10_fdr,
    label = if_else(evidence_status == "Significant by MAPLE", protein, NA_character_),
    coloc_label = if_else(strong_susie_coloc, protein, NA_character_)
  )

summary_table <- comparison %>%
  group_by(outcome) %>%
  summarise(
    n_ranked_proteins = n(),
    n_with_maple_result = sum(has_maple_result),
    n_without_maple_result = sum(!has_maple_result),
    n_maple_fdr_lt_0_05 = sum(maple_sig),
    n_strong_susie_coloc = sum(strong_susie_coloc),
    proteins_without_maple_result = paste(protein[!has_maple_result], collapse = ", "),
    proteins_with_maple_fdr_lt_0_05 = paste(protein[maple_sig], collapse = ", "),
    proteins_with_strong_susie_coloc = paste(protein[strong_susie_coloc], collapse = ", "),
    .groups = "drop"
  )

write.xlsx(
  list(
    ranked_protein_comparison = comparison,
    summary = summary_table,
    ascvd_combined_ranked_proteins = ascvd_ranked,
    hf_combined_ranked_proteins = hf_ranked,
    maple_results = maple,
    susie_coloc_primary_positive = coloc_primary,
    excluded_clinical_variables = data.frame(variable = clinical_vars)
  ),
  file.path(output_dir, "SHAP_MAPLE_ranked_protein_comparison_CAD_Stroke_HF.xlsx"),
  overwrite = TRUE
)

plot_data <- comparison %>% filter(has_maple_result)
y_max <- ceiling(max(plot_data$neg_log10_fdr_plot, na.rm = TRUE) / 10) * 10
y_breaks <- sort(unique(c(0, -log10(0.05), 2, 5, 10, 20, y_max)))
y_breaks <- y_breaks[y_breaks <= y_max]
y_labels <- ifelse(abs(y_breaks - -log10(0.05)) < 1e-8, "FDR=0.05", formatC(y_breaks, format = "fg", digits = 3))

status_palette <- c(
  "Not significant" = "#B9B9B9",
  "Significant by MAPLE" = "#B55246"
)

x_major_breaks <- c(1, 10, 20, 40, 60, 80)
x_minor_breaks <- setdiff(seq(1, 80, by = 1), x_major_breaks)

p <- ggplot(plot_data, aes(x = protein_shap_rank, y = neg_log10_fdr_plot)) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.35, colour = "grey35") +
  geom_point(
    aes(colour = evidence_status),
    size = 2.35,
    alpha = 0.9,
    stroke = 0.7
  ) +
  geom_point(
    data = plot_data %>% filter(strong_susie_coloc),
    aes(shape = "SuSiE colocalization"),
    size = 4.0,
    stroke = 0.85,
    colour = "black",
    fill = NA
  ) +
  ggrepel::geom_text_repel(
    data = plot_data %>% filter(!is.na(label), !strong_susie_coloc),
    aes(label = label),
    size = 2.05,
    family = "Arial",
    min.segment.length = 0,
    box.padding = 0.28,
    point.padding = 0.18,
    max.overlaps = Inf,
    segment.size = 0.22,
    seed = 20260628
  ) +
  ggrepel::geom_text_repel(
    data = plot_data %>% filter(strong_susie_coloc),
    aes(label = coloc_label),
    size = 2.25,
    family = "Arial",
    fontface = "bold",
    colour = "black",
    nudge_y = 0.35,
    min.segment.length = 0,
    box.padding = 0.32,
    point.padding = 0.25,
    max.overlaps = Inf,
    segment.size = 0.24,
    seed = 20260628
  ) +
  scale_x_continuous(
    breaks = x_major_breaks,
    minor_breaks = x_minor_breaks,
    guide = guide_axis(minor.ticks = TRUE),
    expand = expansion(add = c(4.2, 3.0))
  ) +
  scale_y_continuous(
    breaks = y_breaks,
    labels = y_labels,
    trans = scales::pseudo_log_trans(sigma = 0.4, base = 10),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  scale_colour_manual(values = status_palette, name = NULL) +
  scale_shape_manual(values = c("SuSiE colocalization" = 23), name = NULL) +
  facet_wrap(~ outcome, nrow = 1) +
  labs(
    x = "SHAP rank",
    y = "-log10(FDR)"
  ) +
  coord_cartesian(ylim = c(0, y_max), clip = "off") +
  theme_classic(base_size = 7.5, base_family = "Arial") +
  theme(
    axis.line = element_line(linewidth = 0.45, colour = "black"),
    axis.ticks = element_line(linewidth = 0.40, colour = "black"),
    axis.minor.ticks.x.bottom = element_line(linewidth = 0.32, colour = "black"),
    axis.ticks.length.x = unit(1.5, "mm"),
    axis.minor.ticks.length.x = rel(0.55),
    axis.text = element_text(colour = "black", size = 7.2),
    axis.text.x = element_text(margin = margin(t = 3.5)),
    axis.title = element_text(face = "bold", size = 8.5),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 9),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.text = element_text(size = 7.2),
    legend.title = element_text(face = "bold", size = 7.4),
    panel.grid.major = element_line(linewidth = 0.18, colour = "grey90"),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(1.0, "lines"),
    plot.margin = margin(6, 8, 6, 6)
  ) +
  guides(
    colour = guide_legend(order = 1, override.aes = list(size = 3)),
    shape = guide_legend(order = 2, override.aes = list(size = 3.4, colour = "black", fill = NA))
  )

png_file <- file.path(output_dir, "Figure_SHAP_MAPLE_ranked_protein_vs_FDR_CAD_Stroke_HF.png")
ragg::agg_png(png_file, width = 183 / 25.4, height = 92 / 25.4, units = "in", res = 600, background = "white")
print(p)
dev.off()

cat("Wrote:", png_file, "\n")
print(summary_table)
