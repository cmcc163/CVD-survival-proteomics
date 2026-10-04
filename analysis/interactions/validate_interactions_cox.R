rm(list = ls())
options(stringsAsFactors = FALSE)

required_packages <- c("data.table", "readxl", "openxlsx", "dplyr", "survival", "stringr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) stop("Install required packages: ", paste(missing_packages, collapse = ", "))

suppressPackageStartupMessages({
  library(data.table)
  library(readxl)
  library(openxlsx)
  library(dplyr)
  library(survival)
  library(stringr)
})

script_arg <- commandArgs(FALSE)[grepl("^--file=", commandArgs(FALSE))]
output_dir <- if (length(script_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/"))
} else {
  getwd()
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

section_dir <- normalizePath(file.path(output_dir, ".."), winslash = "/", mustWork = TRUE)
interaction_xlsx <- file.path(section_dir, "Candidate_predictor_interactions_all_lassonet.xlsx")
data_root <- Sys.getenv("CVD_DATA_ROOT", unset = "data")

outcomes <- c("Total_CVD", "ASCVD", "HF")
time_col <- "time"
event_col <- "Is_Incident"

clinical_vars <- c(
  "age", "sex", "ever_smoked", "Diabetes_baseline",
  "Cholesterol_treatment", "hdl_cholesterol", "non_hdl_cholesterol",
  "hypertension_treatment", "average_SBP", "eGFR", "BMI"
)

cat_vars <- c(
  "sex", "ever_smoked", "Diabetes_baseline",
  "hypertension_treatment", "Cholesterol_treatment"
)

continuous_clinical_vars <- setdiff(clinical_vars, cat_vars)

safe_name <- function(x) {
  out <- gsub("[^0-9A-Za-z_]+", "_", as.character(x))
  out <- gsub("_+", "_", out)
  out <- gsub("^_|_$", "", out)
  out <- ifelse(grepl("^[0-9]", out), paste0("v_", out), out)
  out
}

label_encode <- function(x) {
  as.integer(factor(ifelse(is.na(x), "Missing", as.character(x)))) - 1L
}

read_top20_interactions <- function() {
  readxl::read_excel(interaction_xlsx, sheet = "consensus_interactions") %>%
    arrange(Outcome, ConsensusRank) %>%
    group_by(Outcome) %>%
    slice_head(n = 20) %>%
    ungroup() %>%
    mutate(TopN = 20L)
}

required_columns_for_outcome <- function(top20_outcome) {
  features <- unique(c(top20_outcome$Feature_A, top20_outcome$Feature_B))
  cols <- unique(c(
    "eid", "Ethnic", "baseline_date", "Event_Date", event_col,
    "total_cholesterol", "hdl_cholesterol",
    clinical_vars,
    features
  ))
  setdiff(cols, "non_hdl_cholesterol")
}

parse_date_flexible <- function(x) {
  as.Date(x, tryFormats = c("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"))
}

load_outcome_data <- function(outcome, top20_outcome) {
  path <- file.path(data_root, outcome, "followup_incident_20260122.csv")
  if (!file.exists(path)) {
    stop("Input data not found: ", path)
  }

  header <- names(data.table::fread(path, nrows = 0, showProgress = FALSE))
  select_cols <- intersect(required_columns_for_outcome(top20_outcome), header)
  data <- data.table::fread(path, select = select_cols, showProgress = FALSE, data.table = FALSE)

  data$total_cholesterol <- suppressWarnings(as.numeric(data$total_cholesterol))
  data$hdl_cholesterol <- suppressWarnings(as.numeric(data$hdl_cholesterol))
  data$non_hdl_cholesterol <- data$total_cholesterol - data$hdl_cholesterol

  baseline_date <- parse_date_flexible(data$baseline_date)
  event_date <- parse_date_flexible(data$Event_Date)
  data[[time_col]] <- as.numeric(event_date - baseline_date)
  data[[event_col]] <- suppressWarnings(as.numeric(data[[event_col]]))

  raw_n <- nrow(data)
  data <- data[!is.na(data[[time_col]]) & !is.na(data[[event_col]]) & data[[time_col]] > 0, , drop = FALSE]

  for (var in intersect(cat_vars, names(data))) {
    data[[var]] <- label_encode(data[[var]])
  }

  list(
    data = data,
    summary = data.frame(
      Outcome = outcome,
      DataPath = path,
      RawRows = raw_n,
      RowsAfterValidTimeEvent = nrow(data),
      DroppedInvalidTimeEvent = raw_n - nrow(data),
      stringsAsFactors = FALSE
    )
  )
}

model_main_variables <- function(row) {
  fa <- row$Feature_A
  fb <- row$Feature_B
  pair_type <- row$PairType

  proteins <- character(0)
  if (pair_type == "Clinical-protein") {
    proteins <- setdiff(c(fa, fb), clinical_vars)
  } else if (pair_type == "Protein-protein") {
    proteins <- c(fa, fb)
  }

  unique(c(clinical_vars, proteins))
}

standardize_model_data <- function(df, vars) {
  stats <- list()
  for (var in vars) {
    if (!var %in% names(df) || var %in% cat_vars) next
    x <- suppressWarnings(as.numeric(df[[var]]))
    mu <- mean(x, na.rm = TRUE)
    sdv <- stats::sd(x, na.rm = TRUE)
    if (is.finite(sdv) && sdv > 0) {
      df[[var]] <- (x - mu) / sdv
    } else {
      df[[var]] <- x - mu
    }
    stats[[var]] <- data.frame(Variable = var, Mean = mu, SD = sdv)
  }
  attr(df, "scale_stats") <- if (length(stats) > 0) bind_rows(stats) else data.frame()
  df
}

fit_interaction_cox <- function(outcome, data, row) {
  main_vars <- model_main_variables(row)
  fa <- row$Feature_A
  fb <- row$Feature_B
  interaction_col <- paste0("INT__", safe_name(fa), "__x__", safe_name(fb))

  needed <- unique(c(time_col, event_col, main_vars, fa, fb))
  missing_vars <- setdiff(needed, names(data))
  if (length(missing_vars) > 0) {
    return(list(
      interaction = data.frame(),
      coefficients = data.frame(),
      summary = data.frame(
        Outcome = outcome,
        ConsensusRank = row$ConsensusRank,
        Pair = row$Pair,
        PairType = row$PairType,
        Feature_A = fa,
        Feature_B = fb,
        Status = "missing_variables",
        Message = paste(missing_vars, collapse = "; "),
        stringsAsFactors = FALSE
      )
    ))
  }

  model_df <- data[, needed, drop = FALSE]
  numeric_vars <- setdiff(unique(c(main_vars, fa, fb)), cat_vars)
  for (var in numeric_vars) {
    model_df[[var]] <- suppressWarnings(as.numeric(model_df[[var]]))
  }

  model_df <- standardize_model_data(model_df, numeric_vars)
  model_df[[interaction_col]] <- model_df[[fa]] * model_df[[fb]]

  covariates_raw <- unique(c(main_vars, interaction_col))
  covariate_names <- safe_name(covariates_raw)
  names_map <- setNames(covariate_names, covariates_raw)
  colnames(model_df)[match(covariates_raw, colnames(model_df))] <- covariate_names

  model_df <- model_df[, c(time_col, event_col, covariate_names), drop = FALSE]
  before_complete <- nrow(model_df)
  model_df <- model_df[stats::complete.cases(model_df), , drop = FALSE]

  event_n <- sum(model_df[[event_col]] == 1, na.rm = TRUE)
  if (nrow(model_df) < 100 || event_n < 10) {
    return(list(
      interaction = data.frame(),
      coefficients = data.frame(),
      summary = data.frame(
        Outcome = outcome,
        ConsensusRank = row$ConsensusRank,
        Pair = row$Pair,
        PairType = row$PairType,
        Feature_A = fa,
        Feature_B = fb,
        InteractionTerm = interaction_col,
        Status = "insufficient_complete_data_or_events",
        Message = paste0("complete_n=", nrow(model_df), "; events=", event_n),
        NBeforeCompleteCase = before_complete,
        NComplete = nrow(model_df),
        Events = event_n,
        stringsAsFactors = FALSE
      )
    ))
  }

  formula_text <- paste0(
    "survival::Surv(", time_col, ", ", event_col, ") ~ ",
    paste(covariate_names, collapse = " + ")
  )
  fit_status <- "ok"
  fit_message <- ""

  fit <- tryCatch(
    survival::coxph(
      as.formula(formula_text),
      data = model_df,
      ties = "efron",
      robust = TRUE,
      singular.ok = TRUE,
      x = FALSE,
      y = FALSE,
      model = FALSE
    ),
    warning = function(w) {
      fit_status <<- "ok_with_warning"
      fit_message <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    },
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    return(list(
      interaction = data.frame(),
      coefficients = data.frame(),
      summary = data.frame(
        Outcome = outcome,
        ConsensusRank = row$ConsensusRank,
        Pair = row$Pair,
        PairType = row$PairType,
        Feature_A = fa,
        Feature_B = fb,
        InteractionTerm = interaction_col,
        Status = "error",
        Message = conditionMessage(fit),
        NBeforeCompleteCase = before_complete,
        NComplete = nrow(model_df),
        Events = event_n,
        stringsAsFactors = FALSE
      )
    ))
  }

  fit_sum <- summary(fit)
  coef_df <- as.data.frame(fit_sum$coefficients)
  ci_df <- as.data.frame(fit_sum$conf.int)
  coef_df$TermSafe <- rownames(coef_df)
  ci_df$TermSafe <- rownames(ci_df)

  out <- left_join(coef_df, ci_df, by = "TermSafe", suffix = c("", "_ci"))
  out$Term <- names(names_map)[match(out$TermSafe, names_map)]
  out$Term <- ifelse(is.na(out$Term), out$TermSafe, out$Term)

  p_col <- grep("^Pr\\(", names(out), value = TRUE)[1]
  z_col <- grep("^z$", names(out), value = TRUE)[1]
  robust_se_col <- grep("robust se", names(out), value = TRUE)[1]
  se_col <- if (!is.na(robust_se_col)) robust_se_col else "se(coef)"

  coef_results <- data.frame(
    Outcome = outcome,
    ConsensusRank = row$ConsensusRank,
    Pair = row$Pair,
    PairType = row$PairType,
    Feature_A = fa,
    Feature_B = fb,
    Term = out$Term,
    TermSafe = out$TermSafe,
    InteractionTerm = interaction_col,
    Beta = out$coef,
    SE = out[[se_col]],
    Z = if (!is.na(z_col)) out[[z_col]] else NA_real_,
    HR = out$`exp(coef)`,
    HRLower95 = out$`lower .95`,
    HRUpper95 = out$`upper .95`,
    PValue = out[[p_col]],
    NComplete = nrow(model_df),
    Events = event_n,
    FitStatus = fit_status,
    FitMessage = fit_message,
    stringsAsFactors = FALSE
  )

  interaction_result <- coef_results %>%
    filter(TermSafe == safe_name(interaction_col)) %>%
    mutate(
      MeanInteractionSharePct = row$MeanInteractionSharePct,
      MedianInteractionSharePct = row$MedianInteractionSharePct,
      Top20ModelSupport = row$Top20ModelSupport,
      Top50ModelSupport = row$Top50ModelSupport,
      ModelsAvailable = row$ModelsAvailable
    ) %>%
    select(
      Outcome, ConsensusRank, Pair, PairType, Feature_A, Feature_B,
      InteractionTerm, MeanInteractionSharePct, MedianInteractionSharePct,
      Top20ModelSupport, Top50ModelSupport, ModelsAvailable,
      NComplete, Events, Beta, SE, Z, HR, HRLower95, HRUpper95, PValue,
      FitStatus, FitMessage
    )

  model_summary <- data.frame(
    Outcome = outcome,
    ConsensusRank = row$ConsensusRank,
    Pair = row$Pair,
    PairType = row$PairType,
    Feature_A = fa,
    Feature_B = fb,
    InteractionTerm = interaction_col,
    Status = fit_status,
    Message = fit_message,
    NBeforeCompleteCase = before_complete,
    NComplete = nrow(model_df),
    Events = event_n,
    CovariateCount = length(covariate_names),
    MeanInteractionSharePct = row$MeanInteractionSharePct,
    Top20ModelSupport = row$Top20ModelSupport,
    Top50ModelSupport = row$Top50ModelSupport,
    stringsAsFactors = FALSE
  )

  list(
    interaction = interaction_result,
    coefficients = coef_results,
    summary = model_summary
  )
}

write_methods_note <- function() {
  note <- c(
    "# Top20 interaction Cox validation",
    "",
    "## Consensus rank",
    "",
    "ConsensusRank is computed within each outcome by ranking all feature pairs in descending order of MeanInteractionSharePct.",
    "MeanInteractionSharePct is the average, across the seven all_lassonet model classes, of the within-model share of total off-diagonal TreeSHAP interaction strength for that feature pair.",
    "Therefore, Top20 means the 20 pairs with the largest cross-model mean normalized interaction share within each outcome.",
    "",
    "## Cox model specification",
    "",
    "For each outcome, the Top20 consensus interactions were tested one at a time.",
    "",
    "- Clinical-protein pair: 11 clinical variables + the protein main effect + clinical:protein interaction.",
    "- Clinical-clinical pair: 11 clinical variables + clinical:clinical interaction.",
    "- Protein-protein pair: 11 clinical variables + two protein main effects + protein:protein interaction.",
    "",
    "The 11 clinical variables are age, sex, ever_smoked, Diabetes_baseline, Cholesterol_treatment, hdl_cholesterol, non_hdl_cholesterol, hypertension_treatment, average_SBP, eGFR, and BMI.",
    "Continuous clinical variables and proteins are standardized before constructing interaction terms. Categorical variables are label-encoded to follow the training preprocessing logic.",
    "Each interaction is fitted as a separate Cox model with complete-case data for the variables required by that model."
  )
  writeLines(note, file.path(output_dir, "cox_interaction_top20_methods.md"), useBytes = TRUE)
}

top20 <- read_top20_interactions()

all_interactions <- list()
all_coefficients <- list()
all_model_summary <- list()
all_preprocessing <- list()

for (outcome in outcomes) {
  cat("===", outcome, "===\n")
  top20_outcome <- top20 %>% filter(Outcome == outcome)
  loaded <- load_outcome_data(outcome, top20_outcome)
  data <- loaded$data
  all_preprocessing[[outcome]] <- loaded$summary

  for (i in seq_len(nrow(top20_outcome))) {
    row <- top20_outcome[i, ]
    cat(sprintf("  Rank %02d: %s\n", as.integer(row$ConsensusRank), row$Pair))
    result <- fit_interaction_cox(outcome, data, row)
    all_interactions[[length(all_interactions) + 1]] <- result$interaction
    all_coefficients[[length(all_coefficients) + 1]] <- result$coefficients
    all_model_summary[[length(all_model_summary) + 1]] <- result$summary
  }
}

interaction_results <- bind_rows(all_interactions)
coefficient_results <- bind_rows(all_coefficients)
model_fit_summary <- bind_rows(all_model_summary)
preprocessing_summary <- bind_rows(all_preprocessing)

if (nrow(interaction_results) > 0 && "PValue" %in% names(interaction_results)) {
  interaction_results <- interaction_results %>%
    group_by(Outcome) %>%
    mutate(PValue_BH_FDR_within_outcome_top20 = p.adjust(PValue, method = "BH")) %>%
    ungroup()
}

output_xlsx <- file.path(output_dir, "Cox_interaction_top20_results.xlsx")
openxlsx::write.xlsx(
  list(
    top20_consensus_interactions = top20,
    interaction_term_results = interaction_results,
    all_model_coefficients = coefficient_results,
    model_fit_summary = model_fit_summary,
    preprocessing_summary = preprocessing_summary
  ),
  file = output_xlsx,
  overwrite = TRUE
)

write_methods_note()

cat("Wrote:", output_xlsx, "\n")
