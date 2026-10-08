# Shared configuration and SNP/LD checks for the manuscript genetics workflow.
safe_name <- function(x) gsub("[^A-Za-z0-9_.-]+", "_", x)
outcome_map <- c(CHD = "ieu-a-7", CAD = "ebi-a-GCST005195",
                 Stroke = "ebi-a-GCST006908", HF = "ebi-a-GCST009541")

read_options <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% args) {
    cat("Options: --data-root DIR --output-root DIR --outcome CHD|CAD|Stroke|HF\n",
        "--exposure FILE --outcome-file FILE --omega FILE --ld-scores DIR\n",
        "--plink FILE --bfile PREFIX --ld-root DIR --maple-root DIR\n",
        "--main-root DIR --sensitivity-root DIR --proteins A,B --max-snps N\n",
        "--gibbs N --seed N --resume true|false\n", sep = "")
    quit(status = 0)
  }
  allowed <- c("data-root", "output-root", "outcome", "exposure", "outcome-file",
               "omega", "ld-scores", "plink", "bfile", "ld-root", "maple-root",
               "main-root", "sensitivity-root", "proteins", "max-snps", "gibbs",
               "seed", "resume")
  if (length(args) %% 2L) stop("Supply options as --name value pairs.")
  opts <- list()
  for (i in seq_along(args)[seq_along(args) %% 2L == 1L]) {
    key <- sub("^--", "", args[i])
    if (!startsWith(args[i], "--") || !key %in% allowed || key %in% names(opts))
      stop("Unknown or duplicate option: ", args[i])
    opts[[key]] <- args[i + 1L]
  }
  value <- function(key, default) if (is.null(opts[[key]])) default else opts[[key]]
  data_root <- value("data-root", "data/genetics")
  output_root <- value("output-root", "results/genetics")
  outcome <- value("outcome", "Stroke")
  if (!outcome %in% names(outcome_map)) stop("Unknown outcome: ", outcome)
  id <- unname(outcome_map[outcome])
  integer_option <- function(key, default) {
    x <- suppressWarnings(as.numeric(value(key, default)))
    if (length(x) != 1L || !is.finite(x) || x < 1 || x != floor(x) || x > .Machine$integer.max)
      stop("Expected a positive integer for --", key)
    as.integer(x)
  }
  resume <- value("resume", "false")
  if (!resume %in% c("true", "false")) stop("--resume must be true or false.")
  list(data_root = data_root, output_root = output_root, outcome = outcome, outcome_id = id,
       exposure = value("exposure", file.path(data_root, "exposure_with_rsid.csv")),
       outcome_file = value("outcome-file", file.path(data_root, "outcome_snp", paste0(id, "_MAPLE.tsv"))),
       omega = value("omega", file.path(data_root, "Omega", outcome, "MAPLE_all_proteins_Omega.tsv")),
       ld_scores = value("ld-scores", file.path(data_root, "eur_w_ld_chr")),
       plink = value("plink", "plink"), bfile = value("bfile", file.path(data_root, "reference_panel", "EUR")),
       ld_root = value("ld-root", file.path(output_root, "ld_matrix_local")),
       maple_root = value("maple-root", file.path(output_root, "MAPLE_results")),
       main_root = value("main-root", file.path(output_root, "coloc_susie")),
       sensitivity_root = value("sensitivity-root", file.path(output_root, "coloc_susie_sensitivity")),
       proteins = if (is.null(opts$proteins)) NULL else trimws(strsplit(opts$proteins, ",", fixed = TRUE)[[1]]),
       max_snps = integer_option("max-snps", 2500L), gibbs = integer_option("gibbs", 10000L),
       seed = integer_option("seed", 20260101L), resume = resume == "true")
}

require_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Install required R packages: ", paste(missing, collapse = ", "))
}
require_columns <- function(dat, columns) {
  missing <- setdiff(columns, names(dat))
  if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))
}
write_table <- function(dat, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(dat, file, sep = if (grepl("\\.csv$", file)) "," else "\t")
}
select_proteins <- function(proteins, opts) {
  proteins <- sort(unique(as.character(proteins)))
  if (!is.null(opts$proteins)) {
    missing <- setdiff(opts$proteins, proteins)
    if (length(missing)) stop("Requested proteins not available: ", paste(missing, collapse = ", "))
    proteins <- intersect(proteins, opts$proteins)
  }
  if (!length(proteins)) stop("No eligible proteins.")
  proteins
}
read_harmonized <- function(opts, protein) {
  file <- file.path(opts$maple_root, opts$outcome_id, paste0(safe_name(protein), "_harmonized_for_MAPLE.tsv"))
  dat <- data.table::fread(file)
  require_columns(dat, c("SNP", "beta.exposure", "se.exposure", "beta.outcome", "se.outcome",
                         "samplesize.exposure", "samplesize.outcome", "pos.outcome"))
  dat <- dat[!is.na(SNP) & SNP != "" & is.finite(beta.exposure) & is.finite(beta.outcome) &
               is.finite(se.exposure) & is.finite(se.outcome) & se.exposure > 0 & se.outcome > 0]
  dat <- unique(dat, by = "SNP")
  dat[, `:=`(varbeta.exposure = se.exposure^2, varbeta.outcome = se.outcome^2,
             position_coloc = suppressWarnings(as.integer(pos.outcome)))]
  if (all(is.na(dat$position_coloc))) dat[, position_coloc := seq_len(.N)]
  dat
}
load_ld <- function(opts, protein, snps) {
  prefix <- file.path(opts$ld_root, opts$outcome_id, paste0(safe_name(protein), "_", opts$outcome_id))
  matrix_file <- paste0(prefix, "_Sigma.tsv")
  snp_file <- paste0(prefix, "_Sigma_snps.txt")
  source <- "cached"
  if (file.exists(matrix_file) && file.exists(snp_file)) {
    Sigma <- as.matrix(data.table::fread(matrix_file, header = FALSE))
    ld_snps <- as.character(data.table::fread(snp_file, header = FALSE)[[1]])
  } else {
    require_packages("ieugwasr")
    # Signed PLINK correlations, not r-squared; preserve the original allele order.
    Sigma <- ieugwasr::ld_matrix(snps, bfile = opts$bfile, plink_bin = opts$plink, with_alleles = FALSE)
    ld_snps <- rownames(Sigma)
    dir.create(dirname(matrix_file), recursive = TRUE, showWarnings = FALSE)
    # Cached matrices and SNP lists must have no headers (legacy MAPLE format).
    data.table::fwrite(data.table::as.data.table(Sigma), matrix_file, sep = "\t", col.names = FALSE)
    data.table::fwrite(data.table::data.table(SNP = ld_snps), snp_file, col.names = FALSE)
    source <- "generated"
  }
  storage.mode(Sigma) <- "numeric"
  dimnames(Sigma) <- NULL
  if (anyDuplicated(ld_snps) || !identical(dim(Sigma), rep(length(ld_snps), 2L)) ||
      any(!is.finite(Sigma)) || !isSymmetric(Sigma, tol = 1e-6) ||
      any(abs(diag(Sigma) - 1) > 1e-6)) stop("Invalid signed LD matrix or SNP ordering.")
  list(Sigma = Sigma, snps = ld_snps, source = source)
}
align_ld <- function(dat, ld, minimum = 10L) {
  snps <- intersect(ld$snps, dat$SNP)
  if (length(snps) < minimum) stop("Too few SNPs after LD matching.")
  idx <- match(snps, ld$snps)
  Sigma <- ld$Sigma[idx, idx, drop = FALSE]
  dat <- dat[match(snps, SNP)]
  dimnames(Sigma) <- list(snps, snps)
  stopifnot(identical(as.character(dat$SNP), snps))
  list(dat = dat, Sigma = Sigma)
}
