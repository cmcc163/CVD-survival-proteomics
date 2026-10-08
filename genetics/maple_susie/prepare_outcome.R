# Match local GWAS summary statistics to each protein's exposure SNPs.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
opts <- read_options()
require_packages("data.table")
library(data.table)
exp <- fread(opts$exposure)
raw <- fread(opts$outcome_file)
require_columns(exp, c("SNP", "id.exposure"))
require_columns(raw, c("SNP", "CHR", "BP", "REF", "ALT", "BETA", "SE", "AF", "LP", "SS"))
if (anyDuplicated(raw$SNP)) stop("Outcome SNP identifiers must be unique.")
out <- raw[, .(SNP = as.character(SNP), chr.outcome = CHR, pos.outcome = BP,
               effect_allele.outcome = ALT, other_allele.outcome = REF,
               beta.outcome = BETA, se.outcome = SE, eaf.outcome = AF,
               pval.outcome = 10^(-LP), samplesize.outcome = SS,
               id.outcome = opts$outcome_id, outcome = opts$outcome_id)]
proteins <- select_proteins(exp$id.exposure, opts)
rows <- lapply(proteins, function(p) {
  snps <- unique(exp[id.exposure == p, SNP])
  dat <- out[SNP %in% snps]
  dat[, protein_id := p]
  dat
})
combined <- rbindlist(rows)
if (!nrow(combined)) stop("No overlapping exposure/outcome SNPs.")
target <- file.path(opts$output_root, "outcome_data_by_protein", opts$outcome_id,
                    paste0("all_proteins_outcome_", opts$outcome_id, ".csv"))
write_table(combined, target)
message("Outcome preparation completed: ", target)
