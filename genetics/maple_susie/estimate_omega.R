# Estimate MAPLE sample-structure parameters from genome-wide summary data.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
source(file.path(dirname(script), "common.R"))
opts <- read_options()
require_packages(c("data.table", "MAPLE"))
library(data.table)
exposure <- fread(opts$exposure)
outcome <- fread(opts$outcome_file)
columns <- c("SNP", "b", "se", "frq_A1", "A1", "A2", "P", "N")
require_columns(exposure, c(columns, "Protein"))
require_columns(outcome, columns)
proteins <- select_proteins(exposure$Protein, opts)
set.seed(opts$seed)
rows <- lapply(proteins, function(p) {
  x <- exposure[Protein == p, ..columns]
  pars <- MAPLE::est_SS(dat1 = as.data.frame(x), dat2 = as.data.frame(outcome[, ..columns]),
                        trait1.name = p, trait2.name = opts$outcome,
                        ldscore.dir = opts$ld_scores)
  data.table(Protein = p, t1 = pars$Omega[1, 1], t2 = pars$Omega[2, 2],
             t12 = pars$Omega[1, 2], t1_se = pars$Omega.se[1, 1],
             t2_se = pars$Omega.se[2, 2], t12_se = pars$Omega.se[1, 2],
             exposure_n_snp = nrow(x), outcome_n_snp = nrow(outcome))
})
write_table(rbindlist(rows), opts$omega)
message("Sample-structure estimation completed: ", opts$omega)
