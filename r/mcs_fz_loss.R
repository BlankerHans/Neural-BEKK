#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop(
    paste(
      "Usage:",
      "Rscript r/mcs_fz_loss.R <fz_loss_matrix.csv> [output_dir] [B] [statistic] [seed]",
      "\nExample:",
      "Rscript r/mcs_fz_loss.R results/runs/asof_2026-05-03_seed42/fz_loss_matrix.csv"
    ),
    call. = FALSE
  )
}

loss_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- if (length(args) >= 2) args[[2]] else dirname(loss_path)
B <- if (length(args) >= 3) as.integer(args[[3]]) else 5000L
statistic <- if (length(args) >= 4) args[[4]] else "Tmax"
seed <- if (length(args) >= 5) as.integer(args[[5]]) else 42L

if (is.na(B) || B <= 0L) {
  stop("B must be a positive integer.", call. = FALSE)
}
if (!statistic %in% c("Tmax", "TR")) {
  stop("statistic must be either 'Tmax' or 'TR'.", call. = FALSE)
}
if (is.na(seed)) {
  stop("seed must be an integer.", call. = FALSE)
}

if (!requireNamespace("MCS", quietly = TRUE)) {
  stop(
    paste(
      "R package 'MCS' is required.",
      "Install it with install.packages('MCS') and rerun this script."
    ),
    call. = FALSE
  )
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

losses <- read.csv(loss_path, check.names = FALSE)
if (ncol(losses) < 2L) {
  stop("MCS requires at least two model loss columns.", call. = FALSE)
}
if (!all(vapply(losses, is.numeric, logical(1)))) {
  stop("All columns in the FZ-loss matrix must be numeric.", call. = FALSE)
}

loss_matrix <- as.matrix(losses)
storage.mode(loss_matrix) <- "double"

if (anyNA(loss_matrix)) {
  stop("NAs in the FZ-loss matrix are not allowed.", call. = FALSE)
}
if (any(!is.finite(loss_matrix))) {
  stop("Infinite values in the FZ-loss matrix are not allowed.", call. = FALSE)
}

run_mcs <- function(alpha) {
  mcs_args <- list(
    Loss = loss_matrix,
    alpha = alpha,
    B = B,
    statistic = statistic,
    verbose = FALSE
  )
  formal_args <- names(formals(MCS::MCSprocedure))
  if ("seed" %in% formal_args) {
    mcs_args$seed <- seed
  } else {
    set.seed(seed)
    if ("cl" %in% formal_args) {
      mcs_args$cl <- NULL
    }
  }

  fit <- do.call(MCS::MCSprocedure, mcs_args)

  details <- as.data.frame(fit@show, check.names = FALSE)
  details <- cbind(model = rownames(fit@show), details)
  details$alpha <- alpha
  details$confidence <- 1 - alpha
  details$mcs_pvalue <- fit@Info$mcs_pvalue
  details$n_eliminated <- fit@Info$n_elim
  details$statistic <- statistic
  details$B <- B
  details$seed <- seed

  suffix <- paste0(round(100 * (1 - alpha)), "pct")
  detail_path <- file.path(output_dir, paste0("mcs_fz_loss_", suffix, ".csv"))
  write.csv(details, detail_path, row.names = FALSE)

  list(
    alpha = alpha,
    confidence = 1 - alpha,
    included = rownames(fit@show),
    p_value = fit@Info$mcs_pvalue,
    n_eliminated = fit@Info$n_elim,
    details_path = detail_path
  )
}

mcs_90 <- run_mcs(0.10)
mcs_95 <- run_mcs(0.05)

summary_table <- data.frame(
  model = colnames(loss_matrix),
  mean_fz_loss = as.numeric(colMeans(loss_matrix)),
  stringsAsFactors = FALSE
)
summary_table$rank <- rank(summary_table$mean_fz_loss, ties.method = "min")
summary_table$mcs_90 <- summary_table$model %in% mcs_90$included
summary_table$mcs_95 <- summary_table$model %in% mcs_95$included
summary_table <- summary_table[order(summary_table$mean_fz_loss), ]

summary_path <- file.path(output_dir, "mcs_fz_loss_summary.csv")
write.csv(summary_table, summary_path, row.names = FALSE)

metadata_path <- file.path(output_dir, "mcs_fz_loss_metadata.txt")
writeLines(
  c(
    paste("loss_path:", loss_path),
    paste("n_obs:", nrow(loss_matrix)),
    paste("n_models:", ncol(loss_matrix)),
    paste("B:", B),
    paste("statistic:", statistic),
    paste("seed:", seed),
    paste("mcs_90_p_value:", mcs_90$p_value),
    paste("mcs_90_n_eliminated:", mcs_90$n_eliminated),
    paste("mcs_95_p_value:", mcs_95$p_value),
    paste("mcs_95_n_eliminated:", mcs_95$n_eliminated),
    paste("summary_csv:", summary_path),
    paste("mcs_90_details_csv:", mcs_90$details_path),
    paste("mcs_95_details_csv:", mcs_95$details_path)
  ),
  con = metadata_path
)

cat("Wrote MCS summary to:", summary_path, "\n")
cat("Wrote MCS metadata to:", metadata_path, "\n")
