#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  pos <- match(flag, args)
  if (is.na(pos) || pos == length(args)) {
    return(default)
  }
  args[[pos + 1L]]
}

project_dir <- normalizePath(get_arg("--project-dir", getwd()), mustWork = TRUE)
run_id <- get_arg("--run-id", format(Sys.Date(), "%Y-%m-%d"))
max_iter <- as.integer(get_arg("--max-iter", "200"))

setwd(project_dir)

local_lib <- file.path(project_dir, "r_libs")
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

if (!requireNamespace("BEKKs", quietly = TRUE)) {
  stop(
    "Package 'BEKKs' is not installed. In R run: install.packages('BEKKs')",
    call. = FALSE
  )
}
suppressPackageStartupMessages(library(BEKKs))

data_dir <- file.path(project_dir, "data")
output_dir <- file.path(project_dir, "r", "output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_path <- function(prefix) {
  file.path(data_dir, paste0(prefix, "_", run_id, ".csv"))
}

read_return_split <- function(path) {
  if (!file.exists(path)) {
    stop("Input file not found: ", path, call. = FALSE)
  }

  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(df) < 2L) {
    stop("CSV must contain a Date column and at least one asset return column: ", path, call. = FALSE)
  }

  dates <- as.Date(df[[1L]])
  if (anyNA(dates)) {
    stop("First column cannot be parsed as Date in: ", path, call. = FALSE)
  }

  returns_df <- df[-1L]
  numeric_cols <- vapply(returns_df, is.numeric, logical(1L))
  if (!all(numeric_cols)) {
    bad_cols <- names(returns_df)[!numeric_cols]
    stop("Non-numeric return columns in ", path, ": ", paste(bad_cols, collapse = ", "), call. = FALSE)
  }

  returns <- as.matrix(returns_df)
  storage.mode(returns) <- "double"

  list(
    dates = dates,
    returns = returns,
    asset_names = colnames(returns)
  )
}

extract_h_array <- function(H_obj, d) {
  H_mat <- as.matrix(H_obj)
  H_arr <- array(NA_real_, dim = c(nrow(H_mat), d, d))

  for (i in seq_len(nrow(H_mat))) {
    H_arr[i, , ] <- matrix(as.numeric(H_mat[i, ]), nrow = d, byrow = TRUE)
  }

  H_arr
}

forecast_bekk_oos <- function(fit, returns_oos) {
  d <- ncol(returns_oos)
  H_in <- extract_h_array(fit$H_t, d)

  C0 <- fit$C0
  A <- fit$A
  G <- fit$G
  const <- C0 %*% t(C0)
  is_asymmetric <- isTRUE(fit$asymmetric)
  B <- if (is_asymmetric) fit$B else NULL
  signs <- if (is_asymmetric) matrix(fit$signs, ncol = 1L) else NULL

  prev_return <- matrix(as.numeric(tail(fit$data, 1L)), ncol = 1L)
  H_prev <- H_in[dim(H_in)[1L], , ]

  H_oos <- array(NA_real_, dim = c(nrow(returns_oos), d, d))

  for (t in seq_len(nrow(returns_oos))) {
    H_t <- const +
      t(A) %*% (prev_return %*% t(prev_return)) %*% A +
      t(G) %*% H_prev %*% G

    if (is_asymmetric) {
      asym_return <- prev_return * as.numeric(signs * prev_return > 0)
      H_t <- H_t + t(B) %*% (asym_return %*% t(asym_return)) %*% B
    }

    H_t <- (H_t + t(H_t)) / 2
    H_oos[t, , ] <- H_t

    prev_return <- matrix(returns_oos[t, ], ncol = 1L)
    H_prev <- H_t
  }

  H_oos
}

covariances_to_df <- function(H_arr, dates) {
  d <- dim(H_arr)[2L]
  out <- data.frame(Date = dates, check.names = FALSE)

  for (i in seq_len(d)) {
    for (j in seq_len(d)) {
      out[[paste0("h_", i, j)]] <- H_arr[, i, j]
    }
  }

  out
}

write_forecasts <- function(H_arr, dates, model_name) {
  forecast_df <- covariances_to_df(H_arr, dates)

  versioned_path <- file.path(output_dir, paste0("bekk_forecasts_", model_name, "_", run_id, ".csv"))

  write.csv(forecast_df, versioned_path, row.names = FALSE)

  versioned_path
}

old_forecast_files <- list.files(
  output_dir,
  pattern = "^bekk_forecasts_(symmetric|asymmetric).*[.]csv$",
  full.names = TRUE
)
if (length(old_forecast_files) > 0L) {
  unlink(old_forecast_files)
}

logret_obj <- read_return_split(input_path("logret"))
train_obj <- read_return_split(input_path("train_df"))
val_obj <- read_return_split(input_path("val_df"))
test_obj <- read_return_split(input_path("test_df"))

train_data <- train_obj$returns
val_data <- val_obj$returns
test_data <- test_obj$returns
test_dates <- test_obj$dates
asset_names <- train_obj$asset_names

if (!identical(asset_names, val_obj$asset_names) || !identical(asset_names, test_obj$asset_names)) {
  stop("Asset columns differ between train, validation, and test CSVs.", call. = FALSE)
}

if (!identical(asset_names, logret_obj$asset_names)) {
  stop("Asset columns differ between logret and split CSVs.", call. = FALSE)
}

train_val_data <- rbind(train_data, val_data)

fit_and_forecast <- function(asymmetric, model_name) {
  spec <- bekk_spec(model = list(type = "bekk", asymmetric = asymmetric))
  fit <- bekk_fit(
    spec = spec,
    data = train_val_data,
    QML_t_ratios = TRUE,
    max_iter = max_iter
  )

  forecasts <- forecast_bekk_oos(fit, test_data)
  write_forecasts(forecasts, test_dates, model_name)
}

symmetric_path <- fit_and_forecast(
  asymmetric = FALSE,
  model_name = "symmetric"
)

asymmetric_path <- fit_and_forecast(
  asymmetric = TRUE,
  model_name = "asymmetric"
)

cat("BEKK(1,1) forecasts finished\n")
cat("run_id: ", run_id, "\n", sep = "")
cat("project_dir: ", project_dir, "\n", sep = "")
cat("output_dir: ", output_dir, "\n", sep = "")
cat("assets: ", paste(asset_names, collapse = ", "), "\n", sep = "")
cat("train / val / test rows: ", nrow(train_data), " / ", nrow(val_data), " / ", nrow(test_data), "\n", sep = "")
cat("symmetric forecast CSV: ", symmetric_path, "\n", sep = "")
cat("asymmetric forecast CSV: ", asymmetric_path, "\n", sep = "")
