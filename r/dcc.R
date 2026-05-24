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
data_id <- get_arg("--data-id", run_id)
seed <- as.integer(get_arg("--seed", "42"))
solver <- get_arg("--solver", "nlminb")

RNGkind("Mersenne-Twister", "Inversion", "Rejection")
set.seed(seed)

setwd(project_dir)

local_lib <- file.path(project_dir, "r_libs")
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

if (!requireNamespace("rugarch", quietly = TRUE)) {
  stop(
    "Package 'rugarch' is not installed. In R run: install.packages('rugarch')",
    call. = FALSE
  )
}

if (!requireNamespace("rmgarch", quietly = TRUE)) {
  stop(
    "Package 'rmgarch' is not installed. In R run: install.packages('rmgarch')",
    call. = FALSE
  )
}

data_dir <- file.path(project_dir, "data")
output_dir <- file.path(project_dir, "r", "output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_path <- function(prefix) {
  file.path(data_dir, paste0(prefix, "_", data_id, ".csv"))
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
  versioned_path <- file.path(output_dir, paste0("dcc_forecasts_", model_name, "_", run_id, ".csv"))

  write.csv(forecast_df, versioned_path, row.names = FALSE)

  versioned_path
}

extract_fitted_covariances <- function(fit, expected_n, d) {
  H_raw <- rmgarch::rcov(fit)
  dims <- dim(H_raw)

  if (length(dims) != 3L || dims[1L] != d || dims[2L] != d || dims[3L] != expected_n) {
    stop(
      "Unexpected fitted covariance dimensions: ",
      paste(dims, collapse = "x"),
      "; expected ",
      paste(c(d, d, expected_n), collapse = "x"),
      call. = FALSE
    )
  }

  aperm(H_raw, c(3L, 1L, 2L))
}

extract_forecast_covariances <- function(forecast, expected_n, d) {
  H_list <- rmgarch::rcov(forecast)

  if (!is.list(H_list) || length(H_list) != expected_n) {
    stop(
      "Unexpected forecast covariance output length: ",
      length(H_list),
      "; expected ",
      expected_n,
      call. = FALSE
    )
  }

  H_arr <- array(NA_real_, dim = c(expected_n, d, d))

  for (t in seq_len(expected_n)) {
    H_t <- H_list[[t]]
    dims <- dim(H_t)

    if (length(dims) != 3L || dims[1L] != d || dims[2L] != d || dims[3L] < 1L) {
      stop(
        "Unexpected forecast covariance dimensions at step ",
        t,
        ": ",
        paste(dims, collapse = "x"),
        call. = FALSE
      )
    }

    H_arr[t, , ] <- H_t[, , 1L]
  }

  H_arr
}

build_dcc_spec <- function(d, model_type) {
  ugarch_spec <- rugarch::ugarchspec(
    mean.model = list(armaOrder = c(0L, 0L), include.mean = FALSE),
    variance.model = list(model = "sGARCH", garchOrder = c(1L, 1L)),
    distribution.model = "norm"
  )

  multispec <- rugarch::multispec(replicate(d, ugarch_spec, simplify = FALSE))

  rmgarch::dccspec(
    uspec = multispec,
    dccOrder = c(1L, 1L),
    model = model_type,
    distribution = "mvnorm"
  )
}

warn_on_convergence <- function(fit, model_name) {
  convergence <- tryCatch(fit@mfit$convergence, error = function(e) NA)

  if (length(convergence) > 0L && any(is.finite(convergence)) && any(convergence != 0L)) {
    warning(
      model_name,
      " fit reported non-zero convergence code(s): ",
      paste(convergence, collapse = ", "),
      call. = FALSE
    )
  }
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
train_val_dates <- c(train_obj$dates, val_obj$dates)
full_data <- rbind(train_val_data, test_data)

colnames(full_data) <- asset_names

fit_and_forecast <- function(model_type, model_name, seed_offset) {
  set.seed(seed + seed_offset)

  d <- ncol(full_data)
  spec <- build_dcc_spec(d, model_type)
  fit <- rmgarch::dccfit(
    spec = spec,
    data = full_data,
    out.sample = nrow(test_data),
    solver = solver,
    fit.control = list(eval.se = FALSE)
  )
  warn_on_convergence(fit, model_name)

  fitted_train_val <- extract_fitted_covariances(
    fit = fit,
    expected_n = nrow(train_val_data),
    d = d
  )
  train_val_path <- write_forecasts(
    fitted_train_val,
    train_val_dates,
    paste0("fitted_train_val_", model_name)
  )

  forecast <- rmgarch::dccforecast(
    fit,
    n.ahead = 1L,
    n.roll = nrow(test_data) - 1L
  )
  forecasts <- extract_forecast_covariances(
    forecast = forecast,
    expected_n = nrow(test_data),
    d = d
  )
  forecast_path <- write_forecasts(forecasts, test_dates, model_name)

  list(
    train_val_path = train_val_path,
    forecast_path = forecast_path
  )
}

dcc_paths <- fit_and_forecast(
  model_type = "DCC",
  model_name = "dcc",
  seed_offset = 0L
)

adcc_paths <- fit_and_forecast(
  model_type = "aDCC",
  model_name = "adcc",
  seed_offset = 1L
)

cat("DCC(1,1) forecasts finished\n")
cat("data_id: ", data_id, "\n", sep = "")
cat("run_id: ", run_id, "\n", sep = "")
cat("seed: ", seed, "\n", sep = "")
cat("solver: ", solver, "\n", sep = "")
cat("project_dir: ", project_dir, "\n", sep = "")
cat("output_dir: ", output_dir, "\n", sep = "")
cat("assets: ", paste(asset_names, collapse = ", "), "\n", sep = "")
cat("train / val / test rows: ", nrow(train_data), " / ", nrow(val_data), " / ", nrow(test_data), "\n", sep = "")
cat("dcc train+val CSV: ", dcc_paths$train_val_path, "\n", sep = "")
cat("dcc forecast CSV: ", dcc_paths$forecast_path, "\n", sep = "")
cat("adcc train+val CSV: ", adcc_paths$train_val_path, "\n", sep = "")
cat("adcc forecast CSV: ", adcc_paths$forecast_path, "\n", sep = "")
