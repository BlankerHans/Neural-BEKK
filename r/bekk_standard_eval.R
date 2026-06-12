#!/usr/bin/env Rscript

local_lib <- "r_libs"
if (dir.exists(local_lib)) {
  .libPaths(c(normalizePath(local_lib), .libPaths()))
}

if (!requireNamespace("BEKKs", quietly = TRUE)) {
  stop(
    "Package 'BEKKs' is not installed. In R run: install.packages('BEKKs')",
    call. = FALSE
  )
}

read_returns_csv <- function(path) {
  if (!file.exists(path)) {
    stop(
      paste0(
        "Input file not found: ", path, "\n",
        "Export first the exact Python notebook returns, e.g.\n",
        "logret[feature_cols].to_csv('data/logret_nn.csv', index_label='Date')"
      ),
      call. = FALSE
    )
  }

  df <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(df) < 3) {
    stop("CSV must contain a date column and at least two return series.", call. = FALSE)
  }

  first_col <- names(df)[1]
  date_like <- tolower(first_col) %in% c("", "date", "index", "x")

  if (date_like) {
    dates <- as.Date(df[[1]])
    num_df <- df[-1]
  } else {
    dates <- seq_len(nrow(df))
    num_df <- df
  }

  numeric_cols <- vapply(num_df, is.numeric, logical(1))
  if (!all(numeric_cols)) {
    stop("All asset columns must be numeric returns.", call. = FALSE)
  }

  returns <- as.matrix(num_df)
  storage.mode(returns) <- "double"

  list(
    dates = dates,
    returns = returns,
    colnames = colnames(returns)
  )
}

chronological_split <- function(n, train_size = 0.7, val_size = 0.15) {
  train_end <- as.integer(n * train_size)
  val_end <- as.integer(n * (train_size + val_size))

  list(
    train = seq_len(train_end),
    val = seq.int(train_end + 1L, val_end),
    test = seq.int(val_end + 1L, n)
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

gaussian_nll_path <- function(y, H_arr, jitter = 1e-10) {
  d <- ncol(y)
  out <- numeric(nrow(y))

  for (i in seq_len(nrow(y))) {
    H_i <- H_arr[i, , ]
    H_i <- (H_i + t(H_i)) / 2
    diag(H_i) <- diag(H_i) + jitter

    det_i <- determinant(H_i, logarithm = TRUE)
    mahal_i <- drop(t(y[i, ]) %*% solve(H_i, y[i, ]))

    out[i] <- 0.5 * (d * log(2 * pi) + as.numeric(det_i$modulus) + mahal_i)
  }

  out
}

forecast_bekk_oos <- function(fit, returns_oos) {
  d <- ncol(returns_oos)
  H_in <- extract_h_array(fit$H_t, d)

  C0 <- fit$C0
  A <- fit$A
  G <- fit$G
  const <- C0 %*% t(C0)

  prev_return <- matrix(as.numeric(tail(fit$data, 1)), ncol = 1)
  H_prev <- H_in[dim(H_in)[1], , ]

  H_oos <- array(NA_real_, dim = c(nrow(returns_oos), d, d))

  for (t in seq_len(nrow(returns_oos))) {
    H_t <- const +
      t(A) %*% (prev_return %*% t(prev_return)) %*% A +
      t(G) %*% H_prev %*% G

    H_t <- (H_t + t(H_t)) / 2
    H_oos[t, , ] <- H_t

    prev_return <- matrix(returns_oos[t, ], ncol = 1)
    H_prev <- H_t
  }

  H_oos
}

portfolio_from_cov <- function(returns, H_arr, weights, dates) {
  if (length(weights) != ncol(returns)) {
    stop("Length of portfolio weights must match number of assets.", call. = FALSE)
  }

  r_p <- drop(returns %*% weights)
  var_p <- vapply(
    seq_len(nrow(returns)),
    function(i) drop(t(weights) %*% H_arr[i, , ] %*% weights),
    numeric(1)
  )
  vol_p <- sqrt(pmax(var_p, 1e-12))

  data.frame(
    Date = dates,
    r_p = r_p,
    var_p = var_p,
    vol_p = vol_p,
    check.names = FALSE
  )
}

fhs_var_es <- function(z_hist_init, r_oos, vol_oos, alpha = 0.05, window = 500, min_vol = 1e-12) {
  hist <- as.numeric(z_hist_init)
  if (!is.null(window) && length(hist) > window) {
    hist <- tail(hist, window)
  }

  n <- length(r_oos)
  var <- numeric(n)
  es <- numeric(n)
  hits <- logical(n)

  for (t in seq_len(n)) {
    q_alpha <- as.numeric(stats::quantile(hist, probs = alpha, names = FALSE, type = 7))
    hist_tail <- hist[hist <= q_alpha]

    var[t] <- vol_oos[t] * q_alpha
    es[t] <- vol_oos[t] * mean(hist_tail)
    hits[t] <- r_oos[t] <= var[t]

    z_new <- r_oos[t] / max(vol_oos[t], min_vol)
    hist <- c(hist, z_new)

    if (!is.null(window) && length(hist) > window) {
      hist <- tail(hist, window)
    }
  }

  list(var = var, es = es, hits = hits)
}

safe_xlogy <- function(count, prob) {
  if (count == 0) {
    return(0)
  }
  if (!is.finite(prob) || prob <= 0 || prob > 1) {
    return(-Inf)
  }
  count * log(prob)
}

binom_loglik <- function(successes, total, prob) {
  failures <- total - successes
  safe_xlogy(failures, 1 - prob) + safe_xlogy(successes, prob)
}

kupiec_uc_test <- function(returns, var, alpha) {
  hits <- returns <= var
  n_obs <- length(hits)
  n_ex <- sum(hits)
  hit_rate <- n_ex / n_obs

  ll_null <- binom_loglik(n_ex, n_obs, alpha)
  ll_alt <- binom_loglik(n_ex, n_obs, hit_rate)
  lr_uc <- max(0, 2 * (ll_alt - ll_null))
  p_value <- stats::pchisq(lr_uc, df = 1, lower.tail = FALSE)

  list(
    test = "kupiec_uc",
    alpha = alpha,
    n_obs = n_obs,
    n_exceptions = n_ex,
    expected_exceptions = alpha * n_obs,
    hit_rate = hit_rate,
    lr_uc = lr_uc,
    p_value = p_value
  )
}

christoffersen_cc_test <- function(returns, var, alpha) {
  hits <- as.integer(returns <= var)
  n_obs <- length(hits)

  if (n_obs < 2) {
    stop("Christoffersen test requires at least two observations.", call. = FALSE)
  }

  prev_hits <- hits[-n_obs]
  next_hits <- hits[-1]

  n00 <- sum(prev_hits == 0 & next_hits == 0)
  n01 <- sum(prev_hits == 0 & next_hits == 1)
  n10 <- sum(prev_hits == 1 & next_hits == 0)
  n11 <- sum(prev_hits == 1 & next_hits == 1)

  n_transitions <- n00 + n01 + n10 + n11
  pi_hat <- mean(hits)
  pi_01 <- if ((n00 + n01) > 0) n01 / (n00 + n01) else NA_real_
  pi_11 <- if ((n10 + n11) > 0) n11 / (n10 + n11) else NA_real_
  pi_pooled <- (n01 + n11) / n_transitions

  ll_ind_null <- binom_loglik(n01 + n11, n_transitions, pi_pooled)
  ll_ind_alt <- binom_loglik(n01, n00 + n01, pi_01) +
    binom_loglik(n11, n10 + n11, pi_11)

  lr_ind <- max(0, 2 * (ll_ind_alt - ll_ind_null))
  p_value_ind <- stats::pchisq(lr_ind, df = 1, lower.tail = FALSE)

  ll_uc_null <- binom_loglik(sum(hits), n_obs, alpha)
  ll_uc_alt <- binom_loglik(sum(hits), n_obs, pi_hat)
  lr_uc <- max(0, 2 * (ll_uc_alt - ll_uc_null))
  lr_cc <- max(0, lr_uc + lr_ind)
  p_value_cc <- stats::pchisq(lr_cc, df = 2, lower.tail = FALSE)

  list(
    test = "christoffersen_cc",
    alpha = alpha,
    n_obs = n_obs,
    n_exceptions = sum(hits),
    hit_rate = pi_hat,
    transition_counts = list(n00 = n00, n01 = n01, n10 = n10, n11 = n11),
    transition_probs = list(pi_01 = pi_01, pi_11 = pi_11, pi_pooled = pi_pooled),
    lr_ind = lr_ind,
    p_value_ind = p_value_ind,
    lr_uc = lr_uc,
    lr_cc = lr_cc,
    p_value_cc = p_value_cc
  )
}

run_optional_esback <- function(returns, var, es, volatility, alpha) {
  if (!requireNamespace("esback", quietly = TRUE)) {
    return(NULL)
  }

  cov_config <- list(sparsity = "iid", sigma_est = "scl_sp", misspec = TRUE)

  list(
    cc = tryCatch(
      esback::cc_backtest(r = returns, q = var, e = es, s = volatility, alpha = alpha, hommel = TRUE),
      error = function(e) list(error = conditionMessage(e))
    ),
    er = tryCatch(
      esback::er_backtest(r = returns, q = var, e = es, s = volatility, B = 1000),
      error = function(e) list(error = conditionMessage(e))
    ),
    esr_v1 = tryCatch(
      esback::esr_backtest(r = returns, q = var, e = es, alpha = alpha, version = 1, B = 0, cov_config = cov_config),
      error = function(e) list(error = conditionMessage(e))
    ),
    esr_v2 = tryCatch(
      esback::esr_backtest(r = returns, q = var, e = es, alpha = alpha, version = 2, B = 0, cov_config = cov_config),
      error = function(e) list(error = conditionMessage(e))
    ),
    esr_v3 = tryCatch(
      esback::esr_backtest(r = returns, q = var, e = es, alpha = alpha, version = 3, B = 0, cov_config = cov_config),
      error = function(e) list(error = conditionMessage(e))
    )
  )
}

args <- commandArgs(trailingOnly = TRUE)

input_csv <- if (length(args) >= 1) args[1] else "data/logret_nn.csv"
output_dir <- if (length(args) >= 2) args[2] else "results/bekk_standard"

alpha <- 0.05
window <- 500
weights <- c(0.5, 0.4, 0.1)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data_obj <- read_returns_csv(input_csv)
dates <- data_obj$dates
returns <- data_obj$returns
asset_names <- data_obj$colnames

if (length(weights) != ncol(returns)) {
  stop(
    sprintf(
      "Weights length (%d) does not match number of assets (%d).",
      length(weights),
      ncol(returns)
    ),
    call. = FALSE
  )
}

colnames(returns) <- asset_names

idx <- chronological_split(nrow(returns), train_size = 0.7, val_size = 0.15)
train_returns <- returns[idx$train, , drop = FALSE]
val_returns <- returns[idx$val, , drop = FALSE]
test_returns <- returns[idx$test, , drop = FALSE]

train_val_returns <- rbind(train_returns, val_returns)
train_val_dates <- dates[c(idx$train, idx$val)]
test_dates <- dates[idx$test]

spec <- BEKKs::bekk_spec(model = list(type = "bekk", asymmetric = FALSE))
fit <- BEKKs::bekk_fit(
  spec = spec,
  data = train_val_returns,
  QML_t_ratios = FALSE,
  max_iter = 50,
  crit = 1e-9
)

test_cov <- forecast_bekk_oos(fit, test_returns)
test_nll_path <- gaussian_nll_path(test_returns, test_cov)

train_val_cov <- extract_h_array(fit$H_t, ncol(train_val_returns))

portfolio_train_val <- portfolio_from_cov(
  returns = train_val_returns,
  H_arr = train_val_cov,
  weights = weights,
  dates = train_val_dates
)

portfolio_test <- portfolio_from_cov(
  returns = test_returns,
  H_arr = test_cov,
  weights = weights,
  dates = test_dates
)

z_hist_init <- portfolio_train_val$r_p / pmax(portfolio_train_val$vol_p, 1e-12)
fhs_test <- fhs_var_es(
  z_hist_init = z_hist_init,
  r_oos = portfolio_test$r_p,
  vol_oos = portfolio_test$vol_p,
  alpha = alpha,
  window = window
)

portfolio_test$var_fhs <- fhs_test$var
portfolio_test$es_fhs <- fhs_test$es
portfolio_test$hit_fhs <- fhs_test$hits

var_tests <- list(
  kupiec = kupiec_uc_test(portfolio_test$r_p, portfolio_test$var_fhs, alpha),
  christoffersen = christoffersen_cc_test(portfolio_test$r_p, portfolio_test$var_fhs, alpha)
)

esback_tests <- run_optional_esback(
  returns = portfolio_test$r_p,
  var = portfolio_test$var_fhs,
  es = portfolio_test$es_fhs,
  volatility = portfolio_test$vol_p,
  alpha = alpha
)

summary_metrics <- data.frame(
  model = "symmetric_bekk_11",
  alpha = alpha,
  n_train = nrow(train_returns),
  n_val = nrow(val_returns),
  n_test = nrow(test_returns),
  mean_test_nll = mean(test_nll_path),
  test_hit_rate = mean(portfolio_test$hit_fhs),
  expected_hit_rate = alpha,
  mean_test_vol = mean(portfolio_test$vol_p),
  stringsAsFactors = FALSE
)

write.csv(summary_metrics, file.path(output_dir, "summary_metrics.csv"), row.names = FALSE)
write.csv(portfolio_test, file.path(output_dir, "portfolio_test.csv"), row.names = FALSE)
saveRDS(test_cov, file.path(output_dir, "test_covariance_array.rds"))
saveRDS(
  list(
    summary_metrics = summary_metrics,
    var_tests = var_tests,
    esback_tests = esback_tests,
    bekk_fit = fit
  ),
  file.path(output_dir, "full_results.rds")
)

cat("\nBEKK test evaluation finished.\n")
cat("Input CSV: ", input_csv, "\n", sep = "")
cat("Output dir: ", output_dir, "\n", sep = "")
cat("Assets: ", paste(asset_names, collapse = ", "), "\n", sep = "")
cat("Train / Val / Test: ", nrow(train_returns), " / ", nrow(val_returns), " / ", nrow(test_returns), "\n", sep = "")
cat("Mean Gaussian test NLL: ", sprintf("%.6f", mean(test_nll_path)), "\n", sep = "")
cat("Test FHS hit rate: ", sprintf("%.4f", mean(portfolio_test$hit_fhs)), "\n", sep = "")
cat("Expected hit rate: ", sprintf("%.4f", alpha), "\n", sep = "")
cat("Kupiec p-value: ", sprintf("%.6f", var_tests$kupiec$p_value), "\n", sep = "")
cat("Christoffersen CC p-value: ", sprintf("%.6f", var_tests$christoffersen$p_value_cc), "\n", sep = "")
