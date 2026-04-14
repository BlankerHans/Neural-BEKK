library(BEKKs)
library(rugarch)
library(rmgarch)
library(PerformanceAnalytics)
library(quantmod)
library(xts)
library(glue)

# Macro Portfolio ---------------------------------------------------------
setwd("/Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk")

macro_tickers <- c(
              "^GSPC",  # Equity (S&P 500)
              "GC=F",   # Gold
              "CL=F",   # Oil
              "^TNX") # r_m und r_i (MSCI World und BTC)


## daten aus python export --------
heute <- Sys.Date()
df <- read.csv(glue("data/logret_{heute}.csv"), check.names = FALSE)
train_df <- read.csv(glue("data/train_df_{heute}.csv"), check.names = FALSE)
val_df <- read.csv(glue("data/val_df_{heute}.csv"), check.names = FALSE)
test_df <- read.csv(glue("data/test_df_{heute}.csv"), check.names = FALSE)

train_data <- as.matrix(train_df[, -1])
val_data <- as.matrix(val_df[, -1])
test_data <- as.matrix(test_df[, -1])

storage.mode(train_data) <- "double"
storage.mode(val_data)   <- "double"
storage.mode(test_data)  <- "double"

dates <- as.Date(df$Date)
returns_xts <- xts(df[, -1], order.by = dates)

returns <- coredata(returns_xts)
storage.mode(returns) <- "double"

charts.PerformanceSummary(returns_xts)
cor(returns_xts)

head(returns_xts)
dim(returns_xts)
colnames(returns)
nrow(returns_xts)

nrow(train_data) + nrow(val_data) + nrow(test_data)

# train val combined
train_val_data <- rbind(train_data, val_data)

## BEKK(1, 1) -----

spec_bekk <- bekk_spec()

system.time(fit_bekk <- bekk_fit(
  spec = spec_bekk,
  data = train_val_data,
  QML_t_ratios = TRUE,
  max_iter = 200
))
summary(fit_bekk)

## asymmetric BEKK -----

spec_asbekk <- bekk_spec(model=list(type="bekk", asymmetric = TRUE))

system.time(fit_asbekk <- bekk_fit(
  spec = spec_asbekk,
  data = train_val_data,
  QML_t_ratios = TRUE,
  max_iter = 200
))
summary(fit_asbekk)

## portfolio returns -----
weights <- c(0.25, 0.25, 0.25, 0.25)
w <- matrix(weights, ncol = 1)

port_ret <- xts(as.matrix(returns_xts) %*% w, order.by = index(returns_xts))
colnames(port_ret) <- "portfolio_return"

H_bekk <- fit_bekk$H_t

var_p <- apply(H_bekk, 1, function(x) {
  H <- matrix(as.numeric(x), 4, 4)
  as.numeric(t(w) %*% H %*% w)
})

vol_p <- sqrt(var_p)

port_var_xts <- xts(var_p, order.by = index(returns_xts))
port_vol_xts <- xts(vol_p, order.by = index(returns_xts))
colnames(port_var_xts) <- "portfolio_variance"
colnames(port_vol_xts) <- "portfolio_volatility"

var <- BEKKs:::VaR(fit_bekk, p = 0.99, portfolio_weights = weights)
str(var)

port_df <- merge(port_ret, port_var_xts, port_vol_xts)
tail(port_df)

## forecasts ----

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


# forecast array mit bekk cov matrices
bekk_forecasts <- forecast_bekk_oos(fit_bekk, test_data)
asbekk_forecast <- forecast_bekk_oos(fit_asbekk, test_data)


bekk_portfolio_var <- apply(bekk_forecasts, 1, function(H_t) {
  as.numeric(t(w) %*% H_t %*% w)
})

asbekk_portfolio_var <- apply(asbekk_forecast, 1, function(H_t) {
  as.numeric(t(w) %*% H_t %*% w)
})


## backtest -----
backtest_bekk <- backtest(fit_bekk, portfolio_weights = weights)



# Macro + BTC -------------------------------------------------------------


