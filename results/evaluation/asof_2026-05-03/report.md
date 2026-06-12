# Evaluation report — asof_2026-05-03

- Seeds: 11 ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42])
- Out-of-sample observations: 925, VaR level alpha = 0.01
- Backtest rejection level: 0.05, MCS size: 0.1 (90% set, 1000 bootstrap reps)
- Deterministic models (identical across seeds): adcc, bekk_asymmetric, bekk_symmetric, dcc

Best mean FZ0 loss: **neural_bekk_asym_diag** (-2.8938). Models passing the adequacy gate: 5/12 (neural_bekk_asym_diag, bekk_lstm_mix, bekk_lstm_scalar, lstm_normal, lstm_student).

Note: seeds share the same test window; the seed std measures training instability, not sampling uncertainty.

## table1_main

| Model                     | FZ0 loss | Std (seeds) | Hit rate | MCS (90%) | MCS p-value | Adequacy                     |
| ------------------------- | -------- | ----------- | -------- | --------- | ----------- | ---------------------------- |
| Neural BEKK (asym. diag.) | -2.8938  | 0.0256      | 0.0101   | 11/11     | 0.8566      | pass                         |
| Neural BEKK (mix)         | -2.8924  | 0.0106      | 0.0104   | 11/11     | 0.7813      | pass                         |
| Neural BEKK (scalar)      | -2.8891  | 0.0157      | 0.0117   | 11/11     | 0.7754      | pass                         |
| GRU (Normal)              | -2.8695  | 0.0266      | 0.0096   | 11/11     | 0.6755      | fail (dq)                    |
| Neural BEKK (vector)      | -2.8620  | 0.0175      | 0.0114   | 11/11     | 0.5265      | fail (christoffersen_cc, dq) |
| GRU (Student-t)           | -2.8509  | 0.0205      | 0.0092   | 9/11      | 0.4306      | fail (dq)                    |
| BEKK (symmetric)          | -2.8461  | --          | 0.0141   | 11/11     | 0.4835      | fail (christoffersen_cc, dq) |
| BEKK (asymmetric)         | -2.8433  | --          | 0.0141   | 11/11     | 0.4885      | fail (dq)                    |
| LSTM (Normal)             | -2.8384  | 0.0263      | 0.0118   | 11/11     | 0.6387      | pass                         |
| DCC                       | -2.8152  | --          | 0.0141   | 10/11     | 0.2848      | fail (christoffersen_cc, dq) |
| ADCC                      | -2.8152  | --          | 0.0141   | 10/11     | 0.2848      | fail (christoffersen_cc, dq) |
| LSTM (Student-t)          | -2.8144  | 0.0212      | 0.0114   | 10/11     | 0.4531      | pass                         |

## table2_var_backtests

| Model                     | Hit rate | Exceptions | Expected | Kupiec | Chr. ind. | Chr. CC | DQ    |
| ------------------------- | -------- | ---------- | -------- | ------ | --------- | ------- | ----- |
| Neural BEKK (asym. diag.) | 0.0101   | 9.3636     | 9.2500   | 0/11   | 0/11      | 0/11    | 3/11  |
| Neural BEKK (mix)         | 0.0104   | 9.6364     | 9.2500   | 0/11   | 0/11      | 0/11    | 3/11  |
| Neural BEKK (scalar)      | 0.0117   | 10.8182    | 9.2500   | 0/11   | 0/11      | 0/11    | 0/11  |
| GRU (Normal)              | 0.0096   | 8.9091     | 9.2500   | 0/11   | 2/11      | 2/11    | 6/11  |
| Neural BEKK (vector)      | 0.0114   | 10.5455    | 9.2500   | 0/11   | 6/11      | 6/11    | 7/11  |
| GRU (Student-t)           | 0.0092   | 8.5455     | 9.2500   | 0/11   | 2/11      | 2/11    | 7/11  |
| BEKK (symmetric)          | 0.0141   | 13.0000    | 9.2500   | 0/11   | 11/11     | 11/11   | 11/11 |
| BEKK (asymmetric)         | 0.0141   | 13.0000    | 9.2500   | 0/11   | 0/11      | 0/11    | 11/11 |
| LSTM (Normal)             | 0.0118   | 10.9091    | 9.2500   | 0/11   | 0/11      | 0/11    | 3/11  |
| DCC                       | 0.0141   | 13.0000    | 9.2500   | 0/11   | 11/11     | 11/11   | 11/11 |
| ADCC                      | 0.0141   | 13.0000    | 9.2500   | 0/11   | 11/11     | 11/11   | 11/11 |
| LSTM (Student-t)          | 0.0114   | 10.5455    | 9.2500   | 0/11   | 1/11      | 1/11    | 5/11  |

## table3_es_backtests

| Model                     | ER   | CC   | ESR (v1) | ESR (v2) | ESR (v3) |
| ------------------------- | ---- | ---- | -------- | -------- | -------- |
| Neural BEKK (asym. diag.) | 0/11 | 0/11 | 6/11     | 8/11     | 0/11     |
| Neural BEKK (mix)         | 0/11 | 0/11 | 5/11     | 6/11     | 0/11     |
| Neural BEKK (scalar)      | 0/11 | 0/11 | 0/11     | 0/11     | 0/11     |
| GRU (Normal)              | 0/11 | 0/11 | 0/11     | 2/11     | 0/11     |
| Neural BEKK (vector)      | 0/11 | 0/11 | 1/11     | 2/11     | 0/11     |
| GRU (Student-t)           | 0/11 | 0/11 | 1/11     | 0/11     | 0/11     |
| BEKK (symmetric)          | 0/11 | 0/11 | 0/11     | 0/11     | 0/11     |
| BEKK (asymmetric)         | 0/11 | 0/11 | 0/11     | 0/11     | 0/11     |
| LSTM (Normal)             | 0/11 | 0/11 | 0/11     | 3/11     | 0/11     |
| DCC                       | 0/11 | 0/11 | 0/11     | 0/11     | 0/11     |
| ADCC                      | 0/11 | 0/11 | 0/11     | 0/11     | 0/11     |
| LSTM (Student-t)          | 0/11 | 0/11 | 0/11     | 0/11     | 0/11     |

## table4_dm_benchmark

| Model                     | Mean loss diff. | Preferred (seeds) | p<0.05 (seeds) | Mean p-value |
| ------------------------- | --------------- | ----------------- | -------------- | ------------ |
| Neural BEKK (asym. diag.) | -0.0476         | 11/11             | 3/11           | 0.2833       |
| Neural BEKK (mix)         | -0.0463         | 11/11             | 5/11           | 0.1157       |
| Neural BEKK (scalar)      | -0.0430         | 11/11             | 0/11           | 0.3030       |
| GRU (Normal)              | -0.0234         | 10/11             | 0/11           | 0.4881       |
| Neural BEKK (vector)      | -0.0159         | 9/11              | 0/11           | 0.4707       |
| GRU (Student-t)           | -0.0047         | 7/11              | 0/11           | 0.7008       |
| BEKK (asymmetric)         | 0.0028          | 0/11              | 0/11           | 0.7057       |
| LSTM (Normal)             | 0.0077          | 3/11              | 0/11           | 0.7304       |
| DCC                       | 0.0309          | 0/11              | 0/11           | 0.1148       |
| ADCC                      | 0.0309          | 0/11              | 0/11           | 0.1148       |
| LSTM (Student-t)          | 0.0317          | 0/11              | 0/11           | 0.6376       |
