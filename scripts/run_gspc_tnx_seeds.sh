#!/usr/bin/env bash
# Robustness study driver: 2-asset (S&P 500 + 10Y yield) dataset.
# For each seed: fit the R BEKK/DCC models, then run the neural pipeline.
# Run this in your OWN terminal (not via the agent) so it is not killed:
#     bash scripts/run_gspc_tnx_seeds.sh 2>&1 | tee /tmp/gspc_sweep.log
#
# Parameterized via env vars (defaults = full-history 1962 cohort). For the
# modern post-Volcker 1990 cohort (50/15/35), run:
#   GSPC_DATA_ID=gspc_tnx_1990 GSPC_START=1990-01-01 \
#   GSPC_TRAIN_SIZE=0.50 GSPC_VAL_SIZE=0.15 \
#   GSPC_SOURCE_CLOSE=data/close_gspc_tnx_1962.csv \
#   bash scripts/run_gspc_tnx_seeds.sh 2>&1 | tee /tmp/gspc_sweep_1990.log
set -euo pipefail

cd /Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk

# Config (env-overridable). These are exported so prep + neural pick them up.
export GSPC_DATA_ID="${GSPC_DATA_ID:-gspc_tnx_1962}"
export GSPC_START="${GSPC_START:-1900-01-01}"
export GSPC_TRAIN_SIZE="${GSPC_TRAIN_SIZE:-0.60}"
export GSPC_VAL_SIZE="${GSPC_VAL_SIZE:-0.15}"
export GSPC_SOURCE_CLOSE="${GSPC_SOURCE_CLOSE:-}"
SEEDS=(${GSPC_SEEDS:-42 0 1 2 3})

DATA_ID="$GSPC_DATA_ID"
echo "### dataset=${DATA_ID}  split=${GSPC_TRAIN_SIZE}/${GSPC_VAL_SIZE}  seeds=${SEEDS[*]}"

# Data must exist; prepare if missing (uses the env vars above).
if [ ! -f "data/train_df_${DATA_ID}.csv" ]; then
  echo ">>> preparing data ${DATA_ID}"
  python scripts/prepare_gspc_tnx_data.py
fi

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "############################################################"
  echo "### ${DATA_ID}  SEED ${seed}  ($(date '+%H:%M:%S'))"
  echo "############################################################"

  run_id="asof_${DATA_ID}_seed${seed}"

  # R models (skip if already fitted for this seed)
  if [ ! -f "r/output/dcc_forecasts_adcc_${run_id}.csv" ]; then
    echo ">>> [R] BEKK sym/asym  (seed ${seed})"
    Rscript r/bekks.R --project-dir "$(pwd)" --data-id "${DATA_ID}" --run-id "${run_id}" --seed "${seed}" --max-iter 200
    echo ">>> [R] DCC / aDCC     (seed ${seed})"
    Rscript r/dcc.R   --project-dir "$(pwd)" --data-id "${DATA_ID}" --run-id "${run_id}" --seed "${seed}"
  else
    echo ">>> [R] outputs for seed ${seed} already present - skipping R"
  fi

  # Neural pipeline (8 models) + FHS + backtests -> all_backtest_results.json
  # -u = unbuffered so per-epoch prints stream live through the tee pipe.
  echo ">>> [NEURAL] seed ${seed}"
  MODEL_SEED="${seed}" python -u run_models_gspc_tnx.py

  echo ">>> SEED ${seed} COMPLETE  ($(date '+%H:%M:%S'))"
done

echo ""
echo "All seeds finished. Aggregate with:"
echo "  python evaluate_runs.py --runs-dir results/runs --out results/evaluation_gspc_tnx --no-overleaf"
echo "(look at the asof_${DATA_ID} group in that output folder)"
