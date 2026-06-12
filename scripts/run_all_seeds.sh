#!/usr/bin/env bash
set -euo pipefail

cd /Users/bayesed/Desktop/Studium/Masterthesis/Code/neural_bekk

for seed in 42 0 1 2 3 4 5 6 7 8 9; do
  echo "[$(date '+%H:%M:%S')] Running seed ${seed}"

  MODEL_SEED="$seed" \
    jupyter nbconvert \
      --to notebook \
      --execute run_models.ipynb \
      --ExecutePreprocessor.timeout=-1 \
      --stdout > /dev/null

  echo "[$(date '+%H:%M:%S')] Finished seed ${seed}"
done

echo "[$(date '+%H:%M:%S')] All seeds finished"
