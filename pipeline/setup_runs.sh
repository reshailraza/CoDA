#!/bin/bash
# Creates pipeline/runs/R_<AR>/ (one working folder per aspect ratio) from the repository files.
# Usage (from the repository root or from pipeline/):  bash pipeline/setup_runs.sh
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$R/pipeline"
for ar in 1.06 1.10 1.16 1.22; do
  d="$W/runs/R_$ar"; mkdir -p "$d/results"
  cp "$R/datasets/ratio_$ar/samples_500.mat" "$R/datasets/ratio_$ar/param_samples_500.mat" "$d/"
  cp "$R/iterative_results_5001.mat" "$R/mask.mat" "$R/exp_data_rev_16.mat" "$R/landseed_processed__new_data.mat" "$R/CAFNet.h5" "$d/"
done
# experimental run folder (used by exp_from_cafnet.py)
mkdir -p "$W/runs/EXP/results"
cp "$W/runs/R_1.22/"{samples_500.mat,param_samples_500.mat,iterative_results_5001.mat,mask.mat,exp_data_rev_16.mat,CAFNet.h5,landseed_processed__new_data.mat} "$W/runs/EXP/"
echo "run folders created under $W/runs"
