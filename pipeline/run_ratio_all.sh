#!/bin/bash
# Runs every method (CAFNet, BN fine-tuning, AdaBN, Tent, CoDA, AdaptNet) for the four aspect ratios, then the additional analyses.
# Requires: Python 3.7 with TensorFlow 2.x, numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$R/pipeline/setup_runs.sh"
for ar in 1.22 1.06 1.10 1.16; do
  cd "$R/pipeline/runs/R_$ar"
  python "$R/pipeline/run_ratio.py" $ar >> run.log 2>&1             # main comparison (held-out 25)
  python "$R/pipeline/coda_ema_only.py" $ar >> run.log 2>&1         # CoDA without its gradient step
  python "$R/pipeline/extra_analyses.py" $ar A B C >> run.log 2>&1  # statistics-weight sweep, all-500 scoring, Tent/CoDA sensitivity
  python "$R/pipeline/extra_ema500.py" $ar >> run.log 2>&1          # statistics-only CoDA scored on all 500
done
cd "$R/pipeline"
python collect_ratio.py && python collect_extra.py
