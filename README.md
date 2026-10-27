# Unsupervised transfer learning for deep diffuse optical imaging through batch normalization

Code, datasets and results for *Unsupervised Transfer Learning for Deep Diffuse Optical Imaging through Batch Normalization: From Circular to Elliptical Phantoms* (Iftikhar, Hsu, Pan).

A channel-attention fusion network (CAFNet, 42 batch-normalization layers) is trained on circular phantoms and transferred to elliptical phantoms **without any target label** by recalibrating its BN layers on unlabeled target measurements. Four label-free methods are tested on the same footing, with a supervised BN fine-tuning as a labeled reference:

| Method | What is updated | Labels |
|---|---|---|
| AdaBN | BN running statistics (full replacement) | no |
| Tent | BN gamma/beta (entropy) + running statistics | no |
| CoDA (earlier conference method, here one of the tested methods) | BN gamma/beta (consistency loss) + running statistics (EMA) | no |
| AdaptNet | per-sample residual blocks, CAFNet frozen | no |
| BN fine-tuning (reference) | BN gamma/beta + running statistics | yes |

Training protocol of every method: Table 4 of the paper (source CAFNet: Adam 1e-3, batch 64, 100 epochs; BN-FT: Adam 1e-3, batch 8, 20 epochs; AdaBN: 20 passes over the 475-sample pool; Tent and CoDA: Adam 1e-5, batch 8, 50 iterations; AdaptNet: 200 steps per sample).

## Layout

| Path | Content |
|---|---|
| `AdABN.py`, `TenT.py`, `CODA_Github.py` | corrected method implementations (adaptation pool = `x_train_20_2d[:475]`; 50 optimizer steps; trainable BN during statistics passes) |
| `helper_20_1.py`, `custom_*.py`, `plotting.py` | data handling, metrics (SSIM, CSD), custom layers |
| `CAFNet.h5` | source-trained network (circular phantoms) |
| `datasets/ratio_1.06 ... ratio_1.22` | **new target datasets**: 500 elliptical samples each, identical sample parameters, only the aspect ratio differs |
| `AR_1.06, AR_1.15, AR_1.21, AR_1.22` | legacy datasets of the conference paper (frequency, diameter and contrast differ between them) |
| `landseed_processed__new_data.mat`, `exp_data_rev_16.mat`, ... | experimental phantom measurement |
| `pipeline/` | run scripts, additional analyses, metric code, aggregation scripts |
| `results/` | all results of the paper: per aspect ratio metric/parameter box plots, qualitative sample, per-sample metric tables, summary tables (`results/Summary_tables`), experimental phantom |

## Reproduce

Python 3.7, TensorFlow 2.x, numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl.

```bash
bash pipeline/run_ratio_all.sh     # builds pipeline/runs/R_<AR>, runs all methods and the additional analyses, aggregates
python pipeline/exp_from_cafnet.py # experimental phantom: every method applied directly to CAFNet
```

`CODA_ROOT` can be set to the directory that contains `CODA_Github.py` and `helper_20_1.py` (default: repository root). Splits are seeded (seed 19): 475 unlabeled adaptation samples and 25 held-out samples per aspect ratio, the same indices for all aspect ratios.

## Results in brief (mean over the four aspect ratios, mu_a)

| Method | SSIM | RMSE (1e-3 mm^-1) |
|---|---|---|
| CAFNet (no adaptation) | 0.49 | 2.57 |
| Tent | 0.79 | 2.17 |
| CoDA | 0.82 | 2.20 |
| AdaBN | 0.83 | 2.31 |
| BN fine-tuning (labeled) | 0.86 | 1.88 |

Contrast recovery (CSD) is not improved by any label-free method. The extent of the statistics update controls a trade-off (`results/Summary_tables/sweepA_statistics_weight.csv`); gradient-based label-free losses add little to it and are unstable at larger budgets (`sweepC_budget_lr.csv`). See `results/Summary_tables/` for every table of the paper.

## Notes

* The legacy scripts of the conference paper adapted on the 25 held-out samples and scored a different set; the files here are the corrected versions (see the git history for the originals).
* Adapted models (about 100 MB each) are not stored; they are recreated by the pipeline.
* License: see `LICENSE`.
