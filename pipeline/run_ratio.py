"""Rerun every method for one aspect ratio with the corrected scripts and evaluate on the TRUE held-out 25 samples.

Usage:  python run_ar.py 1.06
Run folder: Response_Work/runs/AR_<ar>/ (data copies + outputs). Original project files are never written.
Corrections relative to the old workflow:
  * adaptation pool = x_train_20_2d[:475]  (was x_test_20_2d[:475] == the 25 held-out samples)
  * evaluation set = selected_indices[test_indices]  (the held-out samples; the old s_range=test_indices
    pointed at different samples, 24/25 of which were in the training pool)
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ORIG = os.environ.get("CODA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
HERE = os.path.dirname(os.path.abspath(__file__))
AR = sys.argv[1]
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "20"))   # Table 5: 20 epochs
RUN = os.path.join(HERE, "runs", "R_" + AR)
os.chdir(RUN)
sys.path.insert(0, ORIG)
sys.path.insert(0, HERE)
os.makedirs("results", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import builtins
from keras import backend as _b
builtins.backend = _b          # CODA_Github.py used `backend` before importing it (interactive-session leftover)

T0 = time.time()
def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)

import CODA_Github as C        # executes the (seeded) split with the corrected adaptation pool
import AdABN as A
import TenT as T
import helper_20_1 as H
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update(H.custom_objects)
import mask_utils
from scipy.io import loadmat as _lm
DS_MASK = _lm("samples_500.mat", squeeze_me=True)["mask"].astype(bool)
mask_utils.set_mask(H, DS_MASK)      # evaluate inside this dataset's own elliptical mask
log("evaluation mask pixels:", int(DS_MASK.sum()), "axis ratio of dataset:", AR)
# clone_model() rebuilds the Lambda mask layer without custom objects; its function needs multiply_constant as a global
import sys as _sys
_lam = _sys.modules[tf.keras.layers.Lambda.__module__]
for _k, _v in H.custom_objects.items():
    setattr(_lam, _k, _v)

assert np.array_equal(C.x_test_20_2d, A.x_test_20_2d) and np.array_equal(C.x_train_20_2d, T.x_train_20_2d)
perm = C.selected_indices_20_2d
HELD = perm[C.test_indices_20_2d]                 # original indices of the held-out samples
assert len(HELD) == 25 and len(C.x_train_20_2d) == 475
assert np.allclose(H.PHI_meas_16x15["training_20"][HELD], C.x_test_20_2d)   # x_test really is those samples
np.save("results/gt_held.npy", H.MU["training_20"][HELD])
np.savez("results/split.npz", perm=perm, train_pos=C.train_indices_20_2d, test_pos=C.test_indices_20_2d, held=HELD)
log("AR", AR, "held-out original indices:", sorted(HELD.tolist()))



def metrics_frame(MU_est, key, s_range):
    return H.ground_truth_vs_estimated_direct(key, MU_est, s_range=s_range)[0]


def evaluate_holdout(name, preds):
    out = "results/%s_%s.xlsx" % (name, AR)
    MU = np.zeros((500, 2, 64, 64), np.float32)
    MU[HELD] = np.concatenate(list(preds)[:2], axis=1)
    df = metrics_frame(MU, "training_20", HELD)
    df.to_excel(out)
    np.save("results/pred_%s.npy" % name, MU[HELD])
    log(name, "held-out: ssim_a %.3f ssim_s %.3f csd_a %.3f csd_s %.3f" % (
        np.nanmean(df["ssim"]["mu_a"]), np.nanmean(df["ssim"]["mu_sp"]),
        np.nanmean(df["R_csd"]["mu_a"]), np.nanmean(df["R_csd"]["mu_sp"])))


def evaluate_exp(name, model_path_or_model):
    out = "results/exp_%s_%s.xlsx" % (name, AR)
    m = load_model(model_path_or_model, compile=False, custom_objects=H.custom_objects) \
        if isinstance(model_path_or_model, str) else model_path_or_model
    p = m.predict(PHI_EXP, batch_size=4, verbose=0)
    MU = np.concatenate(list(p)[:2], axis=1)
    df = metrics_frame(MU, "calibrated", range(len(MU)))
    df.to_excel(out)
    np.save("results/exppred_%s.npy" % name, MU)
    log(name, "exp sample 11: ssim_a %.3f csd_a %.3f csd_s %.3f" % (
        df["ssim"]["mu_a"].iloc[11], df["R_csd"]["mu_a"].iloc[11], df["R_csd"]["mu_sp"].iloc[11]))


def done(name):
    return os.path.exists("results/%s_%s.xlsx" % (name, AR))


# ---------------------------------------------------------------- CAFNet (no adaptation)
if not done("CAFNet"):
    m = load_model("CAFNet.h5", compile=False, custom_objects=H.custom_objects)
    evaluate_holdout("CAFNet", m.predict(C.x_test_20_2d, batch_size=8, verbose=0))

# ---------------------------------------------------------------- BN-only fine-tuning (labeled, 475 samples)
if not done("FineTuned"):
    t = time.time()
    m = load_model("CAFNet.h5", compile=False, custom_objects=H.custom_objects)
    for l in m.layers:
        l.trainable = isinstance(l, BatchNormalization)
    m.compile(loss="mse", optimizer=Adam(learning_rate=1e-3, beta_1=0.5))
    m.fit(C.x_train_20_2d, C.y_train_20_2d, batch_size=8, epochs=FT_EPOCHS, verbose=0)
    m.save("FineTuned_model.h5")
    log("fine-tuning (%d epochs) took %.0fs" % (FT_EPOCHS, time.time() - t))
    evaluate_holdout("FineTuned", m.predict(C.x_test_20_2d, batch_size=8, verbose=0))

# ---------------------------------------------------------------- AdaBN (corrected AdABN.py)
if not done("AdaBN"):
    t = time.time()
    preds = A.run_adabn_full_analysis()
    log("AdaBN took %.0fs" % (time.time() - t))
    os.replace("CAFNet_AdaBN_1.06.h5", "AdaBN_model.h5")
    evaluate_holdout("AdaBN", preds)

# ---------------------------------------------------------------- Tent (corrected TenT.py)
if not done("Tent"):
    t = time.time()
    preds = T.run_tent_adaptation_full_analysis()
    log("Tent took %.0fs" % (time.time() - t))
    os.replace("adapted_CAFNet_TENT_analysis.h5", "Tent_model.h5")
    evaluate_holdout("Tent", preds)

# ---------------------------------------------------------------- CoDA (corrected CODA_Github.py, sigma = 0.05)
if not done("CoDA"):
    t = time.time()
    res = C.run_coda_full_experiment()
    log("CoDA took %.0fs" % (time.time() - t))
    os.replace("adapted_CAFNet_CoDA_complete_analysis.h5", "CoDA_model.h5")
    evaluate_holdout("CoDA", res["predictions"])
    json.dump({k: (v if not isinstance(v, tuple) else list(v)) for k, v in res.items() if k != "predictions"},
              open("results/coda_diagnostics_%s.json" % AR, "w"), default=float, indent=1)

# ---------------------------------------------------------------- AdaptNet (untitled0.py implementation, per sample)
if not os.path.exists("results/AdaptNet_%s.xlsx" % AR):
    import adaptnet_impl as AN
    t = time.time()
    adaptnet, base_frozen, blocks = AN.build_adaptnet("CAFNet.h5")
    p1, p2 = [], []
    for i in range(len(C.x_test_20_2d)):
        y = tf.convert_to_tensor(C.x_test_20_2d[i:i + 1], dtype=tf.float32)
        a, b, _ = AN.adapt_and_predict(adaptnet, base_frozen, blocks, y)
        p1.append(a[0]); p2.append(b[0])
    log("AdaptNet took %.0fs" % (time.time() - t))
    evaluate_holdout("AdaptNet", [np.array(p1), np.array(p2)])

log("ALL DONE", AR)
