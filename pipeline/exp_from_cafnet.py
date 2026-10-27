"""Experimental phantom, every method applied directly to the source model CAFNet (no stacking, original CoDA).
Data: mask1.22/landseed_processed__new_data.mat only (one phantom: calibrated measurement PHI_CLB_meas and ground-truth grids).
(The 12-sample landseed file in the run folder is only needed so that helper_20_1.py can be imported; nothing is read from it.)

A single measurement cannot give batch statistics (a dense-layer BN over a batch of one has zero variance), so the label-free methods
adapt on a pool of K jittered copies of the measurement (Gaussian jitter, std POOL_SIGMA); the consistency noise of CoDA is the usual 0.05.
  CAFNet      : source model, no adaptation
  Fine-Tuned  : BN-only fine-tuned on the labeled ratio_1.22 simulation pool (not adapted on the experiment; labels are not available there)
  AdaBN       : statistics replaced by those of the pool (blend 1.0)
  Tent        : entropy of spatial softmax, Adam 1e-5, 50 iterations (optimizer steps), batch 8 (TenT.py logic)
  CoDA        : original CoDA (EMA statistics + consistency loss on all BN gamma/beta incl. the output layers), Adam 1e-5, 50 iterations (optimizer steps), batch 8, sigma 0.05
  AdaptNet    : per-sample test-time adaptation of the measurement (its native mode)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ORIG = os.environ.get("CODA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))); HERE = os.path.dirname(os.path.abspath(__file__))
EXPFILE = os.path.join(ORIG, "mask1.22", "landseed_processed__new_data.mat")
K = int(os.environ.get("POOL_K", "64"))
os.chdir(os.path.join(HERE, "runs", "EXP")); sys.path.insert(0, ORIG); sys.path.insert(0, HERE)
os.makedirs("results", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import builtins
from keras import backend as _b
builtins.backend = _b
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
import AdABN as A
import TenT as TT
import helper_20_1 as H
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import mask_utils, extra_metrics as EM
get_custom_objects().update(H.custom_objects)
_lam = sys.modules[tf.keras.layers.Lambda.__module__]
for k_, v_ in H.custom_objects.items():
    setattr(_lam, k_, v_)
DS_MASK = loadmat("samples_500.mat", squeeze_me=True)["mask"].astype(bool)      # AR 1.22 ellipse
mask_utils.set_mask(H, DS_MASK)

# ---------------- data: this file only
d = loadmat(EXPFILE, squeeze_me=True)
def prep(phi):
    lg = np.log(np.asarray(phi).astype(complex).reshape(-1))
    ph = H.unwrap_sample(np.imag(lg).copy(), 16)
    return np.concatenate([lg.real, ph]).reshape(1, 2, 16, 15).astype(np.float32)
Y = prep(d["PHI_CLB_meas"])
GT = np.stack([np.asarray(d["Grid_MUA"], float), np.asarray(d["Grid_MUSP"], float)])           # (2,64,64)
# helper-compatible entries for this single phantom (inclusions sorted by radius, padded with zeros)
r_, roc_, th_, ca_, cs_ = [np.atleast_1d(np.asarray(d[k], float)) for k in ("r", "roc", "theta", "ca", "csp")]
order = np.argsort(-r_)
def pad(x):
    out = np.zeros((1, H.MAX_INC)); out[0, :len(x)] = x[order]; return out
KEY = "calibrated"
H.d[KEY] = np.array([[float(d["d"])]]); H.r_data[KEY] = pad(r_); H.roc_data[KEY] = pad(roc_); H.theta_data[KEY] = pad(th_)
H.ca_data[KEY] = pad(ca_); H.csp_data[KEY] = pad(cs_); H.MU[KEY] = GT[None]
log("measurement:", d["freq"] / 1e6, "MHz, d =", d["d"], "mm, inclusions r", r_.tolist(), "ca", ca_.tolist(), "csp", cs_.tolist())

COLS = [("ssim", "mu_a"), ("ssim", "mu_sp"), ("R_csd", "mu_a"), ("R_csd", "mu_sp")]
NAMES = ["ssim_a", "ssim_s", "csd_a", "csd_s"]
def evaluate(pred):
    MU = np.concatenate([np.asarray(pred[0]), np.asarray(pred[1])], axis=0).reshape(1, 2, 64, 64).astype(np.float32)
    df = H.ground_truth_vs_estimated_direct(KEY, MU, s_range=[0])[0]
    r = {n: float(df[c].iloc[0]) for n, c in zip(NAMES, COLS)}
    for k, p in ((0, "a"), (1, "s")):
        m = EM.one(GT[k], MU[0, k], DS_MASK)
        r["rmse_" + p] = m["rmse"] * (1e3 if p == "a" else 1.0); r["nrmse_" + p] = m["nrmse"]; r["cnr_" + p] = m["cnr"]; r["crc_" + p] = m["crc"]; r["loc_px_" + p] = m["loc_px"]
    return r, MU[0]
def load(p): return load_model(p, compile=False, custom_objects=H.custom_objects)
def predict(m): return [p[0] for p in m.predict(Y, batch_size=1, verbose=0)]

SIG = 0.05
ITER = 50
def coda_original(model, pool, epochs=None, batch=8):
    for l in model.layers: l.trainable = isinstance(l, BatchNormalization)
    opt = Adam(learning_rate=1e-5)
    @tf.function
    def step(x):
        noise = tf.random.normal(tf.shape(x), stddev=SIG)
        with tf.GradientTape() as tape:
            p1 = model(x, training=True); p2 = model(x + noise, training=True)
            loss = tf.add_n([tf.reduce_mean(tf.square(a - b)) for a, b in zip(p1, p2)])
        g = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients([(gg, v) for gg, v in zip(g, model.trainable_weights) if gg is not None])
        return loss
    tf.random.set_seed(19)
    batches = list(tf.data.Dataset.from_tensor_slices(pool).batch(batch))
    for it in range(ITER):                       # ITER optimizer steps (Table 5: 50 iterations), cycling through the batches of the pool
        step(batches[it % len(batches)])
    return model
def adabn(model, pool):
    ds = tf.data.Dataset.from_tensor_slices(pool).batch(8)
    A.apply_adabn(model, A.compute_empirical_bn_statistics(model, ds), momentum_blend=1.0)
    return model
def tent(model, pool, epochs=None, batch=8):
    for l in model.layers: l.trainable = isinstance(l, BatchNormalization)
    opt = Adam(learning_rate=1e-5)
    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            loss = TT.entropy_loss(model(x, training=True))          # TenT.py loss
        g = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients([(gg, v) for gg, v in zip(g, model.trainable_weights) if gg is not None])
        return loss
    batches = list(tf.data.Dataset.from_tensor_slices(pool).batch(batch))
    for it in range(ITER):
        step(batches[it % len(batches)])
    return model

rows = []
def add(method, setting, pred):
    r, mu = evaluate(pred); r.update(method=method, setting=setting); rows.append(r)
    np.save("results/pred_%s__%s.npy" % (method.replace(" ", ""), setting.replace(" ", "")), mu)
    log("%-11s %-22s ssim_a %.3f ssim_s %.3f csd_a %7.3f csd_s %7.3f rmse_a %.2f cnr %.2f" % (method, setting, r["ssim_a"], r["ssim_s"], r["csd_a"], r["csd_s"], r["rmse_a"], r["cnr_a"]))

add("CAFNet", "no adaptation", predict(load("CAFNet.h5")))
add("Fine-Tuned", "AR 1.22 labeled model", predict(load("FineTuned_model.h5")))
for sig in (0.05, 0.02, 0.10):
    rng = np.random.RandomState(19)
    pool = (Y + sig * rng.randn(K, 2, 16, 15)).astype(np.float32)
    tag = "pool K=%d, jitter %.2f" % (K, sig)
    add("AdaBN", tag, predict(adabn(load("CAFNet.h5"), pool)))
    add("Tent", tag, predict(tent(load("CAFNet.h5"), pool)))
    add("CoDA", tag, predict(coda_original(load("CAFNet.h5"), pool)))
import adaptnet_impl as AN
adaptnet, base_frozen, blocks = AN.build_adaptnet("CAFNet.h5")
a, b, _ = AN.adapt_and_predict(adaptnet, base_frozen, blocks, tf.convert_to_tensor(Y, dtype=tf.float32))
add("AdaptNet", "per-sample (native)", [a[0], b[0]])
pd.DataFrame(rows).to_csv("results/exp_from_cafnet.csv", index=False)
log("DONE")
