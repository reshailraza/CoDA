import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from scipy.stats import pearsonr
from helper_20_1 import *
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
smooth = 1.
dropout_rate = 0.5
act = "relu"
###############################################################################

def fn_mask(x):
    from scipy.io import loadmat
    mask = loadmat('mask.mat')['mask'].astype(np.float)
    return multiply_constant(x, mask)


##########################################################################################################
from keras import backend
backend.set_image_data_format('channels_first')
print(backend.image_data_format())

lr = 0.001
beta_1 = 0.5


key_3_2d = "training_20"

x0_20_2d = PHI_meas_16x15[key_3_2d]
y0_20_2d = MU[key_3_2d]
total_num_20_2d = x0_20_2d.shape[0]

total_selected_samples = 500

# Generate a list of indices and shuffle them
indices_20_2d = np.arange(total_num_20_2d)
np.random.shuffle(indices_20_2d)

selected_indices_20_2d = indices_20_2d[:total_selected_samples]

x_selected_20_2d = x0_20_2d[selected_indices_20_2d]
y_selected_20_2d = y0_20_2d[selected_indices_20_2d]

train_prop = 0.95
val_prop = 0.0
test_prop = 0.05
train_num_20_2d = int(train_prop * total_selected_samples)  # 80% of 10k
val_num_20_2d = int(val_prop * total_selected_samples)  # 15% of 10k
test_num_20_2d = total_selected_samples - train_num_20_2d - val_num_20_2d  # 5% of 10k
indices_selected_20_2d = np.arange(total_selected_samples)
np.random.shuffle(indices_selected_20_2d)

train_indices_20_2d = indices_selected_20_2d[:train_num_20_2d]
val_indices_20_2d = indices_selected_20_2d[train_num_20_2d:train_num_20_2d + val_num_20_2d]
test_indices_20_2d = indices_selected_20_2d[train_num_20_2d + val_num_20_2d:]
print(test_indices_20_2d)
x_train_20_2d = x_selected_20_2d[train_indices_20_2d]
y_train_20_2d = y_selected_20_2d[train_indices_20_2d]




x_val_20_2d = x_selected_20_2d[val_indices_20_2d]
y_val_20_2d = y_selected_20_2d[val_indices_20_2d]

x_test_20_2d = x_selected_20_2d[test_indices_20_2d]
y_test_20_2d = y_selected_20_2d[test_indices_20_2d]

# Prepare the target arrays (assuming the target is 2D, with two outputs)
y_train_20_2d = [y_train_20_2d[:, [0]], y_train_20_2d[:, [1]]]
y_val_20_2d = [y_val_20_2d[:, [0]], y_val_20_2d[:, [1]]]
y_test_20_2d = [y_test_20_2d[:, [0]], y_test_20_2d[:, [1]]]


PHI_meas_in_20_2d = PHI_meas_16x15[key_3_2d]
input_shape_20_2d = PHI_meas_in_20_2d.shape[1:]
inputs_20_2d = Input(input_shape_20_2d)
# input_ = Reshape((1,) + input_shape)(inputs[0])
inputs_20_2d.shape
# ============================================================
# BN STATISTICS FUNCTIONS
# ============================================================
def compute_empirical_bn_statistics(model, dataset):
    """
    Compute empirical BN statistics by running forward passes on the target
    dataset in training mode (unsupervised, no gradient updates). Returns a
    list of (mean, variance) tuples per BN layer. The model's original
    moving statistics are restored before returning, so this call has no
    side effect on `model`.
    """
    bn_layers = [l for l in model.layers if isinstance(l, BatchNormalization)]
    original_means = [l.moving_mean.numpy().copy() for l in bn_layers]
    original_vars = [l.moving_variance.numpy().copy() for l in bn_layers]
    for l in bn_layers:
        l.moving_mean.assign(tf.zeros_like(l.moving_mean))
        l.moving_variance.assign(tf.ones_like(l.moving_variance))
    for batch in dataset:
        _ = model(batch, training=True)
    empirical_stats = [
        (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
        for l in bn_layers
    ]
    for l, m, v in zip(bn_layers, original_means, original_vars):
        l.moving_mean.assign(m)
        l.moving_variance.assign(v)
    return empirical_stats

def compute_alignment(stats_A, stats_B):
    """
    Scalar total misalignment: A = sum_l ( ||mu_A - mu_B||^2 + ||sigma2_A - sigma2_B||^2 ).
    This is the quantity reported in Table 8 (A_initial, A_final).
    """
    total = 0.0
    for (m_a, v_a), (m_b, v_b) in zip(stats_A, stats_B):
        total += np.linalg.norm(m_a - m_b) ** 2
        total += np.linalg.norm(v_a - v_b) ** 2
    return total

def compute_layer_wise_alignment(stats_A, stats_B):
    """
    Per-layer misalignment A_l, used for the layer-wise ablation figure
    and as the covariate in the correlation analysis.
    """
    return [
        np.linalg.norm(m_a - m_b) ** 2 + np.linalg.norm(v_a - v_b) ** 2
        for (m_a, v_a), (m_b, v_b) in zip(stats_A, stats_B)
    ]

# ============================================================
# BN COMPONENT SUBSTITUTION ABLATION (Table 9 / Analysis 2)
# ============================================================
def set_bn_statistics(model, target_stats):
    """Assign target-domain moving mean/variance onto a model's BN layers."""
    bn_layers = [l for l in model.layers if isinstance(l, BatchNormalization)]
    for layer, (m, v) in zip(bn_layers, target_stats):
        layer.moving_mean.assign(tf.convert_to_tensor(m, dtype=tf.float32))
        layer.moving_variance.assign(tf.convert_to_tensor(v, dtype=tf.float32))

def set_bn_affine(model, target_affine):
    """Assign target-domain gamma/beta onto a model's BN layers."""
    bn_layers = [l for l in model.layers if isinstance(l, BatchNormalization)]
    for layer, (g, b) in zip(bn_layers, target_affine):
        layer.gamma.assign(tf.convert_to_tensor(g, dtype=tf.float32))
        layer.beta.assign(tf.convert_to_tensor(b, dtype=tf.float32))

def compute_delta_stats(model, x, target_stats):
    """
    Delta_stats = E[ || f_theta'(x) - f_theta(x) ||^2 ]
    theta': BN moving statistics replaced with target-domain estimates,
            affine parameters (gamma, beta) frozen at source values.
    """
    pred_orig = model(x, training=False)
    model_variant = tf.keras.models.clone_model(model)
    model_variant.set_weights(model.get_weights())
    _ = model_variant(x[:1], training=False)
    set_bn_statistics(model_variant, target_stats)
    pred_variant = model_variant(x, training=False)
    if isinstance(pred_orig, list):
        diffs = [np.mean((p1.numpy() - p2.numpy()) ** 2) for p1, p2 in zip(pred_orig, pred_variant)]
        return float(np.mean(diffs))
    return float(np.mean((pred_orig.numpy() - pred_variant.numpy()) ** 2))

def compute_delta_affine(model, x, target_affine):
    """
    Delta_affine = E[ || f_theta''(x) - f_theta(x) ||^2 ]
    theta'': affine parameters (gamma, beta) replaced with target-domain
             estimates, BN moving statistics frozen at source values.
    """
    pred_orig = model(x, training=False)
    model_variant = tf.keras.models.clone_model(model)
    model_variant.set_weights(model.get_weights())
    _ = model_variant(x[:1], training=False)
    set_bn_affine(model_variant, target_affine)
    pred_variant = model_variant(x, training=False)
    if isinstance(pred_orig, list):
        diffs = [np.mean((p1.numpy() - p2.numpy()) ** 2) for p1, p2 in zip(pred_orig, pred_variant)]
        return float(np.mean(diffs))
    return float(np.mean((pred_orig.numpy() - pred_variant.numpy()) ** 2))

def obtain_target_affine_via_consistency(model, dataset, iterations=50, lr=1e-5, noise_std=0.02):
    """
    Estimate target-domain BN affine parameters (gamma^T, beta^T) using the
    consistency-driven objective alone, on a CLONE of `model`, while holding
    BN moving statistics fixed at their source values (layer momentum is set
    to 1.0 so forward passes in training mode do not update moving mean/var).
    Does not modify `model`. Returns a list of (gamma, beta) per BN layer,
    for use as the theta'' variant in the substitution ablation.
    """
    model_variant = tf.keras.models.clone_model(model)
    model_variant.set_weights(model.get_weights())
    first_batch = next(iter(dataset))
    _ = model_variant(first_batch[:1], training=False)

    bn_layers = [l for l in model_variant.layers if isinstance(l, BatchNormalization)]
    original_momentum = [l.momentum for l in bn_layers]
    for l in bn_layers:
        l.momentum = 1.0  # freeze moving statistics for this pass
        l.trainable = True
    for l in model_variant.layers:
        if not isinstance(l, BatchNormalization):
            l.trainable = False

    optimizer = Adam(learning_rate=lr)

    @tf.function
    def affine_step(x):
        noise = tf.random.normal(tf.shape(x), stddev=noise_std)
        with tf.GradientTape() as tape:
            p1 = model_variant(x, training=True)
            p2 = model_variant(x + noise, training=True)
            if isinstance(p1, list):
                loss = tf.add_n([tf.reduce_mean(tf.square(a - b)) for a, b in zip(p1, p2)])
            else:
                loss = tf.reduce_mean(tf.square(p1 - p2))
        grads = tape.gradient(loss, model_variant.trainable_weights)
        grads_vars = [(g, v) for g, v in zip(grads, model_variant.trainable_weights) if g is not None]
        optimizer.apply_gradients(grads_vars)
        return loss

    for _ in range(iterations):
        for batch in dataset:
            affine_step(batch)

    target_affine = [(l.gamma.numpy().copy(), l.beta.numpy().copy()) for l in bn_layers]

    for l, m in zip(bn_layers, original_momentum):
        l.momentum = m

    return target_affine

# ============================================================
# CONSISTENCY
# ============================================================
def compute_consistency_per_sample(model, x):
    noise = tf.random.normal(tf.shape(x), stddev=0.02)
    pred_clean = model(x, training=False)
    pred_noisy = model(x + noise, training=False)
    if isinstance(pred_clean, list):
        losses = []
        for pc, pn in zip(pred_clean, pred_noisy):
            l = tf.reduce_mean((pc - pn) ** 2, axis=list(range(1, len(pc.shape))))
            losses.append(l.numpy())
        losses = np.mean(losses, axis=0)
    else:
        losses = tf.reduce_mean((pred_clean - pred_noisy) ** 2,
                                axis=list(range(1, len(pred_clean.shape)))).numpy()
    return losses.flatten()

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def plot_bn_alignment(A_initial_layerwise, A_final_layerwise):
    plt.figure()
    plt.boxplot([
        np.array(A_initial_layerwise).flatten(),
        np.array(A_final_layerwise).flatten()],
        labels=["Before CoDA", "After CoDA"])
    plt.ylabel("BN Misalignment")
    plt.title("BN Alignment Improvement")
    plt.grid(True)
    plt.show()

def plot_bn_alignment_bar(A_initial_layerwise, A_final_layerwise):
    plt.figure(figsize=(10, 4))
    layers = np.arange(len(A_initial_layerwise))
    plt.bar(layers, A_initial_layerwise, alpha=0.5, label="Before CoDA")
    plt.bar(layers, A_final_layerwise, alpha=0.5, label="After CoDA")
    plt.xlabel("Layer Index")
    plt.ylabel("Misalignment $A_\\ell$")
    plt.title("Layer-wise BN Misalignment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_drift(gamma_drifts, beta_drifts, mean_drifts, var_drifts):
    plt.figure()
    plt.boxplot([
        np.array(gamma_drifts).flatten(),
        np.array(beta_drifts).flatten(),
        np.array(mean_drifts).flatten(),
        np.array(var_drifts).flatten()
    ],
        labels=["Δγ", "Δβ", "Δμ", "Δσ²"])
    plt.yscale("log")
    plt.ylabel("Drift (log scale)")
    plt.title("BN Parameter vs Statistics Drift")
    plt.grid(True)
    plt.show()

def plot_bn_distribution_shift(source_stats, target_stats, adapted_stats, layer_idx=0):
    """
    Visualizes BN statistics alignment using ALL channels of one layer.
    Plots mean Gaussian across channels with a shaded (+/- std) band, for
    source, target, and post-CoDA statistics.
    """
    m_s, v_s = source_stats[layer_idx]
    m_t, v_t = target_stats[layer_idx]
    m_a, v_a = adapted_stats[layer_idx]
    m_s, v_s = m_s.flatten(), v_s.flatten()
    m_t, v_t = m_t.flatten(), v_t.flatten()
    m_a, v_a = m_a.flatten(), v_a.flatten()
    x = np.linspace(-3, 3, 300)

    def gaussian(x, m, v):
        return 1 / np.sqrt(2 * np.pi * v + 1e-8) * np.exp(-(x - m) ** 2 / (2 * v + 1e-8))

    def compute_distribution(m, v):
        g_all = np.array([gaussian(x, mi, vi) for mi, vi in zip(m, v)])
        return g_all.mean(axis=0), g_all.std(axis=0)

    g_s_mean, g_s_std = compute_distribution(m_s, v_s)
    g_t_mean, g_t_std = compute_distribution(m_t, v_t)
    g_a_mean, g_a_std = compute_distribution(m_a, v_a)

    plt.figure(figsize=(6, 4))
    plt.plot(x, g_s_mean, linestyle='-', linewidth=2, color='black', label="Source")
    plt.fill_between(x, g_s_mean - g_s_std, g_s_mean + g_s_std, color='black', alpha=0.1)
    plt.plot(x, g_t_mean, linestyle='--', linewidth=2, color='black', label="Target")
    plt.fill_between(x, g_t_mean - g_t_std, g_t_mean + g_t_std, color='black', alpha=0.2)
    plt.plot(x, g_a_mean, linestyle='-.', linewidth=2, color='black', label="After CoDA")
    plt.fill_between(x, g_a_mean - g_a_std, g_a_mean + g_a_std, color='black', alpha=0.3)
    plt.title("BN Statistics Alignment (All Channels)")
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_consistency(before, after):
    before = np.array(before).flatten()
    after = np.array(after).flatten()
    plt.figure()
    plt.boxplot([before, after], labels=["Before CoDA", "After CoDA"])
    plt.ylabel("Consistency Loss")
    plt.title("Prediction Stability")
    plt.grid(True)
    plt.show()

def plot_correlation(A_initial_layerwise, mean_drifts):
    plt.figure()
    plt.scatter(A_initial_layerwise, mean_drifts)
    plt.xlabel("Initial Misalignment")
    plt.ylabel("Mean Drift")
    plt.title("Targeted BN Adaptation")
    plt.grid(True)
    plt.show()

def plot_layer_depth_pattern(A_initial_layerwise, A_final_layerwise, mean_drifts):
    """Three-panel figure: initial misalignment, per-layer improvement, drift magnitude."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    layer_indices = np.arange(1, len(A_initial_layerwise) + 1)

    axes[0].bar(layer_indices, A_initial_layerwise, alpha=0.7)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Initial Misalignment A_ℓ')
    axes[0].set_title('(a) Initial Problem Distribution')
    axes[0].axhline(y=np.median(A_initial_layerwise), linestyle='--', label='Median')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    improvement = [(a_i - a_f) / a_i * 100 if a_i > 0 else 0
                   for a_i, a_f in zip(A_initial_layerwise, A_final_layerwise)]
    axes[1].bar(layer_indices, improvement, alpha=0.7)
    axes[1].axhline(y=0, linewidth=1)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Alignment Improvement (%)')
    axes[1].set_title('(b) Per-Layer Outcomes')
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(layer_indices, mean_drifts, alpha=0.7)
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Mean Drift Δμ')
    axes[2].set_title('(c) Adaptation Magnitude')
    axes[2].grid(True, alpha=0.3)

    worst_layer = np.argmax(A_initial_layerwise) + 1
    for ax in axes:
        ax.axvline(x=worst_layer, linestyle=':', linewidth=2, alpha=0.5,
                    label=f'Worst Layer ({worst_layer})')

    plt.tight_layout()
    plt.savefig('layer_depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================
# MAIN — full CoDA experimental pipeline
# ============================================================
def run_coda_full_experiment():
    print("=" * 70)
    print("CoDA FULL EXPERIMENTAL PIPELINE")
    print("=" * 70)

    pretrained_model = load_model("CAFNet.h5")
    ADAPTED_MODEL_PATH = "adapted_CAFNet_CoDA_complete_analysis.h5"

    # Freeze everything except BN
    for layer in pretrained_model.layers:
        layer.trainable = False
    bn_layers = []
    for layer in pretrained_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
            bn_layers.append(layer)
    print(f"\nTotal BN layers: {len(bn_layers)}")

    pretrained_model.compile(optimizer=Adam(learning_rate=1e-5), loss="mse")

    # Adaptation dataset (target domain, unlabeled)
    x_adapt = tf.cast(x_test_20_2d[:475], tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x_adapt).batch(8)

    # ---- Save source (pretrained) BN state ----
    initial_gamma = [l.gamma.numpy().copy() for l in bn_layers]
    initial_beta = [l.beta.numpy().copy() for l in bn_layers]
    initial_mean = [l.moving_mean.numpy().copy() for l in bn_layers]
    initial_var = [l.moving_variance.numpy().copy() for l in bn_layers]
    source_stats = [(m, v) for m, v in zip(initial_mean, initial_var)]

    # ============================================================
    # STEP 1 — Empirical target-domain statistics (unsupervised)
    # ============================================================
    print("\n[Step 1] Computing empirical target-domain BN statistics...")
    target_stats = compute_empirical_bn_statistics(pretrained_model, dataset)
    A_initial = compute_alignment(source_stats, target_stats)
    A_initial_layerwise = compute_layer_wise_alignment(source_stats, target_stats)
    print(f"  A_initial = {A_initial:.6e}")

    consistency_before = compute_consistency_per_sample(pretrained_model, x_adapt)

    # ============================================================
    # STEP 2 — BN Component Substitution Ablation (Table 9 / Analysis 2)
    # ============================================================
    print("\n[Step 2] BN component substitution ablation (Analysis 2)...")
    delta_stats = compute_delta_stats(pretrained_model, x_adapt, target_stats)
    print(f"  Delta_stats  (statistics replaced, affine frozen) = {delta_stats:.6f}")

    print("  Estimating target-domain affine parameters (consistency loss, "
          "statistics held fixed)...")
    target_affine = obtain_target_affine_via_consistency(
        pretrained_model, dataset, iterations=50, lr=1e-5
    )
    delta_affine = compute_delta_affine(pretrained_model, x_adapt, target_affine)
    print(f"  Delta_affine (affine replaced, statistics frozen)   = {delta_affine:.6f}")

    # ============================================================
    # STEP 3 — CoDA adaptation (both mechanisms, applied in place)
    # ============================================================
    print("\n[Step 3] Running CoDA adaptation (EMA statistics + consistency-driven affine)...")

    @tf.function
    def coda_step(x):
        noise = tf.random.normal(tf.shape(x), stddev=0.02)
        with tf.GradientTape() as tape:
            p1 = pretrained_model(x, training=True)
            p2 = pretrained_model(x + noise, training=True)
            if isinstance(p1, list):
                loss = tf.add_n([tf.reduce_mean(tf.square(a - b)) for a, b in zip(p1, p2)])
            else:
                loss = tf.reduce_mean(tf.square(p1 - p2))
        grads = tape.gradient(loss, pretrained_model.trainable_weights)
        grads_vars = [(g, v) for g, v in zip(grads, pretrained_model.trainable_weights) if g is not None]
        pretrained_model.optimizer.apply_gradients(grads_vars)
        return loss

    N_CYCLES = 50
    for epoch in range(N_CYCLES):
        epoch_losses = []
        for batch in dataset:
            epoch_losses.append(coda_step(batch).numpy())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{N_CYCLES} | Avg consistency loss: {np.mean(epoch_losses):.6f}")

    # ============================================================
    # STEP 4 — Post-adaptation measurements (Table 8, Table 10, correlation)
    # ============================================================
    adapted_stats = [(l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy()) for l in bn_layers]
    A_final = compute_alignment(adapted_stats, target_stats)
    A_final_layerwise = compute_layer_wise_alignment(adapted_stats, target_stats)
    alignment_gain = (A_initial - A_final) / A_initial * 100 if A_initial > 0 else 0.0

    consistency_after = compute_consistency_per_sample(pretrained_model, x_adapt)

    gamma_drifts, beta_drifts, mean_drifts, var_drifts = [], [], [], []
    for i, l in enumerate(bn_layers):
        gamma_drifts.append(np.linalg.norm(l.gamma.numpy() - initial_gamma[i]))
        beta_drifts.append(np.linalg.norm(l.beta.numpy() - initial_beta[i]))
        mean_drifts.append(np.linalg.norm(l.moving_mean.numpy() - initial_mean[i]))
        var_drifts.append(np.linalg.norm(l.moving_variance.numpy() - initial_var[i]))

    avg_gamma_drift, avg_beta_drift = np.mean(gamma_drifts), np.mean(beta_drifts)
    avg_mean_drift, avg_var_drift = np.mean(mean_drifts), np.mean(var_drifts)
    stats_to_params_ratio = (avg_mean_drift + avg_var_drift) / (avg_gamma_drift + avg_beta_drift)

    corr_mean, p_mean = pearsonr(A_initial_layerwise, mean_drifts)
    corr_var, p_var = pearsonr(A_initial_layerwise, var_drifts)

    # ============================================================
    # STEP 5 — Print tables in the paper's format
    # ============================================================
    print("\n" + "=" * 70)
    print("TABLE 8-STYLE OUTPUT — Initial and Final BN Misalignment")
    print("=" * 70)
    print(f"{'A_initial':>14} | {'A_final':>14} | {'Improvement (%)':>16}")
    print(f"{A_initial:>14.4e} | {A_final:>14.4e} | {alignment_gain:>+16.1f}")

    print("\n" + "=" * 70)
    print("TABLE 9-STYLE OUTPUT — BN Component Substitution Ablation")
    print("=" * 70)
    print(f"{'Delta_stats':>14} | {'Delta_affine':>14}")
    print(f"{delta_stats:>14.4f} | {delta_affine:>14.4f}")

    print("\n" + "=" * 70)
    print("TABLE 10-STYLE OUTPUT — Statistics:Parameters Drift Ratio")
    print("=" * 70)
    print(f"  Avg |Delta gamma|   = {avg_gamma_drift:.6f}")
    print(f"  Avg |Delta beta|    = {avg_beta_drift:.6f}")
    print(f"  Avg |Delta mu|      = {avg_mean_drift:.6f}")
    print(f"  Avg |Delta sigma^2| = {avg_var_drift:.6f}")
    print(f"  Statistics:Parameters ratio = {stats_to_params_ratio:.1f} : 1")

    print("\n" + "=" * 70)
    print("CORRELATION TABLE — Initial Misalignment vs Statistics Drift")
    print("=" * 70)
    print(f"  r(Delta mu)      = {corr_mean:.3f}  (p = {p_mean:.4g})")
    print(f"  r(Delta sigma^2) = {corr_var:.3f}  (p = {p_var:.4g})")

    # ============================================================
    # STEP 6 — Final predictions and model save
    # ============================================================
    coda_predictions = pretrained_model.predict(x_test_20_2d, batch_size=8, verbose=1)
    pretrained_model.save(ADAPTED_MODEL_PATH)
    print(f"\nSaved adapted model -> {ADAPTED_MODEL_PATH}")

    # ============================================================
    # STEP 7 — Visualizations (paper figures)
    # ============================================================
    plot_bn_alignment(A_initial_layerwise, A_final_layerwise)
    plot_bn_alignment_bar(A_initial_layerwise, A_final_layerwise)
    plot_drift(gamma_drifts, beta_drifts, mean_drifts, var_drifts)
    plot_consistency(consistency_before, consistency_after)
    plot_correlation(A_initial_layerwise, mean_drifts)
    plot_layer_depth_pattern(A_initial_layerwise, A_final_layerwise, mean_drifts)
    plot_bn_distribution_shift(source_stats, target_stats, adapted_stats, layer_idx=0)

    print("\n" + "=" * 70)
    print("CoDA full experimental pipeline complete.")
    print("=" * 70)

    return {
        "predictions": coda_predictions,
        "A_initial": A_initial,
        "A_final": A_final,
        "improvement_pct": alignment_gain,
        "delta_stats": delta_stats,
        "delta_affine": delta_affine,
        "stats_to_params_ratio": stats_to_params_ratio,
        "corr_mean": (corr_mean, p_mean),
        "corr_var": (corr_var, p_var),
    }

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    results = run_coda_full_experiment()
