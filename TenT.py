import tensorflow as tf
from tensorflow.keras import layers, models
from helper_20_1 import *
np.random.seed(19)
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import KFold
import os
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

# Step 1: Select 10,000 random samples from the original dataset (20,000 samples)
total_selected_samples = 500

# Generate a list of indices and shuffle them
indices_20_2d = np.arange(total_num_20_2d)
np.random.shuffle(indices_20_2d)

# Select 10,000 random indices
selected_indices_20_2d = indices_20_2d[:total_selected_samples]

# Get the corresponding data for the 10,000 chosen indices
x_selected_20_2d = x0_20_2d[selected_indices_20_2d]
y_selected_20_2d = y0_20_2d[selected_indices_20_2d]

# Step 2: Split the 10,000 samples into training (80%), validation (15%), and testing (5%)
train_prop = 0.95
val_prop = 0.0
test_prop = 0.05

# Compute the number of samples for each set
train_num_20_2d = int(train_prop * total_selected_samples)  # 80% of 10k
val_num_20_2d = int(val_prop * total_selected_samples)  # 15% of 10k
test_num_20_2d = total_selected_samples - train_num_20_2d - val_num_20_2d  # 5% of 10k

# Generate a list of indices for the 10k selected samples and shuffle them
indices_selected_20_2d = np.arange(total_selected_samples)
np.random.shuffle(indices_selected_20_2d)

# Split into training, validation, and testing sets
train_indices_20_2d = indices_selected_20_2d[:train_num_20_2d]
val_indices_20_2d = indices_selected_20_2d[train_num_20_2d:train_num_20_2d + val_num_20_2d]
test_indices_20_2d = indices_selected_20_2d[train_num_20_2d + val_num_20_2d:]
print(test_indices_20_2d)
# Get the corresponding data for training, validation, and testing sets
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
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

# ============================================================
# BN STATISTICS UTILITIES (UNCHANGED)
# ============================================================

def compute_empirical_bn_statistics(model, dataset):
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
    total = 0.0
    for (m_a, v_a), (m_b, v_b) in zip(stats_A, stats_B):
        total += np.linalg.norm(m_a - m_b) ** 2
        total += np.linalg.norm(v_a - v_b) ** 2
    return total


def compute_layer_wise_alignment(stats_A, stats_B):
    layer_alignments = []
    for (m_a, v_a), (m_b, v_b) in zip(stats_A, stats_B):
        A_ell = np.linalg.norm(m_a - m_b) ** 2 + np.linalg.norm(v_a - v_b) ** 2
        layer_alignments.append(A_ell)
    return layer_alignments


# ============================================================
#  TENT: ENTROPY LOSS
# ============================================================

def entropy_loss(pred):
    eps = 1e-8

    if isinstance(pred, list):
        losses = []
        for p in pred:
            p = tf.nn.softmax(p, axis=-1)
            ent = -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1)
            losses.append(tf.reduce_mean(ent))
        return tf.add_n(losses)
    else:
        p = tf.nn.softmax(pred, axis=-1)
        ent = -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1)
        return tf.reduce_mean(ent)


# ============================================================
# TENT ADAPTATION STEP
# ============================================================

@tf.function
def adaptation_step_tent(model, x):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        loss = entropy_loss(pred)

    grads = tape.gradient(loss, model.trainable_weights)

    grads_vars = [
        (g, v) for g, v in zip(grads, model.trainable_weights)
        if g is not None
    ]

    model.optimizer.apply_gradients(grads_vars)

    return loss


# ============================================================
# MAIN PIPELINE (TENT VERSION)
# ============================================================

def run_tent_adaptation_full_analysis():

    pretrained_model = load_model("CAFNet.h5")
    ADAPTED_MODEL_PATH = "adapted_CAFNet_TENT_analysis.h5"

    # Freeze all except BN
    for layer in pretrained_model.layers:
        layer.trainable = False

    bn_layers = []
    for layer in pretrained_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
            bn_layers.append(layer)

    print(f"\n{'='*70}")
    print(f"TENT Mode | BN layers: {len(bn_layers)}")
    print(f"{'='*70}\n")

    pretrained_model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=None  # not used
    )

    x_adapt = tf.cast(x_train_20_2d[:475], tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x_adapt).batch(8)

    # ============================================================
    # SAVE INITIAL STATE
    # ============================================================

    initial_gamma = [l.gamma.numpy().copy() for l in bn_layers]
    initial_beta = [l.beta.numpy().copy() for l in bn_layers]
    initial_moving_mean = [l.moving_mean.numpy().copy() for l in bn_layers]
    initial_moving_var = [l.moving_variance.numpy().copy() for l in bn_layers]

    source_stats = [
        (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
        for l in bn_layers
    ]

    # ============================================================
    # EXPERIMENT 1 — INITIAL MISALIGNMENT
    # ============================================================

    print("\n[EXP 1] Initial Misalignment")
    target_stats = compute_empirical_bn_statistics(pretrained_model, dataset)

    A_initial = compute_alignment(source_stats, target_stats)
    A_initial_layerwise = compute_layer_wise_alignment(source_stats, target_stats)

    print(f"A_initial = {A_initial:.6f}")

    # ============================================================
    # EXPERIMENT 2 — MANUAL INJECTION
    # ============================================================

    print("\n[EXP 2] Manual BN Injection")

    manual_model = clone_model(pretrained_model)
    manual_model.set_weights(pretrained_model.get_weights())

    manual_bn_layers = [
        l for l in manual_model.layers if isinstance(l, BatchNormalization)
    ]

    for layer, (m_t, v_t) in zip(manual_bn_layers, target_stats):
        layer.moving_mean.assign(m_t)
        layer.moving_variance.assign(v_t)

    baseline_predictions = pretrained_model.predict(x_test_20_2d, batch_size=8, verbose=0)
    manual_predictions = manual_model.predict(x_test_20_2d, batch_size=8, verbose=0)

    if isinstance(baseline_predictions, list):
        mse_list = []
        for base, manual in zip(baseline_predictions, manual_predictions):
            mse_list.append(np.mean((manual - base) ** 2))
        mse_manual_vs_baseline = np.mean(mse_list)
    else:
            mse_manual_vs_baseline = np.mean(
                ( manual_predictions - baseline_predictions) ** 2
    )    
    print(f"MSE change = {mse_manual_vs_baseline:.6f}")

    # ============================================================
    # 🔥 TENT ADAPTATION
    # ============================================================

    print("\n[TENT] Running adaptation...")

    N_ITERATIONS = 50   # optimizer steps (same budget as CoDA, Table 5); the old loop ran 50 epochs over the pool

    batches = list(dataset)
    for it in range(N_ITERATIONS):
        loss = adaptation_step_tent(pretrained_model, batches[it % len(batches)])
        if (it + 1) % 10 == 0 or it == 0:
            print(f"Iteration {it+1:3d} | Loss: {loss.numpy():.6f}")

    # ============================================================
    # FINAL MISALIGNMENT
    # ============================================================

    adapted_stats = [
        (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
        for l in bn_layers
    ]

    A_final = compute_alignment(adapted_stats, target_stats)

    print(f"\nA_final = {A_final:.6f}")
    print(f"Improvement = {(A_initial - A_final)/A_initial*100:.2f}%")

    # ============================================================
    # DRIFT ANALYSIS
    # ============================================================

    gamma_drifts = []
    beta_drifts = []
    mean_drifts = []
    var_drifts = []

    for i, layer in enumerate(bn_layers):
        gamma_drifts.append(np.linalg.norm(layer.gamma.numpy() - initial_gamma[i]))
        beta_drifts.append(np.linalg.norm(layer.beta.numpy() - initial_beta[i]))
        mean_drifts.append(np.linalg.norm(layer.moving_mean.numpy() - initial_moving_mean[i]))
        var_drifts.append(np.linalg.norm(layer.moving_variance.numpy() - initial_moving_var[i]))

    print("\n--- Drift Summary ---")
    print(f"Avg Δγ: {np.mean(gamma_drifts):.6f}")
    print(f"Avg Δβ: {np.mean(beta_drifts):.6f}")
    print(f"Avg Δμ: {np.mean(mean_drifts):.6f}")
    print(f"Avg Δσ²: {np.mean(var_drifts):.6f}")

    ratio = (np.mean(mean_drifts) + np.mean(var_drifts)) / (
        np.mean(gamma_drifts) + np.mean(beta_drifts) + 1e-8
    )

    print(f"Stats:Params ratio = {ratio:.1f}:1")

    # ============================================================
    # FINAL PREDICTIONS
    # ============================================================

    predictions = pretrained_model.predict(
        x_test_20_2d,
        batch_size=8,
        verbose=1
    )

    pretrained_model.save(ADAPTED_MODEL_PATH)

    print("\nTENT Analysis Complete")

    return predictions


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preds = run_tent_adaptation_full_analysis()