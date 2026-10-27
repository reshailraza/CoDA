import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D, Add, Input
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# -----------------------------------------------------------------------------
# 1. AdaptBlock -- Sec 3.2 / Fig. 2 of the paper (convolutional half only)
# -----------------------------------------------------------------------------

MAP_SHAPE = (1, 64, 64)  # channels-first (C, H, W), matches CAFNet's [x_1, x_2] outputs


def make_adapt_block(name):
    """
    Two 3x3 conv layers, no bias, no activation, zero-initialized, wrapped in
    a residual (skip) connection -- so the block is exactly the IDENTITY map
    before any adaptation happens, i.e. AdaptNet(y) == pretrained model's
    output at t=0. This reproduces the paper's statement (Sec 3.2): "the
    convolutional layers are initialized with 0".
    """
    inp = Input(shape=MAP_SHAPE, name=f'{name}_in')
    h = Conv2D(1, (3, 3), padding='same', use_bias=False,
               kernel_initializer='zeros', data_format='channels_first',
               name=f'{name}_conv1')(inp)
    h = Conv2D(1, (3, 3), padding='same', use_bias=False,
               kernel_initializer='zeros', data_format='channels_first',
               name=f'{name}_conv2')(h)
    out = Add(name=f'{name}_residual')([inp, h])
    return Model(inp, out, name=name)


def build_adaptnet(pretrained_model_path="CAFNet.h5"):
    """
    Loads the frozen pretrained CAFNet (theta*) and attaches one AdaptBlock
    to each of its two outputs (mu_a and mu_sp maps), matching "concatenate
    an adjustable lightweight linear operator ... to the end of the locked
    sub-NN" (Sec 3.2). All pretrained layers are frozen; only the two
    AdaptBlocks (phi) are trainable.
    """
    base = load_model(pretrained_model_path)
    for layer in base.layers:
        layer.trainable = False

    adapt_block_mua = make_adapt_block('adapt_mua')
    adapt_block_musp = make_adapt_block('adapt_musp')

    x1_pretrained, x2_pretrained = base.output
    x1_adapted = adapt_block_mua(x1_pretrained)
    x2_adapted = adapt_block_musp(x2_pretrained)

    adaptnet = Model(base.input, [x1_adapted, x2_adapted], name='AdaptNet')
    return adaptnet, base, [adapt_block_mua, adapt_block_musp]


def reset_adapt_blocks(adapt_blocks):
    """
    Re-zero every AdaptBlock conv kernel, i.e. reset phi back to the
    identity, before adapting to a NEW test sample -- the paper adapts fresh
    per test sample rather than accumulating adaptation across samples (Sec
    1.2 problem setting).
    """
    for block in adapt_blocks:
        for layer in block.layers:
            if isinstance(layer, Conv2D):
                layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])


# -----------------------------------------------------------------------------
# 2. Self-supervised test-time loss (adapted from Eqs. 7-8 -- see header note)
# -----------------------------------------------------------------------------

SIGMA_1 = 0.02      # measurement re-corruption noise std, paper's eps' ~ N(0, sigma_1^2 I)
LAMBDA_NULL = 0.1    # weight of the null-space anchor term, paper's lambda


def range_consistency_loss(adaptnet, y_sample):
    """
    R2R-style substitute for L^range (Eq. 7): two independent noisy copies
    of the SAME measurement should decode to the same reconstruction.
    """
    noise_a = tf.random.normal(tf.shape(y_sample), stddev=SIGMA_1)
    noise_b = tf.random.normal(tf.shape(y_sample), stddev=SIGMA_1)
    pred_a = adaptnet(y_sample + noise_a, training=True)
    pred_b = adaptnet(y_sample + noise_b, training=True)
    loss = 0.0
    for pa, pb in zip(pred_a, pred_b):
        loss += tf.reduce_mean(tf.square(pa - pb))
    return loss


def null_anchor_loss(adaptnet, y_sample, x_star):
    """
    Substitute for L^null (Eq. 8): anchor the adapted prediction to the
    frozen pretrained model's own estimate x* = F(y; theta*).
    """
    pred = adaptnet(y_sample, training=True)
    loss = 0.0
    for p, x in zip(pred, x_star):
        loss += tf.reduce_mean(tf.square(p - x))
    return loss


@tf.function
def adaptnet_train_step(adaptnet, y_sample, x_star):
    with tf.GradientTape() as tape:
        loss = (range_consistency_loss(adaptnet, y_sample)
                + LAMBDA_NULL * null_anchor_loss(adaptnet, y_sample, x_star))
    grads = tape.gradient(loss, adaptnet.trainable_variables)
    adaptnet.optimizer.apply_gradients(zip(grads, adaptnet.trainable_variables))
    return loss


# -----------------------------------------------------------------------------
# 3. Per-sample test-time adaptation loop (Sec 3.2, "Test-time adaption")
# -----------------------------------------------------------------------------

T_ITERS = 200        # paper: T = 200 for MRI, T = 100 for CT (Sec 4, "Model test-time adaptation")
N_MC_FINAL = 20       # Monte-Carlo averaged final prediction (paper's closing equation)
ADAPT_LR = 1e-3


def adapt_and_predict(adaptnet, base_frozen, adapt_blocks, y_sample):
    """
    Runs AdaptNet's test-time adaptation on a single measurement y_sample
    (shape (1, 2, 16, 15)) and returns the adapted prediction plus the loss
    curve for that sample.
    """
    reset_adapt_blocks(adapt_blocks)
    adaptnet.compile(optimizer=Adam(learning_rate=ADAPT_LR))

    x_star = base_frozen(y_sample, training=False)  # frozen pretrained estimate, used only as an anchor

    losses = []
    for _ in range(T_ITERS):
        loss = adaptnet_train_step(adaptnet, y_sample, x_star)
        losses.append(float(loss))

    mc_preds_1, mc_preds_2 = [], []
    for _ in range(N_MC_FINAL):
        noise = tf.random.normal(tf.shape(y_sample), stddev=SIGMA_1)
        p1, p2 = adaptnet(y_sample + noise, training=False)
        mc_preds_1.append(p1.numpy())
        mc_preds_2.append(p2.numpy())
    x1_final = np.mean(mc_preds_1, axis=0)
    x2_final = np.mean(mc_preds_2, axis=0)

    return x1_final, x2_final, losses
