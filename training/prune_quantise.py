"""Task F1: M3 -- 35% STRUCTURED filter (neuron/unit) pruning on a
PolynomialDecay schedule, followed by Full INT8 PTQ.

This deliberately does NOT use tensorflow_model_optimization's
`prune_low_magnitude`: that API performs UNSTRUCTURED per-weight magnitude
pruning (individual weights zeroed inside a dense matrix of unchanged
shape), which is a different technique from what Task F1 specifies.
"Structured filter pruning" for a Dense/MLP architecture (no convolutional
filters here) is the neuron/unit-pruning analogue: rank each hidden
layer's units by the L2 norm of their incoming weight vector, drop the
lowest-scoring units *entirely*, and physically shrink the layer -- so the
resulting model has fewer actual neurons, not just more zeros. That is
what removes real FLOPs/parameters (relevant to the compute-bound Roofline
conclusion in Task B2) rather than producing an unstructured-sparse
tensor of the same dense shape, which a standard TFLite kernel gets no
latency benefit from without specialised sparse-compute support.

The PolynomialDecay schedule itself is implemented directly (not via
tfmot), using the same formula tfmot.sparsity.keras.PolynomialDecay uses:

    sparsity(step) = final_sparsity + (initial_sparsity - final_sparsity)
                      * (1 - (step - begin_step) / (end_step - begin_step)) ** power

with power=3 (the tfmot default), initial_sparsity=0, final_sparsity=0.35.
At each pruning step (one per epoch here) the target sparsity is
recomputed and, if it calls for removing more units than are currently
present, the network is re-ranked on its *current* (already fine-tuned)
weights and shrunk down to the new target width -- gradual, scheduled
structural pruning interleaved with fine-tuning, not a single one-shot
resize. After the schedule reaches end_step, additional pure fine-tuning
epochs run with the network held at its final size to recover accuracy,
tracking the best validation-accuracy checkpoint.
"""
import os

import numpy as np
import tensorflow as tf
import keras

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")

FINAL_SPARSITY = 0.35
INITIAL_SPARSITY = 0.0
POLY_POWER = 3            # tfmot.sparsity.keras.PolynomialDecay default exponent
BEGIN_STEP = 0
END_STEP = 30              # sparsity ramps from 0 -> 35% over these 30 epochs
STABLE_FINETUNE_EPOCHS = 20  # additional fine-tuning once final sparsity is reached
BATCH_SIZE = 16


def polynomial_decay_sparsity(step, begin_step=BEGIN_STEP, end_step=END_STEP,
                               initial_sparsity=INITIAL_SPARSITY,
                               final_sparsity=FINAL_SPARSITY, power=POLY_POWER):
    if step <= begin_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity
    progress = (step - begin_step) / (end_step - begin_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** power


def unit_importance(W_in):
    """L2 norm of each unit's incoming weight vector -- the Dense-layer
    analogue of an L2-norm filter-importance criterion (Li et al., 2016
    style channel/filter pruning, adapted from conv filters to MLP
    units)."""
    return np.linalg.norm(W_in, axis=0)


def shrink_to(model, keep1, keep2):
    """Re-rank the CURRENT model's units by importance and physically
    shrink hidden layer 1 to `keep1` units and hidden layer 2 to `keep2`
    units, carrying over the current (already fine-tuned) weights for
    the units that survive."""
    W1, b1 = model.layers[0].get_weights()
    W2, b2 = model.layers[1].get_weights()
    W3, b3 = model.layers[2].get_weights()

    keep_idx1 = np.sort(np.argsort(-unit_importance(W1))[:keep1])
    W1_p, b1_p = W1[:, keep_idx1], b1[keep_idx1]
    W2_p = W2[keep_idx1, :]

    keep_idx2 = np.sort(np.argsort(-unit_importance(W2_p))[:keep2])
    W2_pp, b2_p = W2_p[:, keep_idx2], b2[keep_idx2]
    W3_p = W3[keep_idx2, :]

    smaller = keras.Sequential([
        keras.layers.Input(shape=(W1.shape[0],)),
        keras.layers.Dense(keep1, activation="relu"),
        keras.layers.Dense(keep2, activation="relu"),
        keras.layers.Dense(W3.shape[1], activation="softmax"),
    ])
    smaller.build((None, W1.shape[0]))
    smaller.set_weights([W1_p, b1_p, W2_pp, b2_p, W3_p, b3])
    return smaller


def prune_and_finetune(keras_model_path, X_train, y_train, X_val, y_val):
    source = keras.models.load_model(keras_model_path)
    n1_full = source.layers[0].get_weights()[0].shape[1]  # 32
    n2_full = source.layers[1].get_weights()[0].shape[1]  # 16
    total_before = source.count_params()

    model = source
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    current_keep1, current_keep2 = n1_full, n2_full

    best_val_acc = -1.0
    best_weights = None
    total_epochs = END_STEP + STABLE_FINETUNE_EPOCHS

    print(f"gradual structured pruning: PolynomialDecay(power={POLY_POWER}) "
          f"from {INITIAL_SPARSITY:.0%} to {FINAL_SPARSITY:.0%} sparsity "
          f"over steps [{BEGIN_STEP}, {END_STEP}], then {STABLE_FINETUNE_EPOCHS} stable epochs")

    for step in range(total_epochs):
        target_sparsity = polynomial_decay_sparsity(step)
        target_keep1 = n1_full - int(round(n1_full * target_sparsity))
        target_keep2 = n2_full - int(round(n2_full * target_sparsity))

        if target_keep1 < current_keep1 or target_keep2 < current_keep2:
            model = shrink_to(model, target_keep1, target_keep2)
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            current_keep1, current_keep2 = target_keep1, target_keep2

        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=1, batch_size=BATCH_SIZE, verbose=0,
        )
        val_acc = history.history["val_accuracy"][0]

        if step % 5 == 0 or step == total_epochs - 1:
            print(f"  step {step:3d}  target_sparsity={target_sparsity:.3f}  "
                  f"units=({current_keep1},{current_keep2})  val_acc={val_acc:.4f}")

        # Only start tracking "best" once the schedule has reached final
        # sparsity -- earlier checkpoints are mid-shrink and not
        # comparable/deployable at the target size.
        if step >= END_STEP and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

    if best_weights is not None:
        model.set_weights(best_weights)

    total_after = model.count_params()
    print(f"\nparameter count: {total_before} -> {total_after} "
          f"({(1 - total_after / total_before):.1%} reduction)")

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Structurally pruned ({FINAL_SPARSITY:.0%}) model validation accuracy: {val_acc:.4f}")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    for c in (0, 1, 2):
        mask = y_val == c
        recall = np.mean(y_pred[mask] == c) if mask.sum() else float("nan")
        print(f"  class {c} recall: {recall:.4f} (n={mask.sum()})")

    return model


def convert_pruned_int8(model, X_train_norm, out_path, n_calib=200, seed=1):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        n = min(n_calib, len(X_train_norm))
        idx = np.random.default_rng(seed).choice(len(X_train_norm), n, replace=False)
        for i in idx:
            yield [X_train_norm[i:i + 1].astype(np.float32)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"M3 Structured-Pruned({FINAL_SPARSITY:.0%})+INT8 -> {out_path} ({len(tflite_model) / 1024:.2f} KB)")
    return out_path


if __name__ == "__main__":
    X_train = np.load(os.path.join(MODELS_DIR, "train_features_norm.npy")).astype(np.float32)
    y_train = np.load(os.path.join(MODELS_DIR, "train_labels.npy"))
    X_val = np.load(os.path.join(MODELS_DIR, "val_features_norm.npy")).astype(np.float32)
    y_val = np.load(os.path.join(MODELS_DIR, "val_labels.npy"))

    keras_path = os.path.join(MODELS_DIR, "logibridge_mlp.keras")
    pruned_model = prune_and_finetune(keras_path, X_train, y_train, X_val, y_val)
    pruned_model.save(os.path.join(MODELS_DIR, "logibridge_mlp_pruned.keras"))
    convert_pruned_int8(pruned_model, X_train, os.path.join(MODELS_DIR, "model_pruned_int8.tflite"))
