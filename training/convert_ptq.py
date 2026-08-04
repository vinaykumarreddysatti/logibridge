"""Task F1: build model variants M1 (FP32 baseline TFLite, for a fair
latency comparison against the quantised variants) and M2 (Post-Training
Full-INT8 Quantisation, >=200 calibration samples from the representative
training distribution).
"""
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "models")


def representative_dataset_gen(X_norm, n_samples=200, seed=0):
    n = min(n_samples, len(X_norm))
    idx = np.random.default_rng(seed).choice(len(X_norm), n, replace=False)
    for i in idx:
        yield [X_norm[i:i + 1].astype(np.float32)]


def convert_fp32(keras_model_path, out_path):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"M1 FP32 baseline -> {out_path} ({len(tflite_model) / 1024:.2f} KB)")
    return out_path


def convert_int8(keras_model_path, X_train_norm, out_path, n_calib=200):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X_train_norm, n_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"M2 PTQ INT8 -> {out_path} ({len(tflite_model) / 1024:.2f} KB)")
    return out_path


if __name__ == "__main__":
    # Calibration set is drawn strictly from the training split (saved by
    # train_model.py) so validation data is never touched during PTQ.
    X_train_norm = np.load(os.path.join(MODELS_DIR, "train_features_norm.npy")).astype(np.float32)

    keras_path = os.path.join(MODELS_DIR, "logibridge_mlp.keras")
    convert_fp32(keras_path, os.path.join(MODELS_DIR, "model_fp32.tflite"))
    convert_int8(keras_path, X_train_norm, os.path.join(MODELS_DIR, "model_int8.tflite"), n_calib=200)
