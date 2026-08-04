"""Task D1: train the LogiEdge classification MLP (2 hidden layers, 32
and 16 units, ReLU) on the generated dataset with an 80/20 held-out
split. Fails loudly if validation accuracy does not exceed the 88%
pharma-grade threshold -- the project is not gradable below this bar.
"""
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from preprocessing import compute_and_save_stats, normalise  # noqa: E402

MIN_VAL_ACCURACY = 0.88


def build_model(input_dim=6, num_classes=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    here = os.path.dirname(__file__)
    X = np.load(os.path.join(here, "dataset_X.npy"))
    y = np.load(os.path.join(here, "dataset_y.npy"))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=0, stratify=y
    )

    # Stats computed ONLY from clean Normal-class (label 0) training
    # windows (Task C2), then frozen -- inference never recomputes them.
    clean_mask = y_train == 0
    stats_path = os.path.join(here, "..", "data_pipeline", "training_stats.npy")
    mean, std = compute_and_save_stats(X_train[clean_mask], path=stats_path)

    X_train_n = normalise(X_train, mean, std)
    X_val_n = normalise(X_val, mean, std)

    model = build_model(input_dim=X.shape[1], num_classes=3)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=15, restore_best_weights=True
    )
    model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=150, batch_size=16, verbose=2, callbacks=[early_stop],
    )

    val_loss, val_acc = model.evaluate(X_val_n, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")

    models_dir = os.path.join(here, "models")
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, "logibridge_mlp.keras"))
    np.save(os.path.join(models_dir, "val_features.npy"), X_val)          # raw, pre-normalisation
    np.save(os.path.join(models_dir, "val_features_norm.npy"), X_val_n)
    np.save(os.path.join(models_dir, "val_labels.npy"), y_val)
    np.save(os.path.join(models_dir, "train_features_norm.npy"), X_train_n)
    np.save(os.path.join(models_dir, "train_labels.npy"), y_train)

    y_pred = np.argmax(model.predict(X_val_n, verbose=0), axis=1)
    print("\nPer-class recall on held-out validation set:")
    for c in (0, 1, 2):
        mask = y_val == c
        recall = np.mean(y_pred[mask] == c) if mask.sum() else float("nan")
        print(f"  class {c} recall: {recall:.4f} (n={mask.sum()})")

    if val_acc < MIN_VAL_ACCURACY:
        raise SystemExit(
            f"VALIDATION ACCURACY {val_acc:.4f} IS BELOW THE {MIN_VAL_ACCURACY:.0%} "
            "PHARMA-GRADE THRESHOLD (Task D1). Revisit features/architecture "
            "before proceeding to conversion/optimisation."
        )


if __name__ == "__main__":
    main()
