"""Task C2 mandatory experiment: measure the classification accuracy
impact of using correct normalisation stats vs. stats shifted by 3
standard deviations -- simulating a stale/mis-deployed training_stats.npy
(e.g. an OTA that updated the model but not the paired stats file).
"""
import json
import os

import numpy as np
import keras

from preprocessing import load_stats, normalise

HERE = os.path.dirname(__file__)


def run_experiment(model_path, val_features_path, val_labels_path,
                    stats_path, out_path):
    model = keras.models.load_model(model_path)
    X = np.load(val_features_path)   # raw, pre-normalisation features
    y = np.load(val_labels_path)
    mean, std = load_stats(stats_path)

    X_correct = normalise(X, mean, std)
    preds_correct = np.argmax(model.predict(X_correct, verbose=0), axis=1)
    acc_correct = float(np.mean(preds_correct == y))

    shifted_mean = mean + 3 * std
    X_shifted = normalise(X, shifted_mean, std)
    preds_shifted = np.argmax(model.predict(X_shifted, verbose=0), axis=1)
    acc_shifted = float(np.mean(preds_shifted == y))

    result = {
        "accuracy_correct_stats": round(acc_correct, 4),
        "accuracy_shifted_3sigma_stats": round(acc_shifted, 4),
        "accuracy_drop": round(acc_correct - acc_shifted, 4),
    }
    print(json.dumps(result, indent=2))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    models_dir = os.path.join(HERE, "..", "training", "models")
    run_experiment(
        model_path=os.path.join(models_dir, "logibridge_mlp.keras"),
        val_features_path=os.path.join(models_dir, "val_features.npy"),
        val_labels_path=os.path.join(models_dir, "val_labels.npy"),
        stats_path=os.path.join(HERE, "training_stats.npy"),
        out_path=os.path.join(HERE, "stats_shift_experiment_results.json"),
    )
