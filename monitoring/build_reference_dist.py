"""Task E1: build the PSI reference distribution from 300 clean
Normal-class inference confidence scores, binned into 4 buckets
([0,0.25), [0.25,0.50), [0.50,0.75), [0.75,1.0]) and frozen to
reference_dist.json.

Confidence must come from the *deployed* TFLite model
(inference/model.tflite, pruned+INT8), not the raw FP32 Keras model --
their confidence calibration differs (quantization and the pruning
fine-tune both shift it), so a reference built from the wrong artifact
never matches live PSI even under clean conditions. This mirrors
inference/inference_service.py's _run_tflite exactly (same int8
quantize-in/dequantize-out steps) so the two stay consistent.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "..", "data_pipeline"))
from simulator import generate_offline_series  # noqa: E402
from preprocessing import extract_features, normalise, load_stats  # noqa: E402
from psi import bin_distribution, BIN_EDGES  # noqa: E402


def _run_tflite(interpreter, input_details, output_details, feat_norm):
    x = feat_norm
    if input_details["dtype"] == np.int8:
        scale, zero_point = input_details["quantization"]
        x = (x / scale + zero_point).round().astype(np.int8)
    else:
        x = x.astype(np.float32)
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])
    if output_details["dtype"] == np.int8:
        scale, zero_point = output_details["quantization"]
        out = (out.astype(np.float32) - zero_point) * scale
    return out[0]


def main(n_windows=300, out_path=None, model_path=None, seed=999):
    out_path = out_path or os.path.join(HERE, "reference_dist.json")
    stats_path = os.path.join(HERE, "..", "data_pipeline", "training_stats.npy")
    # MODEL_PATH env var, same convention as inference/inference_service.py,
    # so the reference is always built against whichever artifact the live
    # service is actually pointed at -- default matches inference's own
    # default deployment target.
    model_path = model_path or os.environ.get(
        "MODEL_PATH", os.path.join(HERE, "..", "inference", "model.tflite")
    )

    mean, std = load_stats(stats_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    duration_s = int(n_windows * 10 + 30)  # enough clean data for n_windows @ 10s step
    series = generate_offline_series("none", duration_s, seed=seed)
    temp_ts, temp_vals = series["temperature"]
    vib_ts, vib_vals = series["vibration"]
    _, feats = extract_features(temp_ts, temp_vals, vib_ts, vib_vals)
    feats = feats[:n_windows].astype(np.float32)
    feats_norm = normalise(feats, mean, std)

    confidences = np.array([
        np.max(_run_tflite(interpreter, input_details, output_details, row[None, :]))
        for row in feats_norm
    ])

    ref_props = bin_distribution(confidences)
    reference = {
        "bin_edges": BIN_EDGES,
        "proportions": ref_props.tolist(),
        "n_windows": int(len(confidences)),
    }
    with open(out_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"reference_dist.json written ({len(confidences)} windows) -> {out_path}")
    print(json.dumps(reference, indent=2))


if __name__ == "__main__":
    main()
