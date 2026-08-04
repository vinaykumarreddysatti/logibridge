"""Offline reproduction of the drift-injection story (Task E1): clean ->
combined anomaly -> clean again, computing rolling PSI exactly the way
the live monitor does. Runs in seconds instead of the ~25 real-time
minutes the full demo takes, so the detection/recovery behaviour can be
validated and the PSI trace captured as evidence for the Final Report
(Section 5) without waiting on wall-clock time.
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
from psi import bin_distribution, compute_psi, BIN_EDGES  # noqa: E402
from build_reference_dist import _run_tflite  # noqa: E402

PSI_ALERT_THRESHOLD = 0.25
ROLLING_WINDOW = 100


def confidences_for(mode, duration_s, interpreter, input_details, output_details, mean, std, seed):
    series = generate_offline_series(mode, duration_s, seed=seed)
    temp_ts, temp_vals = series["temperature"]
    vib_ts, vib_vals = series["vibration"]
    _, feats = extract_features(temp_ts, temp_vals, vib_ts, vib_vals)
    feats_norm = normalise(feats, mean, std).astype(np.float32)
    return np.array([
        np.max(_run_tflite(interpreter, input_details, output_details, row[None, :]))
        for row in feats_norm
    ])


def main():
    stats_path = os.path.join(HERE, "..", "data_pipeline", "training_stats.npy")
    model_path = os.environ.get("MODEL_PATH", os.path.join(HERE, "..", "inference", "model.tflite"))
    ref_path = os.path.join(HERE, "reference_dist.json")

    mean, std = load_stats(stats_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    reference = json.load(open(ref_path))

    # (phase name, anomaly mode, duration seconds, seed). clean_before is
    # long enough (25 min = 150 windows @ 10s step) to fully populate the
    # 100-sample rolling buffer with clean data before injection, matching
    # a live deployment that has been running normally before a fault.
    phases = [
        ("clean_before", "none", 25 * 60, 100),
        ("drift_injected", "combined", 10 * 60, 200),
        ("recovery", "none", 20 * 60, 300),
    ]

    all_conf = []
    for name, mode, dur, seed in phases:
        conf = confidences_for(mode, dur, interpreter, input_details, output_details, mean, std, seed=seed)
        all_conf.extend([(name, c) for c in conf])
        print(f"phase={name:14s} mode={mode:10s} windows={len(conf)} (~{dur/60:.0f} min)")

    injection_start_idx = next(i for i, (name, _) in enumerate(all_conf) if name == "drift_injected")

    # True FIFO rolling buffer of the last 100 inference confidence
    # scores, checked every 6 new samples (~60s at the 10s window step),
    # exactly mirroring the live drift_monitor.py logic.
    rolling = []
    for i in range(ROLLING_WINDOW, len(all_conf) + 1, 6):
        chunk = [c for _, c in all_conf[i - ROLLING_WINDOW:i]]
        phase_name = all_conf[i - 1][0]
        props = bin_distribution(chunk, BIN_EDGES)
        psi = compute_psi(reference["proportions"], props)
        seconds_since_injection = (i - injection_start_idx) * 10
        rolling.append({
            "index": i, "phase": phase_name, "psi": round(psi, 4),
            "seconds_since_injection": seconds_since_injection,
        })

    print()
    for r in rolling:
        flag = " [LOGIBRIDGE DRIFT ALERT]" if r["psi"] > PSI_ALERT_THRESHOLD else ""
        print(f"idx={r['index']:4d} phase={r['phase']:14s} PSI={r['psi']:.3f}{flag}")

    out_path = os.path.join(HERE, "drift_demo_trace.json")
    with open(out_path, "w") as f:
        json.dump(rolling, f, indent=2)

    first_alert = next((r for r in rolling if r["phase"] == "drift_injected" and r["psi"] > PSI_ALERT_THRESHOLD), None)
    during = [r["psi"] for r in rolling if r["phase"] == "drift_injected"]
    after = [r["psi"] for r in rolling if r["phase"] == "recovery"]
    max_during = max(during) if during else float("nan")
    min_after_tail = min(after[-3:]) if len(after) >= 3 else (min(after) if after else float("nan"))

    print(f"\nmax PSI during injection phase: {max_during:.3f} (alert threshold 0.25)")
    if first_alert:
        print(f"PSI first crossed 0.25 at {first_alert['seconds_since_injection']}s "
              f"after injection (target: within 300s)")
    else:
        print("PSI never crossed 0.25 during the injection phase")
    print(f"PSI at end of recovery:   {min_after_tail:.3f} (target < 0.10)")
    print(f"trace written -> {out_path}")


if __name__ == "__main__":
    main()
