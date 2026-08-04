"""Task D1: generate the labelled training dataset by running the sensor
generator in each mode for the prescribed duration and labelling every
window in that run with the class the mode represents.

  Class 0 Normal    --anomaly none        20 min  ~120 windows
  Class 1 Warning   --anomaly temp_drift  15 min  ~90 windows
  Class 2 Critical  --anomaly combined    15 min  ~90 windows

Note (see reports/ for the fuller writeup): the simulator's linear
temperature drift (+0.08C/reading, Task C1) run for a full 15-minute
Warning scenario would accumulate to an unrealistic ~70C above setpoint
without a cap, at which point most of the run's windows would read as
Critical-level temperatures under the Section 2 class thresholds despite
being labelled Warning here. simulator.py caps accumulated drift at a
physically plausible 8C above setpoint (MAX_TEMP_DRIFT_C) so the labelled
data stays physically sane, while this script keeps the mode-based
labelling convention specified above (matching the expected sample
counts) rather than re-deriving labels from instantaneous thresholds.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from simulator import generate_offline_series  # noqa: E402
from preprocessing import extract_features  # noqa: E402

CLASSES = [
    (0, "none", 20 * 60),
    (1, "temp_drift", 15 * 60),
    (2, "combined", 15 * 60),
]


def build_dataset(seed=42, out_dir=None):
    out_dir = out_dir or os.path.dirname(__file__)
    all_X, all_y = [], []
    t_cursor = 0.0
    for label, mode, duration in CLASSES:
        series = generate_offline_series(mode, duration, seed=seed + label, start_ts=t_cursor)
        temp_ts, temp_vals = series["temperature"]
        vib_ts, vib_vals = series["vibration"]
        _, feats = extract_features(temp_ts, temp_vals, vib_ts, vib_vals)
        labels = np.full(len(feats), label)
        all_X.append(feats)
        all_y.append(labels)
        print(f"class {label} ({mode}, {duration/60:.0f} min): {len(feats)} windows, "
              f"temp_mean range [{feats[:,0].min():.2f}, {feats[:,0].max():.2f}]C")
        t_cursor += duration

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    np.save(os.path.join(out_dir, "dataset_X.npy"), X)
    np.save(os.path.join(out_dir, "dataset_y.npy"), y)
    print(f"total: {len(X)} windows, feature dim {X.shape[1]}")
    return X, y


if __name__ == "__main__":
    build_dataset()
