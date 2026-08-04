"""LogiEdge preprocessing pipeline (Task C2).

Sequence: moving-average filtering -> sliding-window feature extraction
(30s window, 10s step, 6-value feature vector) -> z-score normalisation
using frozen training_stats.npy computed once from clean Normal-class
data and never recomputed at runtime.
"""
import numpy as np
from scipy.stats import kurtosis

MOVING_AVG_WINDOW = 5
FEATURE_WINDOW_S = 30.0
FEATURE_STEP_S = 10.0

FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_rate_of_change",
    "vib_rms", "vib_peak", "vib_kurtosis",
]


def moving_average(x, window=MOVING_AVG_WINDOW):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(x, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def extract_features(temp_ts, temp_vals, vib_ts, vib_vals):
    """Extract sliding-window features from filtered temperature and
    vibration series.

    Returns (window_start_times, feature_matrix[N, 6]) with columns
    matching FEATURE_NAMES.
    """
    temp_vals = moving_average(np.asarray(temp_vals, dtype=float))
    vib_vals = moving_average(np.asarray(vib_vals, dtype=float))
    temp_ts = np.asarray(temp_ts, dtype=float)
    vib_ts = np.asarray(vib_ts, dtype=float)

    if len(temp_ts) == 0:
        return np.array([]), np.zeros((0, 6))

    t0, t1 = temp_ts[0], temp_ts[-1]
    # Integer window count rather than a float np.arange endpoint: Unix
    # timestamps (~1.7e9) have float64 precision coarser than a naive
    # epsilon fudge factor, which silently drops the last window.
    duration = t1 - t0
    n_windows = max(int(np.floor((duration - FEATURE_WINDOW_S) / FEATURE_STEP_S)) + 1, 1)
    starts = t0 + np.arange(n_windows) * FEATURE_STEP_S

    rows, valid_starts = [], []
    for ws in starts:
        we = ws + FEATURE_WINDOW_S
        tmask = (temp_ts >= ws) & (temp_ts < we)
        vmask = (vib_ts >= ws) & (vib_ts < we)
        if tmask.sum() < 2 or vmask.sum() < 1:
            continue
        t_win = temp_vals[tmask]
        t_ts_win = temp_ts[tmask]
        v_win = vib_vals[vmask]

        temp_mean = t_win.mean()
        temp_std = t_win.std()
        dt_min = (t_ts_win[-1] - t_ts_win[0]) / 60.0
        temp_rate = (t_win[-1] - t_win[0]) / dt_min if dt_min > 0 else 0.0
        vib_rms = np.sqrt(np.mean(v_win ** 2))
        vib_peak = np.max(np.abs(v_win))
        vib_kurt = kurtosis(v_win, fisher=True, bias=False) if len(v_win) > 3 else 0.0
        vib_kurt = 0.0 if np.isnan(vib_kurt) else vib_kurt

        rows.append([temp_mean, temp_std, temp_rate, vib_rms, vib_peak, vib_kurt])
        valid_starts.append(ws)

    return np.array(valid_starts), np.array(rows)


def compute_and_save_stats(feature_matrix, path="training_stats.npy"):
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1e-6
    np.save(path, np.stack([mean, std]))
    return mean, std


def load_stats(path="training_stats.npy"):
    stats = np.load(path)
    return stats[0], stats[1]


def normalise(feature_matrix, mean, std):
    return (feature_matrix - mean) / std
