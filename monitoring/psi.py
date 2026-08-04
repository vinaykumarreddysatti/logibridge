"""Population Stability Index (PSI) utilities (Task E1)."""
import numpy as np

BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]


def bin_distribution(confidences, bin_edges=BIN_EDGES):
    counts, _ = np.histogram(confidences, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return counts / total


def compute_psi(reference_props, current_props, epsilon=1e-4):
    """PSI = sum((current - reference) * ln(current / reference)) over bins."""
    reference_props = np.clip(np.asarray(reference_props, dtype=float), epsilon, None)
    current_props = np.clip(np.asarray(current_props, dtype=float), epsilon, None)
    return float(np.sum((current_props - reference_props) * np.log(current_props / reference_props)))
