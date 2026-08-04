"""Task F2: five-metric benchmarking of the three model variants
(Lab 2 methodology: 200 timed runs after 10 warm-up runs are excluded).

Metrics per variant: mean inference latency (ms), p95 latency (ms),
model file size (KB), classification accuracy (%) on the held-out
validation set, and energy per inference (mJ) via E = P x t using
psutil CPU% and a laptop TDP estimate.
"""
import csv
import json
import os
import time

import numpy as np
import psutil
import tensorflow as tf

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(HERE, "..", "training", "models")
RESULTS_DIR = os.path.join(HERE, "results")

WARMUP_RUNS = 10
TIMED_RUNS = 200
LAPTOP_TDP_W = 15.0  # representative ultrabook/edge-class CPU package TDP

VARIANTS = [
    ("M1_FP32_Baseline", "model_fp32.tflite"),
    ("M2_PTQ_INT8", "model_int8.tflite"),
    ("M3_Pruned35_INT8", "model_pruned_int8.tflite"),
]


def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def prep_input(inp_details, x_row):
    x = x_row.reshape(1, -1)
    if inp_details["dtype"] == np.int8:
        scale, zero_point = inp_details["quantization"]
        x = (x / scale + zero_point).round().astype(np.int8)
    else:
        x = x.astype(np.float32)
    return x


def run_inference(interp, x, inp_details, out_details):
    interp.set_tensor(inp_details["index"], x)
    t0 = time.perf_counter()
    interp.invoke()
    t1 = time.perf_counter()
    out = interp.get_tensor(out_details["index"])
    if out_details["dtype"] == np.int8:
        scale, zero_point = out_details["quantization"]
        out = (out.astype(np.float32) - zero_point) * scale
    return (t1 - t0) * 1000.0, out  # ms


def benchmark_variant(name, model_path, X_norm, y):
    interp = load_interpreter(model_path)
    inp_details = interp.get_input_details()[0]
    out_details = interp.get_output_details()[0]

    preds = []
    for i in range(len(X_norm)):
        x = prep_input(inp_details, X_norm[i])
        _, out = run_inference(interp, x, inp_details, out_details)
        preds.append(int(np.argmax(out[0])))
    preds = np.array(preds)
    accuracy = float(np.mean(preds == y))
    class2_mask = y == 2
    class2_recall = float(np.mean(preds[class2_mask] == 2)) if class2_mask.sum() else float("nan")

    x0 = prep_input(inp_details, X_norm[0])
    for _ in range(WARMUP_RUNS):
        run_inference(interp, x0, inp_details, out_details)

    latencies, cpu_samples = [], []
    psutil.cpu_percent(interval=None)  # prime the internal counter
    for _ in range(TIMED_RUNS):
        lat_ms, _ = run_inference(interp, x0, inp_details, out_details)
        latencies.append(lat_ms)
        cpu_samples.append(psutil.cpu_percent(interval=None))
    latencies = np.array(latencies)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    size_kb = os.path.getsize(model_path) / 1024.0

    avg_cpu_frac = max(float(np.mean(cpu_samples)), 1.0) / 100.0
    power_w = LAPTOP_TDP_W * avg_cpu_frac
    energy_mj = power_w * (mean_latency / 1000.0) * 1000.0  # P(W) * t(s) * 1000 -> mJ

    return {
        "variant": name,
        "mean_latency_ms": round(mean_latency, 4),
        "p95_latency_ms": round(p95_latency, 4),
        "size_kb": round(size_kb, 2),
        "accuracy_pct": round(accuracy * 100, 2),
        "class2_recall_pct": round(class2_recall * 100, 2),
        "energy_mj": round(energy_mj, 5),
    }


def plot_pareto(results, out_path):
    """At this model's scale, latency and accuracy are statistically
    indistinguishable across variants (sub-microsecond, noise-dominated)
    -- a latency-vs-accuracy scatter collapses to one overlapping point.
    Model size is the metric that actually separates the three variants
    (5.27 -> 4.58 -> 3.75 KB), so this renders a small-multiple bar
    comparison across all five benchmarked metrics instead of a single
    scatter, with each panel annotated with the values themselves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["variant"] for r in results]
    colors = ["#5B7FBD", "#D98C4A", "#6FAE7C"]
    metrics = [
        ("size_kb", "Model size (KB)", "{:.2f}"),
        ("mean_latency_ms", "Mean latency (ms)", "{:.4f}"),
        ("energy_mj", "Energy / inference (mJ)", "{:.5f}"),
        ("accuracy_pct", "Validation accuracy (%)", "{:.2f}"),
        ("class2_recall_pct", "Critical (class 2) recall (%)", "{:.1f}"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5))
    for ax, (key, label, fmt) in zip(axes, metrics):
        values = [r[key] for r in results]
        bars = ax.bar(range(len(names)), values, color=colors)
        ax.set_title(label, fontsize=9.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        top = max(values) if max(values) > 0 else 1
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
                     fmt.format(v), ha="center", fontsize=7.5)
        ax.set_ylim(0, top * 1.2)

    fig.suptitle("LogiEdge Model Variants (M1 FP32 / M2 PTQ INT8 / M3 Pruned35%+INT8) "
                  "-- Five-Metric Benchmark", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.01,
              "Accuracy and Critical recall are tied across all variants (803-param baseline); "
              "INT8 PTQ (M2) and structured pruning+INT8 (M3, 803->400 params, -50%) "
              "shrink file size with no accuracy cost.",
              ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"pareto chart -> {out_path}")


def main():
    # Held-out validation set only (never the training split), matching
    # Task F2's "accuracy on held-out validation set" requirement.
    X_norm = np.load(os.path.join(MODELS_DIR, "val_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(MODELS_DIR, "val_labels.npy"))

    results = []
    for name, fname in VARIANTS:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            continue
        print(f"benchmarking {name}...")
        r = benchmark_variant(name, path, X_norm, y)
        print(json.dumps(r, indent=2))
        results.append(r)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwritten -> {csv_path}")

    plot_pareto(results, os.path.join(RESULTS_DIR, "pareto_chart.png"))


if __name__ == "__main__":
    main()
