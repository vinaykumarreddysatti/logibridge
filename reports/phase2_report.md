<div class="bits-header">
<div class="inst">BIRLA INSTITUTE OF TECHNOLOGY &amp; SCIENCE, PILANI</div>
<div class="div">Work Integrated Learning Programmes Division</div>
<div class="course">AIML ZG535 — Machine Learning on Edge · Mini-Project (EC-I)</div>
</div>

# LogiEdge — Phase 2 Report

<div class="subtitle">Sensor Pipeline, Model Training &amp; Docker Deployment (Modules 3–4, Components C &amp; D) — Group 9</div>

<div class="group-table">

| S.No. | Name | BITS ID | Contribution |
|---|---|---|---|
| 1 | CHANDRA SEKAR S | 2024AC05412 | 100% |
| 2 | KULKARNI HARSHAL RAMAKANT | 2024AC05305 | 100% |
| 3 | SATHI V KRISHNA REDDY | 2024AD05379 | 100% |
| 4 | SATTI VINAYKUMARREDDY | 2024AC05160 | 100% |

</div>

## Task C1 — Sensor Simulator and MQTT Architecture

`data_pipeline/simulator.py` generates the three cold-chain sensor
streams — temperature at 1 Hz (N(4.0 °C, 0.3), setpoint 4 °C),
vibration RMS at 0.5 Hz (N(0.45 g, 0.05)), and discrete door
OPEN/CLOSE events — publishing all of them to a local Mosquitto broker.
The required CLI flag `--anomaly {none|temp_drift|vibration|combined}`
selects the scenario: linear temperature drift of +0.08 °C per reading,
a vibration step to N(1.2 g, 0.15) simulating bearing wear, or both
simultaneously. A dedicated retained `control/anomaly` topic lets the
scenario be switched live on a running simulator, which is how drift is
injected mid-run in the PSI drift-monitoring demo.

```
logibridge/trucks/{truck_id}/
├── sensors/temperature      1 Hz          QoS 0
├── sensors/vibration        0.5 Hz        QoS 0
├── sensors/door             event         QoS 1
├── control/anomaly          event         QoS 1  (retained)
└── inference                ~0.1 Hz       QoS 1
```

<div class="caption"><strong>Figure 1.</strong> MQTT topic tree with per-topic frequency and QoS.</div>

**QoS rationale.** High-frequency sensor streams use QoS 0 — the
30-second windowed statistics downstream are robust to an occasional
dropped sample, so acknowledging ~130,000 daily messages buys nothing.
Door events and inference results use QoS 1: both are low-frequency and
operationally significant (a missed Critical result reproduces the
vaccine-spoilage failure mode; a missed door event undermines
chain-of-custody documentation), and duplicate delivery is harmless
because the receiving logs are idempotent.

## Task C2 — Preprocessing Pipeline and Normalisation Experiment

The pipeline applies a 5-sample moving average to the temperature and
vibration streams, then extracts a six-value feature vector per
30-second sliding window (10-second step): temperature mean, standard
deviation, and rate-of-change (°C/min); vibration RMS, peak, and
kurtosis. Normalisation statistics are computed exactly once from ten
minutes of clean Normal-class output, frozen to
`training_stats.npy`, and loaded — never recomputed — at runtime.

**Mandatory stats-shift experiment.** Inference with correct stats
vs stats shifted by 3σ produced *no accuracy change* (98.31% both ways) —
a real, explainable result rather than a null test. Normalisation stats
derive only from the tight clean-Normal distribution (temperature std ≈
0.05 °C after window averaging), so Warning/Critical windows already sit
tens to hundreds of standard deviations from that mean by construction,
and a further 3σ shift in the stats file is negligible against that
separation. A supplementary sweep confirms the mechanism is genuine —
degradation appears at 10σ and becomes severe by 20–80σ:

<div class="narrow-table">

| Stats shift (× σ) | Held-out accuracy | Change |
|---|---|---|
| 0 (correct stats) | 98.31% | baseline |
| 3 (mandatory experiment) | 98.31% | 0.00 pp |
| 5 | 98.31% | 0.00 pp |
| 10 | 96.61% | −1.70 pp |
| 20 | 69.49% | −28.82 pp |
| 40 | 67.80% | −30.51 pp |
| 80 | 40.68% | −57.63 pp |

</div>

<div class="caption"><strong>Table 1.</strong> Normalisation stats-shift experiment (<code>data_pipeline/stats_shift_experiment.py</code>).</div>

## Task C3 — Data Fusion Justification

LogiEdge uses **feature-level fusion**: each stream is windowed and
feature-extracted independently, then concatenated into one six-value
vector before the model sees it. Data-level fusion was rejected because
raw streams at different rates (1 Hz vs 0.5 Hz) are not naturally
aligned and would require buffering and resampling of high-rate raw data
on the edge node. Decision-level fusion was rejected because separate
temperature and vibration classifiers combined at the output would lose
the cross-signal correlation — a temperature excursion coinciding with a
vibration anomaly — that defines the Critical class in the
`combined` scenario, while adding a second model and a
conflict-resolution policy to maintain.

## Task D1 — Dataset Generation and Model Training

<div class="narrow-table">

| Class | Mode | Duration | Windows | temp_mean range |
|---|---|---|---|---|
| 0 Normal | `none` | 20 min | 117 | 3.88 – 4.10 °C |
| 1 Warning | `temp_drift` | 15 min | 87 | 5.04 – 12.12 °C |
| 2 Critical | `combined` | 15 min | 87 | 5.16 – 12.13 °C |
| **Total** | | | **291** | |

</div>

<div class="caption"><strong>Table 2.</strong> Generated dataset (80/20 stratified split → 232 training / 59 validation windows).</div>

The classifier is the recommended two-hidden-layer MLP —
Input(6) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(3, softmax) —
trained to **98.31% held-out validation accuracy**, clearing the
mandatory 88% gate, with per-class recall of 100% (Normal, n=24), 94.44%
(Warning, n=18), and **100% (Critical, n=17)**.

## Task D2 — Docker Containerisation and OTA Layer-Cache Demo

The inference service (preprocessing → TFLite inference → MQTT publish
to `logibridge/trucks/{truck_id}/inference`) is packaged on
`python:3.11-slim` with every pip-install layer before the
`COPY inference/model.tflite` instruction, and the model variant
switchable via the `MODEL_PATH` environment variable without
rebuild. The layer-cache demonstration was performed with a real build:
the full image is 1.61 GB (dominated by the ~1.46 GB TensorFlow layer)
while the model layer is 4.62 KB; after swapping only the model file,
every preceding layer reported `CACHED` and only the final COPY
layer rebuilt. For an 85-truck OTA cycle this is the difference between
moving ~136.85 GB (≈₹14,013) and ~0.38 MB (≈₹0.04) — roughly
**349,000× less data and cost**.
