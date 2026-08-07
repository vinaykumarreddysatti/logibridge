<div class="bits-header">
<div class="inst">BIRLA INSTITUTE OF TECHNOLOGY &amp; SCIENCE, PILANI</div>
<div class="div">Work Integrated Learning Programmes Division</div>
<div class="course">AIML ZG535 — Machine Learning on Edge · Mini-Project (EC-I)</div>
</div>

# LogiEdge — Phase 1 Report

<div class="subtitle">System Architecture &amp; Hardware Justification (Modules 1–2, Components A &amp; B) — Group 9</div>

<div class="group-table">

| S.No. | Name | BITS ID | Contribution |
|---|---|---|---|
| 1 | CHANDRA SEKAR S | 2024AC05412 | 100% |
| 2 | KULKARNI HARSHAL RAMAKANT | 2024AC05305 | 100% |
| 3 | SATHI V KRISHNA REDDY | 2024AD05379 | 100% |
| 4 | SATTI VINAYKUMARREDDY | 2024AC05160 | 100% |

</div>

## Task A1 — Constraint Analysis

**Latency.** A refrigeration failure raises cargo temperature by
1 °C/minute, and the 90-second detection SLA allows at most a 1.5 °C
rise before an alert must fire. Cloud inference requires a round trip
over rural cellular infrastructure that, on the Nashik–Aurangabad route
alone, loses connectivity entirely for 35–90 minutes at seven documented
locations. During any such gap a cloud-only system cannot receive sensor
data or transmit an alert at all — an outage of up to 90 minutes is sixty
times the SLA, and precisely the failure mode behind the ₹28 lakh vaccine
spoilage incident. On-device TFLite inference completes in well under a
millisecond, so the SLA is trivially met by an edge architecture and
structurally unmeetable by a cloud-dependent one.

**Bandwidth.** At 1 Hz temperature and 500 Hz 3-axis vibration
sampling, a single truck generates roughly 519 MB of raw sensor data per
day, costing approximately ₹52/truck/day (about ₹16.1 lakh/year for the
85-truck pilot) at ₹0.10/MB. Processing at the edge and transmitting only
classification results — one ~110-byte message every 10 seconds — reduces
this to about 0.95 MB/truck/day, roughly 546 times less data and under
₹3,000/year for the whole pilot fleet. Edge processing here is not an
optimisation; it is the difference between a viable and a non-viable
telecom bill.

**Connectivity.** During the documented 35–90 minute signal gaps a
cloud-only system is blind: no ingestion, no inference, no alerting, and
no chain-of-custody records for the very interval where an undetected
failure is most damaging. The LogiEdge edge node classifies every window
locally regardless of connectivity, raises the driver alert immediately,
and appends every Warning/Critical result to a durable local alert log;
once coverage returns, unsynced records are automatically republished to
the operations centre. This store-and-forward behaviour was verified by
seeding an unsynced alert-log entry with the broker unreachable and
confirming automatic republication on reconnect.

**Privacy.** Because inference runs on the truck, raw pharmaceutical
cargo telemetry never leaves the vehicle — only a class label, a
confidence score, and a timestamp are transmitted, over an authenticated
TLS username/password MQTT channel in the production broker profile,
with per-role access-control lists (sensor publisher, inference
consumer, fleet-ops monitor). This lets FreightBridge contractually
demonstrate to its pharmaceutical clients that condition data cannot be
accessed by unauthorised third parties: the raw data never transits a
third-party network at all, and the little that does transit is
role-scoped, encrypted, and auditable.

## Task A2 — System Architecture

![](../scenario_architecture/system_architecture.png)

<div class="caption"><strong>Figure 1.</strong> LogiEdge system architecture: on-vehicle sensors, local Mosquitto
broker, preprocessing + TFLite inference pipeline, durable local alert log with driver alarm,
cellular uplink with store-and-forward sync, and operations-centre backend.</div>

## Task B1 — Constraint Triangle Application

| Option | Performance | Power | Cost (85 / 265 trucks) | Verdict |
|---|---|---|---|---|
| Raspberry Pi 5 + AI HAT+ (13 TOPS) | Ample for an 803-parameter MLP; runs full Docker/MQTT/Ansible stack | 7.5 W — 25% headroom under the 10 W budget | ₹12.75 L / ₹39.75 L | **Selected** |
| Jetson Orin Nano Super (67 TOPS) | 5× the compute this workload can use | 15 W — exceeds the 10 W budget | ₹38.25 L / ₹1.19 Cr | Rejected |
| STM32H7 custom MCU | Cannot run the Docker/MQTT/Ansible MLOps stack | 0.4 W — excellent | ₹2.98 L / ₹9.28 L | Rejected |

The dominant Constraint Triangle vertex for this deployment is
**power-qualified lifecycle cost under a hard reliability
constraint** — not peak performance. The Pi 5 + AI HAT+ meets the
90-second SLA with sub-millisecond inference, sits inside the 10 W
budget with 25% headroom, and costs a third of the Jetson at full
265-truck scale for compute this model does not need. The Jetson's extra
67 TOPS buys nothing for a 6-feature MLP while breaking the power budget
from the 12 V truck supply. The STM32H7 saves capital cost but forfeits
the Docker containerisation, OTA update path, and drift-monitoring
capability the pilot exists to prove — a false economy for a
safety-critical fleet deployment.

## Task B2 — Arithmetic Intensity and Roofline Analysis

For the given workload of 45 MFLOPs and 18 MB accessed per inference,
on the Pi 5 CPU (16 GFLOP/s NEON SIMD, 12 GB/s LPDDR4X):

<div class="narrow-table">

| Quantity | Value |
|---|---|
| Arithmetic Intensity = 45 MFLOP / 18 MB | **2.5 FLOP/byte** |
| Ridge point = 16 GFLOP/s / 12 GB/s | **1.33 FLOP/byte** |
| Classification (2.5 > 1.33) | **Compute-bound** |

</div>

The model sits to the right of the ridge point, so it is
**compute-bound**: additional memory bandwidth would not reduce
latency. The optimisation lever that will is reducing arithmetic per
inference — INT8 quantisation (NEON-accelerated integer maths) and
structured pruning to shrink FLOPs — which is exactly the optimisation
programme implemented in the Phase 2/Final work (Component F).
