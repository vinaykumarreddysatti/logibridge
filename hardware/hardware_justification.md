# Task B1/B2 -- Hardware Selection and Roofline Analysis (evidence draft)

> Rewrite in your own words for the Final Report -- see the note in `reports/`.

## Task B1 -- Constraint Triangle

| Option | Hardware | Price/truck | TDP | 265-truck fleet cost |
|---|---|---|---|---|
| 1 | Raspberry Pi 5 (8GB) + AI HAT+ (13 TOPS Hailo-8L) | ₹15,000 | 7.5W | ₹39.75 lakh |
| 2 | Jetson Orin Nano Super Dev Kit (67 TOPS) | ₹45,000 | 15W (moderate load) | ₹1.19 crore |
| 3 | STM32H7 custom MCU + sensor ICs | ₹3,500 | 0.4W | ₹9.28 lakh |

**Dominant constraint: power budget + fleet-scale cost, not compute.**
The 90-second latency SLA is generous relative to what any of these three
options need: our benchmarked TFLite MLP inference is sub-millisecond on a
laptop CPU (`optimisation/results/benchmark_results.csv`), and even
allowing an order of magnitude of slowdown for a lower-clocked embedded
core, inference latency is not the bottleneck for a 752-parameter, 6-feature
classifier -- the 10-second feature-window step dominates. What *does*
differentiate the three options is (a) whether they fit the 10W AI power
budget available from the truck's 12V supply via a DC-DC converter, and
(b) unit economics at 85-truck pilot / 265-truck full-fleet scale.

**Recommendation: Option 1 (Raspberry Pi 5 + AI HAT+).**

- **For:** 7.5W leaves 25% headroom under the 10W budget for other
  truck-side electronics sharing the DC-DC rail. At ₹15,000/truck it is
  3x cheaper than the Jetson at full fleet scale (₹39.75L vs ₹1.19Cr --
  a difference of ~₹79 lakh) while comfortably running the full software
  stack this project requires: Docker (Task D2), a co-located MQTT broker,
  Python/TFLite inference, and Ansible-managed OTA (Task E2). A 13 TOPS
  NPU is already vast overprovisioning for a 45 MFLOP model; the point of
  the AI HAT+ is headroom for future scope (e.g. a camera-based cargo
  inspection model), not current necessity.

- **Against Jetson Orin Nano:** 67 TOPS and 15W buys compute this workload
  cannot use. Paying 3x the unit cost for capability the project doesn't
  need is difficult to justify against a mid-sized 3PL's margins,
  especially when the pilot's explicit goal is proving unit economics
  scale to 265 vehicles.

- **Against STM32H7:** cheapest and lowest-power by a wide margin, but it
  cannot run Docker, cannot host a local MQTT broker, and has no
  straightforward path to the Python/TFLite/Ansible toolchain this project
  is built on (Tasks D2, E1, E2 all assume a Linux userspace). Choosing it
  would mean re-architecting the entire deployment/monitoring stack around
  bare-metal C, which is a legitimate embedded design for a next-generation
  product but is not compatible with the pilot's 10-week Docker+Ansible+MQTT
  scope. It also has no realistic upgrade path if the fleet later adds a
  vision or audio model.

## Task B2 -- Arithmetic Intensity and Roofline

Given (assignment-specified): 45 MFLOPs/inference, 18 MB data access/inference,
Raspberry Pi 5 CPU peak compute 16 GFLOP/s (NEON SIMD), 12 GB/s LPDDR4X
bandwidth.

```
Arithmetic Intensity (AI) = FLOPs / Bytes
                          = 45x10^6 / 18x10^6
                          = 2.5 FLOPs/byte

Ridge point = peak_compute / peak_bandwidth
            = 16 GFLOP/s / 12 GB/s
            = 1.333 FLOPs/byte

AI (2.5) > ridge point (1.333)  =>  model is COMPUTE-BOUND
```

At AI = 2.5, the memory-bandwidth roofline would allow
`2.5 x 12 GB/s = 30 GFLOP/s` -- above the CPU's 16 GFLOP/s compute
ceiling. The attainable performance is therefore `min(16, 30) = 16 GFLOP/s`,
set by compute, not memory bandwidth.

**Optimisation implication:** since the workload is compute-bound, the
correct lever is *reducing FLOPs per inference*, not improving memory
locality or prefetching. This is exactly what Tasks F1/F2 do:

- INT8 quantisation (M2) replaces FP32 multiply-accumulates with INT8
  ops, which NEON executes at higher throughput per cycle.
- Structured pruning (M3, 35% sparsity) removes ~35% of the
  multiply-accumulates outright.

Memory-side optimisations (better cache blocking, prefetch tuning) would
have limited effect here, since bandwidth was never the bottleneck at this
AI.

*(Note: the assignment's 45 MFLOP/18 MB figures are a given exercise input,
not our actual trained model's footprint -- our real MLP has only 752
parameters (~3 KB), which is trivially memory-light. The 18 MB figure
plausibly represents a larger reference model such as a small CNN; we use
the assignment's numbers as instructed for the Roofline calculation itself,
and note the discrepancy for transparency.)*
