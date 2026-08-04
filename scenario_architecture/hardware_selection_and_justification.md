# Component B — Hardware Selection and Roofline Analysis

## Task B1 — Constraint Triangle Application & Hardware Justification

The **Edge AI Constraint Triangle** balances three interdependent trade-offs: **Compute Power (Performance/SLA)**, **Power Consumption (Energy/TDP)**, and **Unit Cost (CAPEX at scale)**.

![Constraint Triangle](constraint_triangle.svg)

---

### 1. Hardware Options Comparison Matrix

| Option | Hardware | Unit Price (India) | TDP (Power) | Pilot Cost (85 Trucks) | Full Fleet Cost (265 Trucks) | SLA Latency Compliance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1** | **Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS)** | **~₹15,000** | **7.5W** | **₹12,75,000** | **₹39,75,000** | **Fully Compliant** (<10s detection) |
| **Option 2** | Jetson Orin Nano Super Dev Kit (67 TOPS) | ~₹45,000 | 15W (moderate) | ₹38,25,000 | ₹1,19,25,000 | Exceeds requirement, but violates power & cost |
| **Option 3** | STM32H7 Custom MCU + Sensor ICs | ~₹3,500 | 0.4W | ₹2,97,500 | ₹9,27,500 | Fails SLA (insufficient RAM for signal processing) |

---

### 2. Dominant Constraint Vertex Selection
**The Dominant Constraint Vertex for FreightBridge’s cold-chain fleet is Power Consumption (Thermal & Electrical Enclosure Budget ≤ 10W) combined with Cost at Scale.**



* **Power Constraint Enforcement**: Operating off a 12V DC-DC vehicle auxiliary power supply inside an enclosed, IP67-rated sealed truck dashboard compartment mandates a thermal/electrical budget of ≤ 10W. Exceeding 10W risks thermal throttling under summer cabin temperatures (up to 50°C in Maharashtra) or continuous vehicle battery drain during long reefer engine-off idling.
* **Cost Scaling Enforcement:** Scaling from the 85-truck pilot to all 265 vehicles requires tight capital expenditure control.

---

### 3. Hardware Selection Argument

#### Chosen Selection: Option 1 — Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS)
* **Power Budget Fit:** Operates at **7.5W TDP**, comfortably beneath the strict **10W power budget**.
* **Latency SLA Compliance:** The quad-core ARM Cortex-A76 CPU paired with the 13 TOPS Hailo-8L NPU executes feature extraction and model inference in **< 15 ms**, easily fulfilling the 90-second detection SLA.
* **Fleet Financial Justification:** At **~₹15,000 per truck**, the total pilot cost is **₹12.75 Lakhs**, and full 265-truck fleet rollout costs **₹39.75 Lakhs**. This offers an optimal balance between affordability and local compute capability.

#### Rejected Options:
* **Argument Against Option 2 (Jetson Orin Nano):** While highly capable (67 TOPS), its **15W power consumption** exceeds the 10W thermal/power envelope, risking battery drain and overheating inside sealed truck enclosures. Furthermore, at **~₹45,000 per unit**, scaling to 265 trucks costs **₹1.19 Crores** (~3x Option 1), representing prohibitive over-engineering for a 6-feature classification task.
* **Argument Against Option 3 (STM32H7 MCU):** Although extremely low power (0.4W) and cheap (~₹3,500), its **1 MB SRAM and 480 MHz CPU** are insufficient to handle 500 Hz 3-axis continuous vibration feature extraction/sliding window buffering alongside local Docker/Mosquitto containerized pipelines. It lacks the memory capacity to reliably execute multi-modal feature fusion and risk detection within the 90-second SLA.

---

## Task B2 — Arithmetic Intensity and Roofline Analysis

### 1. Mathematical Formulas & Calculations
#### A. Arithmetic Intensity (I)
Arithmetic Intensity measures the operational density of a model in FLOPs per Byte of memory accessed:

I = Total Operations (FLOPs) / Total Memory Accessed (Bytes)

Given:

Total Operations = 45 MFLOPs = 45 × 10^6 FLOPs

Total Memory Access (Weights + Activations) = 18 MB = 18 × 10^6 Bytes

I = (45 × 10^6 FLOPs) / (18 × 10^6 Bytes) = 2.50 FLOP / Byte

#### B. Roofline Ridge Point (I_ridge)
The Ridge Point defines the boundary on the Roofline Model where performance transitions between memory-bandwidth bound and compute-bound regimes:

I_ridge = Peak Compute Performance / Memory Bandwidth

Given Raspberry Pi 5 CPU Specs:

Peak Compute Performance = 16 GFLOP/s = 16 × 10^9 FLOP/s (NEON SIMD)

Memory Bandwidth = 12 GB/s = 12 × 10^9 Bytes/s (LPDDR4X)

I_ridge = (16 × 10^9 FLOP/s) / (12 × 10^9 Bytes/s) = 1.3333 FLOP / Byte

### 2. Model Classification & Roofline Interpretation
 Performance
 ```
 (GFLOP/s)
   16.0 |----------------------------+==================== (Compute Bound Ceiling: 16 GFLOP/s)
        |                           /
        |                          /  <- Model Operating Point (I = 2.50 FLOP/Byte)
        |                         /
        |                        /
        |                       /
        |                      /
        |                     /|
        |____________________/_|________________________
        0                   1.33 2.50                  Arithmetic Intensity
                             (I_ridge)                 (FLOP/Byte)
```
**Classification:** Since the model's Arithmetic Intensity (I = 2.50 FLOP/Byte) is greater than the Ridge Point (I_ridge = 1.33 FLOP/Byte), the baseline model is Compute-Bound (operating on the horizontal ceiling of the Roofline Curve).

**Performance Ceiling:** The CPU is execution-bound by arithmetic execution throughput rather than memory transfer speeds.

### 3. Roofline-Guided Optimization Strategy
To improve inference latency and reduce compute requirements according to the Roofline Model:

**Model Quantization (FP32 → INT8):** Converting weights and activations to INT8 reduces total memory footprint from 18 MB to ≈ 4.5 MB while leveraging SIMD vector processing (NEON).

**Structured Pruning:** Pruning inactive filter weights by 35% reduces the absolute operation count (MFLOPs) from 45 MFLOPs to ≈ 29.25 MFLOPs, directly dropping execution cycle requirements on the compute-bound ceiling and reducing overall latency.