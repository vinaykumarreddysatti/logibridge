# Component A: System Architecture and Deployment Justification

# Task A1 — Constraint Analysis

## 1. Latency Constraint

FreightBridge's refrigerated trucks transport temperature-sensitive pharmaceuticals where refrigeration failures can increase cargo temperature by approximately **1°C per minute**. The system must therefore detect a fault signature and generate an alert within a strict **90-second Service Level Agreement (SLA)**.

### Cloud Feasibility

A cloud-only inference architecture is **not suitable** for this deployment. Rural logistics routes across Maharashtra and Andhra Pradesh experience variable cellular latency, packet retransmissions, and complete coverage loss. Even when network connectivity exists, cloud inference requires:

1. Uploading sensor data to the cloud
2. Server-side processing
3. Returning the inference result to the truck
4. Triggering an operational alert

Although normal cellular round-trip latency may be only a few hundred milliseconds, network congestion and intermittent connectivity make it impossible to **guarantee** the required 90-second SLA. During complete network outages, cloud inference becomes unavailable altogether.

In contrast, an edge device performs preprocessing and inference locally. A 30-second feature window can be processed in well under one second (typically **<50 ms** on modern edge hardware), allowing immediate driver notification while simultaneously storing the alert for later synchronisation. Therefore, only Edge AI can consistently satisfy the required latency constraint.

---

# 2. Bandwidth & Transmission Cost Constraint

Each refrigerated truck continuously generates sensor data from temperature, vibration, and door sensors.

## Raw Data Calculation (Per Truck Per Day)

| Sensor | Sampling Rate | Assumption | Daily Data |
|---------|---------------|------------|-----------:|
| Temperature | 1 Hz | 4-byte float | 345.6 KB |
| Vibration (3-axis) | 500 Hz | 12 bytes/sample (X,Y,Z) | 518.4 MB |
| Door Events | ~50/day | 64 bytes/event | 3.2 KB |

### Calculation

**Temperature**

```text
1 sample/s × 4 bytes × 86,400 s
= 345,600 bytes
≈ 345.6 KB/day
```

**Vibration**

Each vibration sample contains three axes (X, Y and Z):

```text
500 samples/s × 12 bytes/sample × 86,400 s
= 518,400,000 bytes
≈ 518.4 MB/day
```

**Total Raw Data**

```text
≈ 518.75 MB per truck per day
```

## Cellular Transmission Cost

At **₹0.10 per MB**:

| Deployment | Calculation | Cost |
|------------|-------------|-----:|
| One Truck | 518.75 × ₹0.10 | **₹51.88/day** |
| Pilot Fleet (85 trucks) | ₹51.88 × 85 × 30 | **₹132,294/month** |
| Full Fleet (265 trucks) | ₹51.88 × 265 × 30 | **₹412,446/month** |

## Edge-Processed Data Comparison

Instead of continuously transmitting raw sensor streams, the edge device performs local feature extraction and classification. Only periodic health summaries and anomaly alerts are transmitted.

Assuming:

- Status message = **200 bytes/minute**
- 1,440 messages/day

```text
200 × 1,440
= 288,000 bytes
≈ 288 KB/day
≈ 0.288 MB/day
```

Monthly communication cost for the pilot fleet:

```text
0.288 × ₹0.10 × 85 × 30
≈ ₹73.44/month
```

Compared with transmitting raw sensor streams, Edge AI reduces communication bandwidth and cellular cost by **more than 99.9%**, making fleet-scale deployment economically practical.

---

# 3. Connectivity Constraint

FreightBridge's Nashik–Aurangabad route contains **seven documented cellular coverage gaps**, each lasting approximately **35–90 minutes**.

## Cloud-Only Failure Mode

A cloud-dependent monitoring system becomes completely unavailable during these connectivity gaps.

For example, if the refrigeration compressor fails five minutes after entering a 90-minute dead zone:

- No telemetry reaches the cloud.
- No inference is performed.
- No warning is generated.
- Cargo temperature continues rising unchecked.

This could easily reproduce the previous **₹28 lakh vaccine spoilage incident** described in the business case.

## Edge AI Solution

The proposed LogiEdge architecture performs sensing, preprocessing and inference entirely on the truck's embedded edge node.

When a critical anomaly is detected:

1. A local visual/audio alarm immediately notifies the driver.
2. The alert and supporting sensor metadata are stored in a **persistent local alert log** (implemented using SQLite).
3. Once cellular connectivity returns, the stored alerts are automatically synchronised with the operations centre through MQTT (recommended using QoS 1 to ensure reliable delivery).

This design guarantees uninterrupted monitoring regardless of network availability.

---

# 4. Privacy & Chain-of-Custody Constraint

FreightBridge's pharmaceutical customers require assurance that cargo-condition information remains protected from unauthorised access while maintaining an auditable chain of custody.

A cloud-centric architecture continuously uploads raw high-frequency telemetry, increasing exposure to interception, unauthorised access, and unnecessary transfer of commercially sensitive operational data such as delivery schedules, routing information and cargo handling conditions.

The proposed Edge AI architecture performs all preprocessing and model inference directly on the truck. Raw sensor streams remain within the vehicle and are **never continuously transmitted** over public cellular networks.

Only the following information is uploaded:

- Classified cargo state (Normal, Warning or Critical)
- Timestamped alert events
- Health summaries
- Audit logs required for compliance

All transmitted data is protected using TLS encryption, while local processing significantly reduces the attack surface. This architecture supports pharmaceutical confidentiality requirements, strengthens contractual chain-of-custody compliance, and provides secure, verifiable audit records without exposing raw operational telemetry.

---

# Conclusion

The FreightBridge deployment is constrained by four critical Edge AI factors: **latency, bandwidth, connectivity, and privacy**. Cloud-only inference cannot satisfy the required 90-second response time, incurs substantial communication costs (over **₹132,000 per month** for the pilot fleet), and fails during prolonged cellular outages. By performing local inference on the truck, the proposed LogiEdge architecture provides real-time decision-making, reduces communication bandwidth by over **99.9%**, continues operating during network outages, and protects sensitive pharmaceutical telemetry through on-device processing. These characteristics make Edge AI the most appropriate architecture for reliable, scalable cold-chain fleet monitoring.