# Component A — Task A2: System Architecture Design

## LogiEdge System Architecture Overview

The **LogiEdge** platform implements a decoupled, event-driven edge intelligence architecture designed for offline-first operational resilience on FreightBridge refrigerated trucks.

---

## 1. System Architecture Diagram

```mermaid
flowchart TB
    subgraph SENSORS ["1. On-Vehicle Physical Sensors"]
        S1["Temperature Sensor\n(1 Hz)"]
        S2["3-Axis Accelerometer\n(500 Hz Vibration)"]
        S3["Door State Switch\n(Discrete Event)"]
    end

    subgraph EDGE_NODE ["2. Truck Edge Compute Node (On-Vehicle)"]
        direction TB
        
        subgraph BROKER ["Local Message Broker"]
            MB["Mosquitto MQTT Broker\n(Localhost:1883)"]
        end

        subgraph INFERENCE_PIPELINE ["Edge Inference & Data Pipeline"]
            MA["5-Sample Moving Average Filter"]
            FE["Sliding Window Feature Extractor\n(30s Window / 10s Step -> 6 Features)"]
            NORM["Z-score Normalization\n(training_stats.npy)"]
            INF["TFLite Inference Engine\n(M1 / M2 / M3 Models)"]
        end

        subgraph LOCAL_STORAGE ["Local Storage & Alerting"]
            LOG["Local Alert Log & Store-and-Forward Cache\n(append-only alert_log.jsonl on disk)"]
            CAB["Cab Audio-Visual Buzzer / Driver Alert"]
        end
    end

    subgraph UPLINK ["3. Network Boundary"]
        CELL["Cellular Uplink Engine\n(4G/3G Auto-Sync Queue, QoS 1)"]
    end

    subgraph CLOUD_BACKEND ["4. Operations Centre Central Cloud"]
        EMQ["Central MQTT Gateway / EMQX Broker"]
        DRIFT["PSI Drift Monitor\n(reference_dist.json, drift_monitor.py)"]
        DB["Time-Series DB / Fleet Monitoring Dashboard"]
        OPS["Operations Center Incident Management"]
    end

    %% Flow Connections
    S1 -->|Raw Temp Data| MB
    S2 -->|Raw Vibration Data| MB
    S3 -->|Door Open/Close Event| MB

    MB -->|Subscribe Raw Streams| MA
    MA --> FE
    FE --> NORM
    NORM --> INF
    
    INF -->|Classification Result: Class 0/1/2, every window| MB
    
    MB -->|Publish Warning/Critical only| LOG
    LOG -->|Trigger Immediate Cab Alarm| CAB
    
    LOG -->|Store Unsent Logs during Network Gaps| CELL
    CELL -.->|Retry & Sync when Cellular Coverage Restored| EMQ

    EMQ -->|Subscribe to inference topic, ops-side| DRIFT
    EMQ --> DB
    DB --> OPS

    classDef sensor fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef edge fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef cloud fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    class S1,S2,S3 sensor;
    class MB,MA,FE,NORM,INF edge;
    class LOG,CAB storage;
    class EMQ,DRIFT,DB,OPS cloud;
```

## 2. Diagram-to-implementation notes

- **Local alert log:** `alert_log.jsonl`, an append-only JSON-lines file
  written by `inference/inference_service.py::_log_alert`, not SQLite --
  corrected here to match the code.
- **PSI Drift Monitor placement:** drawn on the operations-centre side,
  not inside the truck's on-vehicle pipeline. `monitoring/drift_monitor.py`
  is a separate process that subscribes to the truck's `inference` MQTT
  topic; it never runs inside `inference/inference_service.py` or the
  Docker container. This also matches the role-based MQTT ACL design in
  `data_pipeline/mqtt_architecture.md`, where drift monitoring is
  explicitly the `logibridge_ops` role (fleet monitoring), distinct from
  the `logibridge_inference` role that runs on the truck.
- **Inference topic traffic:** every classified window (Class 0, 1, or 2)
  is published to the truck's `inference` MQTT topic -- this is what the
  drift monitor consumes for its rolling PSI window, and it needs Normal-
  class traffic too, not just alerts. Only Warning/Critical results are
  additionally written to the local persistent alert log for
  offline-durable delivery.