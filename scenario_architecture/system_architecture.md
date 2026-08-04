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
            DRIFT["PSI Drift Monitor\n(reference_dist.json)"]
        end

        subgraph LOCAL_STORAGE ["Local Storage & Alerting"]
            LOG["Local Alert Log & Store-and-Forward Cache\n(SQLite / Persistent Disk Buffer)"]
            CAB["Cab Audio-Visual Buzzer / Driver Alert"]
        end
    end

    subgraph UPLINK ["3. Network Boundary"]
        CELL["Cellular Uplink Engine\n(4G/3G Auto-Sync Queue, QoS 1)"]
    end

    subgraph CLOUD_BACKEND ["4. Operations Centre Central Cloud"]
        EMQ["Central MQTT Gateway / EMQX Broker"]
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
    INF --> DRIFT
    
    INF -->|Classification Result: Class 0/1/2| MB
    
    MB -->|Publish Class 1/2 Alerts| LOG
    LOG -->|Trigger Immediate Cab Alarm| CAB
    
    LOG -->|Store Unsent Logs during Network Gaps| CELL
    CELL -.->|Retry & Sync when Cellular Coverage Restored| EMQ
    
    EMQ --> DB
    DB --> OPS

    classDef sensor fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef edge fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef cloud fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

    class S1,S2,S3 sensor;
    class MB,MA,FE,NORM,INF,DRIFT edge;
    class LOG,CAB storage;
    class EMQ,DB,OPS cloud;