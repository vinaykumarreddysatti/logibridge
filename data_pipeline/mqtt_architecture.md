# LogiEdge MQTT Topic Tree and QoS Design

## Topic tree

```
logibridge/
└── trucks/
    └── {truck_id}/                      e.g. TRUCK001
        ├── sensors/
        │   ├── temperature               1 Hz,   QoS 0
        │   ├── vibration                  0.5 Hz, QoS 0
        │   └── door                       event,  QoS 1
        ├── control/
        │   └── anomaly                    event,  QoS 1, retain=true
        └── inference                      ~0.1 Hz (10s step), QoS 1
```

Every message is a JSON object. Examples:

- `sensors/temperature`: `{"truck_id": "TRUCK001", "timestamp": 1731...,  "value": 4.12}`
- `sensors/vibration`: `{"truck_id": "TRUCK001", "timestamp": 1731...,  "rms": 0.47}`
- `sensors/door`: `{"truck_id": "TRUCK001", "timestamp": 1731...,  "event": "OPEN"}`
- `control/anomaly`: raw string payload, one of `none|temp_drift|vibration|combined`
- `inference`: `{"truck_id": "TRUCK001", "timestamp": 1731..., "class": 2, "label": "Critical", "confidence": 0.91}`

## QoS justification per topic

| Topic | QoS | Why |
|---|---|---|
| `sensors/temperature`, `sensors/vibration` | 0 (fire-and-forget) | High-frequency streams (1 Hz / 0.5 Hz). The Task C2 preprocessing pipeline computes 30s windowed statistics (mean, std, RMS, kurtosis) that are robust to an occasional dropped sample. Paying the ACK round-trip cost of QoS 1/2 on every one of ~130,000 daily messages buys negligible reliability improvement for a value that is about to be smoothed by a moving average anyway. |
| `sensors/door` | 1 (at-least-once) | Discrete, operationally significant events used for chain-of-custody documentation (the pharma-shipment-rejection incident in Section 1 was caused by *missing* documentation of this kind). Loss is not acceptable; duplicate delivery is harmless since OPEN/CLOSE events are idempotent to log. |
| `control/anomaly` | 1, retained | A late-joining or reconnecting subscriber (e.g. the simulator restarting) must pick up the last commanded anomaly mode rather than silently defaulting to `none`. |
| `inference` | 1 (at-least-once) | This is the topic the drift monitor and the ops-centre backend depend on for Warning/Critical alerts. A missed Critical classification is the exact failure mode that caused the ₹28 lakh vaccine spoilage incident. Duplicate delivery is harmless (the local `alert_log.jsonl` is keyed on timestamp and dedupes trivially). |

## Broker security: dev vs. production

Two Mosquitto configs are provided, and they are **not interchangeable**:

- `deployment/mosquitto_local.conf` -- **development/testing only**.
  `allow_anonymous true`, plaintext port 1883. It provides no
  authentication and must never be used on an actual truck.
- `deployment/mosquitto_production.conf` -- TLS-only (port 8883,
  `tls_version tlsv1.2`), `allow_anonymous false`, per-client
  username/password auth (`mosquitto_passwd.txt`), and a topic-scoped ACL
  (`mosquitto_acl.conf`) with **three least-privilege roles, not one
  shared credential**: `logibridge_sensors` (write sensors/#, read
  control/anomaly -- what the simulator needs), `logibridge_inference`
  (read sensors/#, write inference -- what the inference service needs),
  and `logibridge_ops` (read everything, write control/anomaly -- fleet
  monitoring and the drift monitor). Regenerate credentials with
  `deployment/generate_mqtt_credentials.sh` (outputs are gitignored).

**This is wired into the actual Python clients, not just the broker
config.** `simulator.py`, `inference_service.py`, and `drift_monitor.py`
all go through the shared `data_pipeline/mqtt_security.py` helper, which
reads `MQTT_USERNAME` / `MQTT_PASSWORD` / `MQTT_TLS_CA` /
`MQTT_TLS_INSECURE` from the environment and configures
`username_pw_set()` / `tls_set()` on the paho client accordingly. With
none of those set, a client behaves exactly as before (plaintext,
anonymous, port 1883) -- every test elsewhere in this repo runs that way,
for convenience. Setting them switches a client to the production
profile.

This was run for real, end to end, not just asserted: with
`mosquitto_production.conf` live, `simulator.py` (as `logibridge_sensors`),
`inference_service.py` (as `logibridge_inference`), and `drift_monitor.py`
(as `logibridge_ops`) were started simultaneously against
`localhost:8883` with `MQTT_TLS_CA` pointed at the generated CA cert. The
broker log confirms all three negotiated `TLSv1.3` under their own
distinct identity, and classification results flowed end to end exactly
as in the plaintext dev tests (Normal -> Critical as `combined`-mode
drift matured). Two negative tests then confirmed the ACLs are actually
enforced, not just declared: `logibridge_sensors` attempting to publish
to the `inference` topic and `logibridge_inference` attempting to publish
to `sensors/temperature` were both rejected, logged by the broker as
`Denied PUBLISH`. The "on-device inference over an authenticated channel"
privacy claim in `scenario_architecture/constraint_analysis.md` refers to
this profile, exercised exactly as described here.

## Offline tolerance

The inference service (`inference/inference_service.py`) never blocks on the
broker: sensor windows are processed and classified locally regardless of
connection state. Warning/Critical results are always appended to a local
`alert_log.jsonl` with a `synced` flag; on (re)connect, any unsynced entries
are replayed to the `inference` topic before live traffic resumes. This is
what makes the offline-during-connectivity-gap requirement (Section 2,
"⚠ IMPORTANT") concrete rather than aspirational -- verified in
`inference/inference_service.py::_flush_alert_log` and exercised in testing
by starting the service against a pre-seeded log entry marked `synced: false`
with no simulator running, and confirming it is marked `synced: true` and
re-published as soon as the broker becomes reachable.
