"""LogiEdge inference microservice (Task D2).

Subscribes to a truck's sensor topics, runs the Task C2 preprocessing
pipeline on rolling buffers, performs on-device TFLite inference,
classifies the cargo compartment state, and publishes the result.
Warning/Critical results are appended to a local, offline-durable log
and re-published once MQTT connectivity is available -- so no violation
is lost during a cellular connectivity gap, satisfying the "no
dependency on continuous cellular connectivity" requirement.
"""
import json
import os
import sys
import time
import threading
from collections import deque

import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))  # repo layout
sys.path.append(os.path.join(os.path.dirname(__file__), "data_pipeline"))        # container layout
from preprocessing import extract_features, load_stats, normalise  # noqa: E402
from mqtt_security import configure_client_security, default_port  # noqa: E402

MODEL_PATH = os.environ.get("MODEL_PATH", "model.tflite")
STATS_PATH = os.environ.get("STATS_PATH", "training_stats.npy")
BROKER = os.environ.get("MQTT_BROKER", "localhost")
PORT = int(os.environ.get("MQTT_PORT", str(default_port())))
TRUCK_ID = os.environ.get("TRUCK_ID", "TRUCK001")
ALERT_LOG = os.environ.get("ALERT_LOG", "alert_log.jsonl")

WINDOW_S = 30.0
STEP_S = 10.0
CLASS_NAMES = {0: "Normal", 1: "Warning", 2: "Critical"}


class LogiEdgeInferenceService:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.is_int8 = self.input_details["dtype"] == np.int8
        self.mean, self.std = load_stats(STATS_PATH)

        self.temp_buf = deque()
        self.vib_buf = deque()
        self.last_window_end = None
        self.lock = threading.Lock()

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"inference-{TRUCK_ID}")
        configure_client_security(self.client)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False

    def _topic(self, suffix):
        return f"logibridge/trucks/{TRUCK_ID}/{suffix}"

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        self.connected = True
        client.subscribe(self._topic("sensors/temperature"))
        client.subscribe(self._topic("sensors/vibration"))
        print(f"[inference] connected, model={MODEL_PATH}, int8={self.is_int8}")
        self._flush_alert_log()

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        ts = payload["timestamp"]
        with self.lock:
            if msg.topic.endswith("/temperature"):
                self.temp_buf.append((ts, payload["value"]))
            elif msg.topic.endswith("/vibration"):
                self.vib_buf.append((ts, payload["rms"]))
            self._trim_buffers()
            self._maybe_infer()

    def _trim_buffers(self):
        cutoff = time.time() - WINDOW_S - 5
        while self.temp_buf and self.temp_buf[0][0] < cutoff:
            self.temp_buf.popleft()
        while self.vib_buf and self.vib_buf[0][0] < cutoff:
            self.vib_buf.popleft()

    def _maybe_infer(self):
        if not self.temp_buf:
            return
        latest = self.temp_buf[-1][0]
        if self.last_window_end is not None and latest - self.last_window_end < STEP_S:
            return
        window_start = latest - WINDOW_S
        temp_pts = [(t, v) for t, v in self.temp_buf if t >= window_start]
        vib_pts = [(t, v) for t, v in self.vib_buf if t >= window_start]
        if len(temp_pts) < 10 or len(vib_pts) < 5:
            return

        temp_ts = np.array([p[0] for p in temp_pts])
        temp_vals = np.array([p[1] for p in temp_pts])
        vib_ts = np.array([p[0] for p in vib_pts])
        vib_vals = np.array([p[1] for p in vib_pts])

        _, feats = extract_features(temp_ts, temp_vals, vib_ts, vib_vals)
        if len(feats) == 0:
            return
        feat = feats[-1:].astype(np.float32)
        feat_norm = normalise(feat, self.mean, self.std)

        pred_class, confidence = self._run_tflite(feat_norm)
        self.last_window_end = latest
        self._publish_result(pred_class, confidence, latest)

    def _run_tflite(self, feat_norm):
        x = feat_norm
        if self.is_int8:
            scale, zero_point = self.input_details["quantization"]
            x = (x / scale + zero_point).round().astype(np.int8)
        else:
            x = x.astype(np.float32)
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])
        if self.output_details["dtype"] == np.int8:
            scale, zero_point = self.output_details["quantization"]
            out = (out.astype(np.float32) - zero_point) * scale
        probs = out[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        return pred_class, confidence

    def _publish_result(self, pred_class, confidence, ts):
        result = {
            "truck_id": TRUCK_ID,
            "timestamp": ts,
            "class": pred_class,
            "label": CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4),
        }
        line = json.dumps(result)
        print(f"[inference] {line}")

        published = False
        if self.connected:
            try:
                info = self.client.publish(self._topic("inference"), line, qos=1)
                published = info.rc == mqtt.MQTT_ERR_SUCCESS
            except Exception:
                published = False

        if pred_class in (1, 2):
            self._log_alert(result, synced=published)

    def _log_alert(self, result, synced):
        entry = dict(result)
        entry["synced"] = synced
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _flush_alert_log(self):
        """On (re)connect, re-publish any alert entries written while
        offline, so a connectivity gap never loses a Warning/Critical
        alert -- this is the chain-of-custody guarantee for pharma
        clients (Task A1 privacy/compliance requirement)."""
        if not os.path.exists(ALERT_LOG):
            return
        lines = open(ALERT_LOG).readlines()
        changed = False
        out_lines = []
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            if not entry.get("synced"):
                try:
                    payload = {k: v for k, v in entry.items() if k != "synced"}
                    self.client.publish(self._topic("inference"), json.dumps(payload), qos=1)
                    entry["synced"] = True
                    changed = True
                except Exception:
                    pass
            out_lines.append(json.dumps(entry))
        if changed:
            with open(ALERT_LOG, "w") as f:
                f.write("\n".join(out_lines) + "\n")

    def run(self):
        while True:
            try:
                self.client.connect(BROKER, PORT, keepalive=60)
                break
            except Exception as e:
                print(f"[inference] broker unreachable ({e}), retrying in 5s -- operating offline")
                time.sleep(5)
        self.client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    LogiEdgeInferenceService().run()
