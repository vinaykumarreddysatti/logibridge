"""Task E1: PSI drift monitoring on the model's live inference confidence
scores. Subscribes to the truck's inference topic, maintains a rolling
window of the last 100 confidence scores, and reports PSI every 60s
against the frozen reference_dist.json. Prints
"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}" whenever PSI exceeds 0.25.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import deque

import paho.mqtt.client as mqtt

from psi import bin_distribution, compute_psi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from mqtt_security import configure_client_security, default_port  # noqa: E402

HERE = os.path.dirname(__file__)
PSI_ALERT_THRESHOLD = 0.25
ROLLING_WINDOW = 100
CHECK_INTERVAL_S = 60


class DriftMonitor:
    def __init__(self, broker, port, truck_id, reference_path):
        self.truck_id = truck_id
        with open(reference_path) as f:
            ref = json.load(f)
        self.reference_props = ref["proportions"]
        self.bin_edges = ref["bin_edges"]
        self.confidences = deque(maxlen=ROLLING_WINDOW)
        self.lock = threading.Lock()

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"drift-monitor-{truck_id}")
        configure_client_security(self.client)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker, self.port = broker, port

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(f"logibridge/trucks/{self.truck_id}/inference")
        print(f"[drift-monitor] subscribed to truck={self.truck_id}, reference={self.reference_props}")

    def _on_message(self, client, userdata, msg):
        payload = json.loads(msg.payload.decode())
        with self.lock:
            self.confidences.append(payload["confidence"])

    def _check_loop(self):
        while True:
            time.sleep(CHECK_INTERVAL_S)
            with self.lock:
                if len(self.confidences) < 10:
                    print(f"[drift-monitor] insufficient samples ({len(self.confidences)}) -- skipping")
                    continue
                current_props = bin_distribution(list(self.confidences), self.bin_edges)
            psi = compute_psi(self.reference_props, current_props)
            print(f"[drift-monitor] PSI={psi:.3f} (n={len(self.confidences)})")
            if psi > PSI_ALERT_THRESHOLD:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")

    def run(self):
        self.client.connect(self.broker, self.port, keepalive=60)
        threading.Thread(target=self._check_loop, daemon=True).start()
        self.client.loop_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=default_port())
    parser.add_argument("--truck-id", default="TRUCK001")
    parser.add_argument("--reference", default=os.path.join(HERE, "reference_dist.json"))
    args = parser.parse_args()
    DriftMonitor(args.broker, args.port, args.truck_id, args.reference).run()


if __name__ == "__main__":
    main()
