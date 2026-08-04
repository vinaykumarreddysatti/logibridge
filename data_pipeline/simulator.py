#!/usr/bin/env python3
"""LogiEdge cold-chain sensor simulator (Task C1).

Publishes three sensor streams for a simulated refrigerated truck to a
local Mosquitto broker: temperature (1 Hz), vibration_rms (0.5 Hz) and
discrete door_event messages. Supports injectable anomaly modes via
--anomaly, and the anomaly mode can also be switched at runtime by
publishing to the truck's control/anomaly topic (used to demonstrate
live drift injection for Task E1).

The statistical sample generator (ColdChainSensorGenerator) is kept
separate from the MQTT publishing loop so the exact same sampling logic
can be reused, without wall-clock sleeps, for fast offline dataset
generation (see generate_offline_series, used by
training/generate_dataset.py).
"""
import argparse
import json
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from mqtt_security import configure_client_security, default_port

ANOMALY_MODES = ("none", "temp_drift", "vibration", "combined")

SETPOINT_C = 4.0
TEMP_NOISE_STD = 0.3
TEMP_DRIFT_PER_SAMPLE = 0.08  # deg C per 1 Hz reading, linear drift
# Physical ceiling on accumulated drift: linear drift at 0.08C/reading run
# for a full 15-minute Warning scenario (Task D1) would otherwise reach
# ~70C above setpoint, which is not physically plausible for a
# refrigerated cargo compartment (it cannot exceed ambient air
# temperature) and would push most of a "Warning" run's windows past the
# Critical threshold before the run even finishes. Capping at 8C keeps
# the scenario clearly and increasingly faulty without runaway values.
MAX_TEMP_DRIFT_C = 8.0

VIB_NORMAL_MEAN, VIB_NORMAL_STD = 0.45, 0.05
VIB_ANOMALY_MEAN, VIB_ANOMALY_STD = 1.20, 0.15  # bearing wear step

DOOR_EVENTS_PER_HOUR = 6  # ~ one open/close pair every 10 minutes


class ColdChainSensorGenerator:
    """Pure statistical sample generator, decoupled from I/O and timing."""

    def __init__(self, anomaly_mode="none", setpoint=SETPOINT_C, seed=None):
        if anomaly_mode not in ANOMALY_MODES:
            raise ValueError(f"anomaly_mode must be one of {ANOMALY_MODES}")
        self.anomaly_mode = anomaly_mode
        self.setpoint = setpoint
        self.rng = np.random.default_rng(seed)
        self._temp_drift_accum = 0.0
        self._door_open = False

    def set_anomaly_mode(self, mode):
        if mode not in ANOMALY_MODES:
            raise ValueError(f"unknown anomaly mode: {mode}")
        self.anomaly_mode = mode

    def _has_temp_drift(self):
        return self.anomaly_mode in ("temp_drift", "combined")

    def _has_vibration_anomaly(self):
        return self.anomaly_mode in ("vibration", "combined")

    def next_temperature(self):
        """Called at 1 Hz. Drift accumulates while temp_drift/combined is
        active and resets once the anomaly clears."""
        if self._has_temp_drift():
            self._temp_drift_accum = min(self._temp_drift_accum + TEMP_DRIFT_PER_SAMPLE, MAX_TEMP_DRIFT_C)
        else:
            self._temp_drift_accum = 0.0
        mean = self.setpoint + self._temp_drift_accum
        return float(self.rng.normal(mean, TEMP_NOISE_STD))

    def next_vibration(self):
        """Called at 0.5 Hz. Vibration anomaly is a step change, not a
        drift, matching a sudden bearing-wear failure signature."""
        if self._has_vibration_anomaly():
            value = self.rng.normal(VIB_ANOMALY_MEAN, VIB_ANOMALY_STD)
        else:
            value = self.rng.normal(VIB_NORMAL_MEAN, VIB_NORMAL_STD)
        return float(max(0.0, value))

    def maybe_door_event(self, dt_seconds, rate_per_hour=DOOR_EVENTS_PER_HOUR):
        """Poisson-process door events, called once per second."""
        p = rate_per_hour * dt_seconds / 3600.0
        if self.rng.random() < p:
            self._door_open = not self._door_open
            return "OPEN" if self._door_open else "CLOSE"
        return None


def generate_offline_series(anomaly_mode, duration_s, seed=None, start_ts=0.0):
    """Fast (no wall-clock sleep) generation of a full sensor series,
    sample-for-sample identical in distribution to the live MQTT loop.
    Used by training/generate_dataset.py to build the labelled dataset
    without waiting the real 20/15/15-minute durations."""
    gen = ColdChainSensorGenerator(anomaly_mode=anomaly_mode, seed=seed)

    n_temp = int(duration_s * 1.0)
    temp_ts = start_ts + np.arange(n_temp) * 1.0
    temp_vals = np.array([gen.next_temperature() for _ in range(n_temp)])

    n_vib = int(duration_s * 0.5)
    vib_ts = start_ts + np.arange(n_vib) * 2.0
    vib_vals = np.array([gen.next_vibration() for _ in range(n_vib)])

    door_events = []
    t = 0.0
    while t < duration_s:
        ev = gen.maybe_door_event(1.0)
        if ev:
            door_events.append((start_ts + t, ev))
        t += 1.0

    return {
        "temperature": (temp_ts, temp_vals),
        "vibration": (vib_ts, vib_vals),
        "door": door_events,
    }


class SensorPublisher:
    """Live MQTT publishing loop (real time, three threads)."""

    def __init__(self, broker, port, truck_id, anomaly_mode, seed=None):
        self.truck_id = truck_id
        self.broker = broker
        self.port = port
        self.generator = ColdChainSensorGenerator(anomaly_mode=anomaly_mode, seed=seed)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"simulator-{truck_id}")
        configure_client_security(self.client)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_control_message
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def _topic(self, suffix):
        return f"logibridge/trucks/{self.truck_id}/{suffix}"

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self._topic("control/anomaly"))
        print(f"[simulator] connected to {self.broker}:{self.port}, "
              f"truck={self.truck_id}, mode={self.generator.anomaly_mode}")

    def _on_control_message(self, client, userdata, msg):
        mode = msg.payload.decode().strip()
        try:
            with self._lock:
                self.generator.set_anomaly_mode(mode)
            print(f"[simulator] anomaly mode switched -> {mode}")
        except ValueError as e:
            print(f"[simulator] {e}")

    def _temperature_loop(self):
        while not self._stop.is_set():
            with self._lock:
                value = self.generator.next_temperature()
            payload = json.dumps({"truck_id": self.truck_id, "timestamp": time.time(), "value": round(value, 3)})
            self.client.publish(self._topic("sensors/temperature"), payload, qos=0)
            time.sleep(1.0)

    def _vibration_loop(self):
        while not self._stop.is_set():
            with self._lock:
                value = self.generator.next_vibration()
            payload = json.dumps({"truck_id": self.truck_id, "timestamp": time.time(), "rms": round(value, 4)})
            self.client.publish(self._topic("sensors/vibration"), payload, qos=0)
            time.sleep(2.0)

    def _door_loop(self):
        while not self._stop.is_set():
            with self._lock:
                event = self.generator.maybe_door_event(1.0)
            if event:
                payload = json.dumps({"truck_id": self.truck_id, "timestamp": time.time(), "event": event})
                self.client.publish(self._topic("sensors/door"), payload, qos=1)
            time.sleep(1.0)

    def run(self, duration_s=None):
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        threads = [
            threading.Thread(target=self._temperature_loop, daemon=True),
            threading.Thread(target=self._vibration_loop, daemon=True),
            threading.Thread(target=self._door_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            if duration_s:
                time.sleep(duration_s)
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            time.sleep(0.2)
            self.client.loop_stop()
            self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    parser.add_argument("--anomaly", choices=ANOMALY_MODES, default="none")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=default_port())
    parser.add_argument("--truck-id", default="TRUCK001")
    parser.add_argument("--duration", type=float, default=None, help="seconds; omit to run forever")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    publisher = SensorPublisher(args.broker, args.port, args.truck_id, args.anomaly, seed=args.seed)
    publisher.run(duration_s=args.duration)


if __name__ == "__main__":
    main()
