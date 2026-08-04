#!/usr/bin/env bash
# Regenerates the MQTT auth password file and self-signed TLS cert chain
# used by mosquitto_production.conf. Run from the deployment/ directory.
# Outputs (mosquitto_passwd.txt, certs/) are gitignored -- these are
# demo/dev credentials; rotate them before any real deployment.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p certs
mosquitto_passwd -c -b mosquitto_passwd.txt logibridge_sensors "${SENSORS_MQTT_PASSWORD:?set SENSORS_MQTT_PASSWORD}"
mosquitto_passwd -b mosquitto_passwd.txt logibridge_inference "${INFERENCE_MQTT_PASSWORD:?set INFERENCE_MQTT_PASSWORD}"
mosquitto_passwd -b mosquitto_passwd.txt logibridge_ops "${OPS_MQTT_PASSWORD:?set OPS_MQTT_PASSWORD}"

openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
  -keyout certs/ca.key -out certs/ca.crt \
  -subj "/C=IN/O=FreightBridge/CN=LogiEdge-CA"

openssl req -nodes -newkey rsa:2048 \
  -keyout certs/broker.key -out certs/broker.csr \
  -subj "/C=IN/O=FreightBridge/CN=logibridge-broker.local"

openssl x509 -req -in certs/broker.csr -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial \
  -out certs/broker.crt -days 825

echo "done: mosquitto_passwd.txt, certs/{ca,broker}.{crt,key}"
