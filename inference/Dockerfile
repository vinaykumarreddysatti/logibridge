# LogiEdge inference container (Task D2).
# Build from the repo root: docker build -f inference/Dockerfile -t logibridge-inference .
FROM python:3.11-slim

WORKDIR /app

# All pip install layers appear before any COPY, and specifically before
# the model file copy below, so that an OTA update touching only
# model.tflite invalidates just the final layer -- not this one.
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    paho-mqtt==2.1.0 \
    numpy==1.26.4 \
    scipy==1.13.1

COPY data_pipeline/preprocessing.py ./data_pipeline/preprocessing.py
COPY data_pipeline/mqtt_security.py ./data_pipeline/mqtt_security.py
COPY data_pipeline/training_stats.npy ./training_stats.npy
COPY inference/inference_service.py ./inference_service.py

# Model file copied last: an OTA update that only swaps model.tflite
# rebuilds/pulls only this layer (see the layer-cache demo in Task D2).
COPY inference/model.tflite ./model.tflite

ENV MODEL_PATH=/app/model.tflite
ENV STATS_PATH=/app/training_stats.npy
ENV MQTT_BROKER=host.docker.internal
ENV MQTT_PORT=1883
ENV TRUCK_ID=TRUCK001
ENV ALERT_LOG=/app/alert_log.jsonl

# Production MQTT security (TLS + auth, see mqtt_architecture.md) is off
# by default here to keep local/dev `docker run` working unchanged. For
# a real truck deployment, set these (and mount the CA cert into the
# container) via the Ansible playbook's environment block:
#   MQTT_USERNAME=logibridge_inference
#   MQTT_PASSWORD=<from mosquitto_passwd.txt>
#   MQTT_TLS_CA=/app/certs/ca.crt   (bind-mount deployment/certs/ca.crt here)
#   MQTT_PORT=8883

CMD ["python", "inference_service.py"]
