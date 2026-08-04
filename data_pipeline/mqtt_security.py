"""Shared TLS/auth configuration for every LogiEdge MQTT client
(simulator, inference service, drift monitor). Reads the same
environment variables everywhere so a single credential/cert setup
(deployment/generate_mqtt_credentials.sh + mosquitto_production.conf)
covers all three:

  MQTT_USERNAME       -- e.g. logibridge_sensors / logibridge_inference / logibridge_ops
  MQTT_PASSWORD
  MQTT_TLS_CA         -- path to the CA cert (deployment/certs/ca.crt); presence enables TLS
  MQTT_TLS_INSECURE   -- "true" to skip hostname verification (self-signed dev/test certs)

With none of these set, a client connects exactly as before: plaintext,
anonymous, port 1883 (mosquitto_local.conf) -- so existing local dev/test
usage is unaffected. Setting MQTT_USERNAME/PASSWORD and MQTT_TLS_CA
switches it to the production profile (mosquitto_production.conf, TLS on
port 8883, authenticated, ACL-scoped).
"""
import os


def default_port():
    return 8883 if os.environ.get("MQTT_TLS_CA") else 1883


def configure_client_security(client):
    """Apply username/password and TLS settings to a paho mqtt.Client
    from environment variables, if present. Returns the client for
    chaining."""
    username = os.environ.get("MQTT_USERNAME")
    password = os.environ.get("MQTT_PASSWORD")
    if username:
        client.username_pw_set(username, password)

    ca_path = os.environ.get("MQTT_TLS_CA")
    if ca_path:
        client.tls_set(ca_certs=ca_path)
        if os.environ.get("MQTT_TLS_INSECURE", "").lower() in ("1", "true", "yes"):
            # Self-signed dev/test certs (CN=logibridge-broker.local) won't
            # match "localhost" -- skip hostname verification for those,
            # never for a real deployment with a properly issued cert.
            client.tls_insecure_set(True)

    return client
