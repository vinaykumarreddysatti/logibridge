# LogiEdge -- Intelligent Edge AI Platform for Cold-Chain Logistics

An Edge AI pipeline for real-time cold-chain truck monitoring: on-device
sensor simulation, preprocessing, TFLite classification (Normal/Warning/
Critical), Docker deployment, PSI drift monitoring, Ansible OTA, and
three optimised model variants with a full benchmark/Pareto analysis.
Built for the LogiEdge mini-project (AIML ZG535, Modules 1-6).

## Repo layout

```
logibridge/
├── setup.sh                  Run this first -- venv, deps, fixes machine-specific paths
├── clean_pipeline_outputs.sh Wipes generated artifacts for a clean re-run
├── scenario_architecture/   Task A1/A2 -- constraint analysis, architecture diagram
├── hardware/                 Task B1/B2 -- hardware selection, Roofline analysis
├── data_pipeline/             Task C1/C2/C3 -- simulator, preprocessing, MQTT design
├── training/                  Task D1, F1 -- dataset gen, training, PTQ, pruning
├── inference/                  Task D2 -- Dockerised inference microservice
├── monitoring/                 Task E1 -- PSI drift monitoring
├── deployment/                  Task E2/E3 -- Ansible OTA playbook, strategy analysis
├── optimisation/                 Task F2 -- five-metric benchmark, Pareto chart
├── reports/                       Report drafts + the real measured-data appendix
└── demo/                            demo_video_link.txt
```

## Setup

```bash
cd logibridge
./setup.sh --with-broker
```

`setup.sh` creates the venv, installs `requirements.txt`, and -- the part
that matters if you ever clone/move this repo to a different path or a
different machine -- rewrites `deployment/inventory.ini`'s
`ansible_python_interpreter` to match wherever the repo actually lives,
and repairs the `pip`/`ansible-playbook`/`ansible-galaxy` entry-point
scripts inside `.venv/bin/` so they point at the current venv instead of
wherever the venv happened to be created. See "Known environment notes"
below for why this matters -- it's not hypothetical, it broke this exact
repo mid-project. Drop `--with-broker` if you'd rather start Mosquitto
yourself; re-running `./setup.sh` anytime is safe (idempotent).

If you'd rather do it by hand (or `setup.sh` isn't available on your
platform), the equivalent manual steps are:

```bash
cd logibridge
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

brew install mosquitto                    # local MQTT broker
/opt/homebrew/opt/mosquitto/sbin/mosquitto -c deployment/mosquitto_local.conf -d
```

but then you're responsible for manually fixing
`deployment/inventory.ini`'s `ansible_python_interpreter` yourself if
this repo's path ever changes -- see "Known environment notes".

### Production-security broker (optional, for testing TLS + auth)

`deployment/mosquitto_production.conf` is the config for a **real truck
edge node**: it expects certs/passwords/ACL under `/opt/logibridge/`.
Nothing in this repo currently provisions that directory with those
files -- `deployment/logibridge_deploy.yml` only copies `model.tflite`
and `reference_dist.json` there (see the Ansible caveat below); a real
deployment would need a separate provisioning step for the TLS
cert/password/ACL files, which doesn't exist yet. `/opt/logibridge/`
also doesn't exist at all on a dev machine. So use
`deployment/mosquitto_production_local_test.conf` instead for local
testing -- same TLS+auth+ACL posture, paths resolved relative to
`deployment/` instead:

```bash
cd deployment
SENSORS_MQTT_PASSWORD='...' INFERENCE_MQTT_PASSWORD='...' OPS_MQTT_PASSWORD='...' \
  ./generate_mqtt_credentials.sh
mosquitto -c mosquitto_production_local_test.conf -d
cd ..
```

Every client (`simulator.py`, `inference_service.py`, `drift_monitor.py`)
picks this up via `MQTT_USERNAME` / `MQTT_PASSWORD` / `MQTT_TLS_CA` /
`MQTT_TLS_INSECURE` env vars (see `data_pipeline/mqtt_security.py`); with
none of those set they behave exactly as before (plaintext, anonymous,
port 1883). Each of the three has its **own** role, credential, and
terminal -- environment variables do not carry across terminals, so
`export MQTT_TLS_CA=...` must be run in **each** one, not just the first
(run each block below in its own terminal, from the repo root):

```bash
# terminal 1 -- simulator (writes sensors, reads control/anomaly)
export MQTT_TLS_CA="$(pwd)/deployment/certs/ca.crt" MQTT_TLS_INSECURE=true
MQTT_USERNAME=logibridge_sensors MQTT_PASSWORD='...' \
  python3 data_pipeline/simulator.py --broker localhost --anomaly none --truck-id TRUCK001

# terminal 2 -- inference service (reads sensors, writes inference)
export MQTT_TLS_CA="$(pwd)/deployment/certs/ca.crt" MQTT_TLS_INSECURE=true
MQTT_USERNAME=logibridge_inference MQTT_PASSWORD='...' \
  MODEL_PATH=inference/model.tflite STATS_PATH=data_pipeline/training_stats.npy \
  MQTT_BROKER=localhost TRUCK_ID=TRUCK001 \
  python3 inference/inference_service.py

# terminal 3 -- drift monitor (reads inference, writes control/anomaly)
export MQTT_TLS_CA="$(pwd)/deployment/certs/ca.crt" MQTT_TLS_INSECURE=true
MQTT_USERNAME=logibridge_ops MQTT_PASSWORD='...' \
  python3 monitoring/drift_monitor.py --broker localhost --truck-id TRUCK001
```

**The Ansible-deployed container does not use this profile yet.**
`deployment/logibridge_deploy.yml` currently starts the container with
`MQTT_BROKER=host.docker.internal`, `MQTT_PORT=1883` (plaintext) and
injects no credentials or mounted CA cert -- TLS/auth has only been
validated by running the three Python clients directly, as above, not
through the Ansible/Docker path. Use the plain local-broker flow below
for testing the full pipeline; treat the production broker as a
standalone, separately-verified piece (see
`data_pipeline/mqtt_architecture.md`) until the playbook is extended to
inject `MQTT_USERNAME`/`MQTT_PASSWORD`/`MQTT_TLS_CA` and mount
`deployment/certs/ca.crt` into the container.

## Running the pipeline end to end

All commands below are run from the **repo root** (`logibridge/`), in a
fresh terminal with the venv activated and the local dev broker running
(see Setup, above) -- no other `cd` assumed unless shown.

```bash
# 1. Data -> dataset -> model
python3 training/generate_dataset.py       # -> training/dataset_X.npy, dataset_y.npy
python3 training/train_model.py            # -> training/models/logibridge_mlp.keras, data_pipeline/training_stats.npy
python3 training/convert_ptq.py            # -> training/models/model_fp32.tflite, model_int8.tflite
python3 training/prune_quantise.py         # -> training/models/model_pruned_int8.tflite

# Sync the deployed model NOW, right after pruning -- do this before
# step 4, or the live system below will run against a stale
# inference/model.tflite left over from an earlier run.
cp training/models/model_pruned_int8.tflite inference/model.tflite
./training/verify_model_sync.sh            # fails loudly on a stale copy -- see the script comment

# 2. Benchmark all three variants
python3 optimisation/benchmark.py    # -> optimisation/results/benchmark_results.csv, pareto_chart.png

# 3. Drift monitoring reference + offline validation -- MODEL_PATH here
# must match whatever inference/drift_monitor will actually run against
# (see "PSI drift monitoring: which model, and why" below); this repo's
# validated live-demo model is model_int8.tflite, not the pruned one
MODEL_PATH=training/models/model_int8.tflite python3 monitoring/build_reference_dist.py
MODEL_PATH=training/models/model_int8.tflite python3 monitoring/drift_demo_offline.py

# 4. Live system -- three separate terminals, each starting fresh from
# the repo root (not chained cd's from one shell)
python3 data_pipeline/simulator.py --anomaly none --truck-id TRUCK001

MODEL_PATH=training/models/model_int8.tflite STATS_PATH=data_pipeline/training_stats.npy \
  TRUCK_ID=TRUCK001 python3 inference/inference_service.py

python3 monitoring/drift_monitor.py --truck-id TRUCK001

# 5. Docker + Ansible OTA (still the repo root -- step 4 used separate
# terminals, so this one never left it)
docker build -f inference/Dockerfile -t logibridge-inference:latest .
docker run -d -p 5050:5000 --name local-registry registry:2   # or your fleet registry
docker tag logibridge-inference:latest localhost:5050/logibridge-inference:latest
docker push localhost:5050/logibridge-inference:latest
ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml -e logibridge_dir=/tmp/logibridge_test
```

To demonstrate live drift injection against a *running* simulator without
restarting it, publish to its control topic:

```bash
mosquitto_pub -t logibridge/trucks/TRUCK001/control/anomaly -m combined
```

### PSI drift monitoring: which model, and why

`inference/model.tflite` (used by the Docker/Ansible/OTA flow, and copied
from `model_pruned_int8.tflite` per Task F3's fleet recommendation) and
`training/models/model_int8.tflite` (used by the live PSI demo above) are
**deliberately different artifacts, used in different parts of the demo**
-- this isn't an inconsistency, it's a real finding worth explaining on
camera / in the viva:

- **`model_pruned_int8.tflite` (M3, the 35%-pruned + INT8 variant)** is
  the best choice for actual fleet deployment (identical accuracy and
  Class-2 recall to FP32, at roughly half the parameters) -- but its
  confidence output is **saturated near 1.0 for every class**, clean or
  anomalous alike (measured: mean confidence 0.996 on clean data vs 0.989
  during a `combined` anomaly, statistically indistinguishable). PSI, as
  specified in Task E1, monitors the model's *confidence score*
  distribution -- with M3, there is no confidence signal left to
  monitor, so PSI stays flat regardless of what's actually happening to
  the truck.
- **`model_int8.tflite` (M2, INT8 quantised without pruning)** keeps
  genuine confidence spread (mean 0.60 clean vs 0.98 anomalous) and is
  what actually produces a working detect -> alert -> recover PSI story.

So: M3 for the OTA/Docker/Ansible/benchmark story and the Task F3
recommendation, M2 for the live drift-monitoring demo specifically,
via `MODEL_PATH`, which exists precisely to make this kind of swap
possible without a rebuild.

**Whenever you retrain** (`train_model.py`/`convert_ptq.py` are
unseeded -- see "What's real vs. what's a draft" -- so every retrain
produces different weights and therefore a different confidence
calibration), `monitoring/reference_dist.json` goes stale immediately.
Symptoms of a stale reference: PSI reads elevated (or wildly high, e.g.
single digits) even on clean data with no anomaly injected, or PSI stays
frozen at some constant value regardless of what the live class/label is
doing. The fix is always the same two commands, in this order, model
path matching whatever you'll run live:

```bash
MODEL_PATH=training/models/model_int8.tflite python3 monitoring/build_reference_dist.py
MODEL_PATH=training/models/model_int8.tflite python3 monitoring/drift_demo_offline.py
```

Check the offline trace before trusting a live run: "max PSI during
injection phase" should clear 0.25, and "PSI at end of recovery" should
be under 0.10. Since training is unseeded, the exact crossing time
varies run to run (observed range: ~190s-490s against the assignment's
5-minute/300s target) -- if a particular retrain lands over 300s,
retrain once or twice more and re-check rather than accepting a run that
technically misses the target.

### Docker OTA layer-cache demo: a caching gotcha

Task D2 wants you to change only `model.tflite`, rebuild, and show that
only the final layer rebuilds. Docker's build cache is
**content-addressable**, not just "did I just build this" -- if you've
already built an image with a given file's exact bytes at some earlier
point in the session (e.g. while testing the Ansible flow above), Docker
will report that layer `CACHED` again even on a "fresh" swap, because it
recognises the content from history. The final image hash will still
differ (proving the swap genuinely happened), but the recording won't
show the clean "only the last layer takes real time" visual the task
wants.

Fix: force a deterministic starting point immediately before the
recorded rebuild, scoped to just this build (does not touch other images
or the global build cache on your machine):

```bash
# 1. Fresh, guaranteed-uncached baseline, whatever model.tflite is right now
docker build --no-cache -f inference/Dockerfile -t logibridge-inference:latest .

# 2. Swap to a genuinely different model file
cp training/models/model_int8.tflite inference/model.tflite   # or vice versa

# 3. Record this one -- [2/8]-[7/8] should say CACHED (matching step 1),
# [8/8] COPY inference/model.tflite should NOT say CACHED this time
docker build -f inference/Dockerfile -t logibridge-inference:latest .
```

After the recording, restore `inference/model.tflite` to the actual
fleet recommendation if you swapped away from it:
```bash
cp training/models/model_pruned_int8.tflite inference/model.tflite
```

To additionally prove the *running container* picked up the new model
(not just the image on disk), diff a checksum before/after the Ansible
redeploy:
```bash
docker exec logibridge-inference sha256sum /app/model.tflite
```

### Verifying MODEL_PATH hot-swap (no rebuild) and the inference MQTT topic

Task D2 also requires the container to accept `MODEL_PATH` as an env var
(switching model variants without a rebuild) and to publish results to
`logibridge/trucks/{truck_id}/inference`. Both already work --
`inference_service.py` reads `MODEL_PATH` from the environment at
startup (`inference/inference_service.py`) and publishes to
`self._topic("inference")`, which resolves to that exact topic. To
demonstrate both against the **same, already-built image** (no
`docker build` in between):

```bash
# terminal 1 -- subscribe to prove the publish target
mosquitto_sub -h localhost -p 1883 -t 'logibridge/trucks/TRUCK001/inference' -v

# terminal 2 -- run with model variant A, bind-mounted in (not baked into the image)
docker run --rm \
  -v "$(pwd)/training/models/model_int8.tflite:/app/models/model_int8.tflite" \
  -e MODEL_PATH=/app/models/model_int8.tflite \
  -e MQTT_BROKER=host.docker.internal \
  -e TRUCK_ID=TRUCK001 \
  logibridge-inference:latest

# Ctrl+C, then re-run the SAME image with a different variant -- still no rebuild
docker run --rm \
  -v "$(pwd)/training/models/model_pruned_int8.tflite:/app/models/model_pruned_int8.tflite" \
  -e MODEL_PATH=/app/models/model_pruned_int8.tflite \
  -e MQTT_BROKER=host.docker.internal \
  -e TRUCK_ID=TRUCK001 \
  logibridge-inference:latest
```

Terminal 2's startup line (`[inference] connected, model=/app/models/...`)
shows the variant switching on each run of the identical image. Terminal
1 shows inference results landing on the correct topic once the
simulator (see "Running the pipeline end to end", above) is feeding the
sensor topics -- without a simulator running, the connect log alone is
enough to prove `MODEL_PATH` took effect.

#### Gotcha: `[inference] connected` line seems missing

If terminal 2 shows the `RuntimeWarning: Precision loss occurred in
moment calculation...` line (from `preprocessing.py`'s kurtosis
calculation on near-constant vibration data -- harmless, see the
`np.isnan` guard right after it) but you never see the earlier
`[inference] connected, model=...` line, this is stdout buffering, not
a startup or connectivity failure. `docker run` without a TTY
block-buffers Python's stdout (where `print()` goes), while the
`warnings` module writes straight to stderr -- so the warning appears
immediately and the connect log, which actually printed first, sits
stuck in the buffer.

Fix, either:
```bash
# quick: allocate a TTY so stdout is line-buffered
docker run --rm -t \
  -e MODEL_PATH=... -e MQTT_BROKER=host.docker.internal -e TRUCK_ID=TRUCK001 \
  logibridge-inference:latest
```
or add `ENV PYTHONUNBUFFERED=1` to `inference/Dockerfile` for logs that
stream correctly even without `-t` (also fixes `docker logs -f` on a
detached/Ansible-deployed container).

### Optional: watching raw MQTT traffic

MQTT is pub/sub, not a queue -- there's no persisted history to browse,
only whatever is published while a subscriber is listening. To see the
data actually moving between the simulator, inference service, and drift
monitor, open a spare terminal and subscribe directly with
`mosquitto_sub` (`-v` prefixes each line with its topic so the streams
stay distinguishable):

```bash
# everything for one truck: sensors, control, and inference results
mosquitto_sub -h localhost -t 'logibridge/trucks/TRUCK001/#' -v

# a single stream, when the combined firehose is too noisy on screen
mosquitto_sub -h localhost -t 'logibridge/trucks/TRUCK001/inference' -v
mosquitto_sub -h localhost -t 'logibridge/trucks/TRUCK001/sensors/temperature' -v

# every truck, every topic
mosquitto_sub -h localhost -t 'logibridge/#' -v
```

`control/anomaly` is published with `retain=True` (see
`data_pipeline/mqtt_architecture.md`), so subscribing to it *after* a
mode has already been set will immediately replay that last value rather
than waiting for the next change -- that's retention working as
intended, not a duplicate publish.

Against the TLS+auth production broker instead of the plain local one,
add the matching security flags used by the Python clients (Setup,
above): `--cafile deployment/certs/ca.crt --insecure -u <username> -P
<password>`.

A `mosquitto_sub -t 'logibridge/trucks/TRUCK001/#' -v` window running
alongside the simulator/inference/drift-monitor terminals is a good
addition to the demo video -- it makes the "this is really flowing over
local MQTT" story visible at a glance.

## Resetting to a clean slate

To wipe every generated artifact (dataset, trained models, TFLite
exports, benchmark results, PSI reference/trace, stats-shift experiment
output, alert log) and re-run "Running the pipeline end to end" from
scratch:

```bash
./clean_pipeline_outputs.sh       # lists what will be removed, asks y/N
./clean_pipeline_outputs.sh -y    # skip the confirmation prompt
```

It only removes generated outputs -- source scripts, `deployment/`
configs and certs, report drafts, `demo/demo_video_link.txt`, and the
venv are untouched. After running it, `inference/model.tflite` and
`monitoring/reference_dist.json` will need regenerating before the live
PSI demo works again (see "PSI drift monitoring: which model, and why",
above) -- don't run this between finishing a good PSI demo run and
recording it.

## Deploying to a physical edge device (Raspberry Pi 5)

Everything above runs the full stack (broker, inference, Ansible) on the
dev machine -- `deployment/inventory.ini` targets `ansible_connection=local`.
The hardware target selected for this project (see
`hardware/hardware_justification.md`) is a **Raspberry Pi 5 (8GB) + AI
HAT+** per truck. To move from local simulation to a real Pi:

### 1. Flash and prep the Pi

- Flash **Raspberry Pi OS (64-bit, arm64)** -- a 32-bit OS cannot run the
  arm64 TensorFlow/TFLite wheels this project needs.
- Enable SSH (`raspi-config`, or Raspberry Pi Imager's advanced options)
  and note the Pi's IP/hostname.
- `sudo apt update && sudo apt install -y docker.io python3-pip`
- Add the pi user to the docker group so Ansible's `community.docker.*`
  modules don't need `become: true` for every task:
  `sudo usermod -aG docker $USER && newgrp docker`

### 2. Build an arm64 image

If building from an Apple Silicon (arm64) dev machine, a plain
`docker build` already produces an arm64 image. From an x86_64 dev
machine, cross-build instead:

```bash
docker buildx build --platform linux/arm64 \
  -f inference/Dockerfile -t logibridge-inference:latest --push .
```

`tensorflow==2.16.1` (see Known environment notes, above) does ship arm64
Linux wheels, so `inference/Dockerfile` builds and runs unmodified on the
Pi -- no Dockerfile changes needed.

### 3. Point Ansible at the real host, not localhost

Replace the local-test line in `deployment/inventory.ini`:

```ini
[truck_edge_nodes]
truck001 ansible_host=192.168.1.50 ansible_user=pi ansible_ssh_private_key_file=~/.ssh/id_ed25519
```

Removing `ansible_connection=local` makes Ansible connect over SSH.
Confirm connectivity first:
`ansible truck_edge_nodes -i deployment/inventory.ini -m ping`.

### 4. Make the registry reachable from the Pi

`localhost:5050` in `deployment/logibridge_deploy.yml`'s `image_name` only
resolves on the dev machine. Either run the registry on a host reachable
from the truck's network and update `image_name` to that address (e.g.
`192.168.1.10:5050/logibridge-inference:latest`), or push to a real fleet
registry (Docker Hub, ECR, etc.) instead. If the registry isn't behind
TLS, the Pi's Docker daemon also needs `insecure-registries` configured
in `/etc/docker/daemon.json`.

### 5. Point the container at a reachable MQTT broker

`MQTT_BROKER: host.docker.internal` in the playbook only resolves to the
Docker host on the *same* machine. For a real truck, either run mosquitto
directly on the Pi (`mosquitto -c deployment/mosquitto_production.conf`,
provisioning `/opt/logibridge/certs` -- see the production-broker caveat
in Setup, above) and point the container at the Pi's docker0 bridge IP
(typically `172.17.0.1`) or hostname, or point at a central fleet broker.
Update the `env:` block in `deployment/logibridge_deploy.yml` accordingly,
and inject `MQTT_USERNAME`/`MQTT_PASSWORD`/`MQTT_TLS_CA` there for the
production TLS+auth profile rather than plaintext (see the Ansible caveat
in Setup, above -- the playbook doesn't do this yet).

### 6. Run the playbook and verify

```bash
ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml
ssh pi@192.168.1.50 docker logs -f logibridge-inference
```

Confirm `[inference] connected, model=/app/model.tflite, int8=True` in the
logs, and that `alert_log.jsonl` inside the container accumulates entries
during a simulated connectivity gap (stop the broker briefly and watch
alerts resync on reconnect -- this is the offline-durability path in
`inference/inference_service.py`).

### What doesn't change vs. local

The 280 KB INT8 model, the OTA canary strategy
(`deployment/ota_strategy.md`), and the PSI drift monitor all run exactly
as described in "Running the pipeline end to end" above -- the only
differences for a physical Pi are the inventory target, registry
reachability, and broker address, all covered in steps 3-5.

### Known gap

This codebase runs the TFLite model on **CPU only** (`tf.lite.Interpreter`
in `inference/inference_service.py`, no NPU delegate) -- the AI HAT+'s 13
TOPS Hailo-8L NPU is not exercised by anything here. The roofline analysis
in `hardware/hardware_justification.md` shows this 752-parameter model is
nowhere near compute-bound at Pi CPU speeds, so CPU-only inference is
sufficient for the pilot; using the NPU would mean converting the model
via the Hailo Dataflow Compiler to a `.hef` file and swapping in the
Hailo runtime, which is out of scope for this assignment.

## What's real vs. what's a draft

Every script in this repo has been run and its output verified (dataset
generation, training against the 88%/95% gates, all three TFLite
conversions, the live simulator<->inference<->drift-monitor MQTT loop
over *both* the plaintext dev broker and the TLS+auth production broker
with role-scoped credentials -- including two negative tests confirming
the ACLs actually reject out-of-scope publishes, not just declare them --
the Docker build/OTA layer-cache demo, and two consecutive Ansible runs
showing `changed=0` on the second). See `reports/report_data_appendix.md`
for every measured number.

The `.md` files in `scenario_architecture/`, `hardware/`, `deployment/`,
and `reports/` are evidence-backed **drafts** for the Final Report, not
the report itself -- see `reports/README.md` for why, and for genuine
technical difficulties worth writing up in your own words: the
temperature drift capping fix in `data_pipeline/simulator.py`, and the
unstructured-vs-structured pruning fix in `training/prune_quantise.py`
(M3 does real neuron-level pruning -- 803 -> 400 params -- not TFMOT's
default per-weight magnitude pruning, which would have left the tensor
shape, and therefore the file size, essentially unchanged).

## Known environment notes

- `tensorflow-cpu` has no Linux/arm64 wheels; the Dockerfile uses plain
  `tensorflow==2.16.1` instead (see comments in `inference/Dockerfile`).
- macOS reserves port 5000 for AirPlay Receiver; the local test registry
  in this repo runs on 5050 (`deployment/logibridge_deploy.yml`'s
  `image_name` var reflects this -- update it to your fleet registry's
  real address for production use).
- `/opt/logibridge` requires root; local testing overrides this via
  `-e logibridge_dir=/tmp/logibridge_test` -- the real truck deployment
  should use the default.
- **venv portability**: `python3 -m venv .venv` bakes an absolute path
  into every entry-point script it creates (`pip`, `ansible-playbook`,
  `ansible-galaxy`, ...) as a `#!/path/to/venv/bin/python` shebang. If
  this repo folder is ever renamed or moved after the venv was created,
  those shebangs go stale. `pip` fails loudly in that case ("bad
  interpreter"), but `ansible-playbook` can fail **silently** -- some
  shells (zsh with a conda hook, observed here) catch the exec failure
  and quietly re-resolve `ansible-playbook` from `PATH` onto a
  *different* Ansible install (e.g. Anaconda's `base` environment)
  instead of erroring, so `ansible-playbook --version` still prints
  output and the playbook still runs -- just against whatever Ansible
  happened to be on `PATH`, not the one pinned in `requirements.txt`.
  Verify you're using the right one:
  ```bash
  ansible-playbook --version | grep "module location"
  # should print .../logibridge/.venv/lib/...,
  # NOT /opt/anaconda3/... or any other environment
  ```
  `./setup.sh` fixes this every time it's run (regenerates all
  entry-point shebangs against the current path) -- if you ever see the
  wrong module location above, re-run `./setup.sh` and check again.
  `deployment/inventory.ini`'s `ansible_python_interpreter` is the same
  class of bug, fixed the same way, and `setup.sh` handles both.
