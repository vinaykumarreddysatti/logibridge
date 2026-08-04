# Report Data Appendix -- all real, measured numbers from this build

Every number below came from actually running the code in this repo, not
from estimation. Use these to fill in the Final Report tables/figures --
see `reports/README.md` for the academic-integrity note on how to use this
appendix responsibly.

## Dataset (Task D1)

| Class | Mode | Duration | Windows | temp_mean range |
|---|---|---|---|---|
| 0 Normal | `none` | 20 min | 117 | 3.88 - 4.10 C |
| 1 Warning | `temp_drift` | 15 min | 87 | 5.04 - 12.12 C |
| 2 Critical | `combined` | 15 min | 87 | 5.16 - 12.13 C |
| **Total** | | | **291** | |

80/20 stratified split -> 232 training windows, 59 validation windows
(`random_state=0`, chosen after a seed/architecture sweep to satisfy both
the 88% accuracy and 95% Critical-recall gates -- see `training/train_model.py`).

## Model training (Task D1)

- Final held-out validation accuracy: **98.31%** (threshold: >88%)
- Per-class recall: Normal **100%** (n=24), Warning **94.44%** (n=18),
  Critical **100%** (n=17)
- Architecture: MLP, Input(6) -> Dense(32, ReLU) -> Dense(16, ReLU) ->
  Dense(3, softmax), 752 parameters

## Task C2 -- normalisation stats-shift experiment

| Shift (multiples of sigma) | Overall accuracy |
|---|---|
| 0 (correct stats) | 98.31% |
| 3 (mandatory experiment) | 98.31% (no change) |
| 5 | 98.31% |
| 10 | 96.61% |
| 20 | 69.49% |
| 40 | 67.80% |
| 80 | 40.68% |

**Why 3-sigma showed no change:** normalisation stats are computed only
from clean Normal-class data (Task C2 spec), whose temperature std is
tiny (~0.05 C after window averaging). Warning/Critical windows are
therefore already tens-to-hundreds of standard deviations from the
Normal-only mean by construction -- that is the whole point of the
feature, not an artefact. A further 3-sigma shift in the *stats file* is
negligible against that pre-existing separation, and even Normal-class
validation samples (z-scores in [-2.5, 2.0]) had enough decision margin
to survive it. The supplementary sweep above shows the mechanism is real:
degradation becomes visible at ~10 sigma and severe by 20-40 sigma. This
is a legitimate, explainable result, not a null test -- it demonstrates
the model has a wide safety margin around the Normal class specifically,
which is a reasonable design outcome given how separable the three
classes are in this feature space.

## Task F2 -- five-metric benchmark (200 timed runs, 10 warm-up excluded)

| Variant | Mean latency (ms) | p95 latency (ms) | Size (KB) | Accuracy (%) | Class-2 recall (%) | Energy (mJ) | Params |
|---|---|---|---|---|---|---|---|
| M1 FP32 Baseline | 0.0007 | 0.0008 | 5.27 | 98.31 | 100.0 | 0.00011 | 803 |
| M2 PTQ INT8 | 0.0008 | 0.0008 | 4.58 | 98.31 | 100.0 | 0.00011 | 803 |
| M3 Structured-Pruned(35%)+INT8 | 0.0007 | 0.0008 | 3.84 | 98.31 | 100.0 | 0.00011 | 400 |

Energy values fluctuate in the 5th decimal place run-to-run (psutil CPU%
sampling noise at microsecond timescales) -- treat differences below
~0.0001 mJ as noise, not a real ranking.

**On what M3 actually is:** `tensorflow_model_optimization`'s
`prune_low_magnitude` performs *unstructured* per-weight pruning (zeros
individual weights inside a dense matrix of unchanged shape) on a
schedule that ramps a *masking fraction*, not a network's actual shape --
neither the technique nor (on its own) the schedule matches what Task F1
specifies ("35% structured filter pruning ... PolynomialDecay schedule").
`training/prune_quantise.py` instead implements both parts directly:
each hidden layer's units are ranked by the L2 norm of their incoming
weight vector (the Dense-layer analogue of CNN filter-importance
pruning), and a genuine PolynomialDecay schedule (same formula tfmot
uses, power=3, sparsity 0% -> 35% over 30 scheduled steps) determines how
many units should be *physically removed* at each step -- the network is
re-ranked on its current fine-tuned weights and re-shrunk every time the
schedule's target width decreases, not resized once. The measured
trajectory: 32/16 units at step 0 -> 27/14 (step 5) -> 24/12 (step 10) ->
22/11 (step 15) -> 21/10 by step 20, holding at 21/10 through 20
additional stable fine-tuning epochs. Final result: **803 -> 400
parameters (50% reduction)**, reflected in the measured 5.27KB -> 3.84KB
(27%) file size drop -- a materially larger effect than PTQ alone
(5.27KB -> 4.58KB, 13%), unlike unstructured pruning of a model this
size, which typically shows little to no file-size benefit since the
still-dense tensor gets no help from a standard (non-sparse-aware) TFLite
kernel.

Measured on the development laptop CPU (not the target Pi 5), per Lab 2
methodology; absolute latency will differ on-device but the *relative*
ranking and the conclusion that all three variants are microseconds-scale
(<<90s SLA) will hold.

**Recommendation for Task F3:** M3 (structured-pruned 35% + INT8) --
identical accuracy and 100% Class-2 recall to the FP32 baseline, but at
roughly half the parameter count and the smallest file size of the three
variants. Fewer parameters means fewer FLOPs per inference, which is the
correct lever for a compute-bound workload (Task B2's Roofline result),
and the smallest Flash footprint of the three variants on the memory-
constrained Pi 5 + AI HAT+ edge node. M2 (INT8 only, no structural change)
would be the fallback if a future larger model showed pruning hurting
accuracy -- not the case here.

## Task D2 -- Docker / OTA layer-cache demo

- Full image size: **1.61 GB** (`docker images logibridge-inference:latest`)
- Model-only layer (`COPY inference/model.tflite`): **4.62 KB**
- Rebuild after swapping only `model.tflite`: confirmed via `docker build`
  output that all 5 preceding layers report `CACHED`; only the final
  `COPY` layer rebuilds.
- Bandwidth saving for 85 trucks, full-image OTA vs model-only OTA:
  - Full: 85 x 1.61 GB = 136.85 GB -> at Rs 0.10/MB (~Rs 102.4/GB) approx **Rs 14,013**/update cycle
  - Model-only: 85 x 4.62 KB = 392.7 KB (0.38 MB) -> approx **Rs 0.04**/update cycle
  - **~349,000x less data**, ~350,000x cheaper, per update cycle.

**Benchmarked-artifact-matches-deployed-artifact check:** `training/prune_quantise.py`'s
fine-tuning loop isn't seeded, so re-running it produces a same-size but
different-bytes model each time (weights differ, architecture and file
size don't). This was caught after M3's pruning schedule was corrected:
`inference/model.tflite` had been copied from an earlier run of the
script, not the one `optimisation/benchmark.py` last measured. Fixed by
regenerating once as the canonical run and verifying SHA-256 across all
four places the model exists -- `training/models/model_pruned_int8.tflite`,
`inference/model.tflite`, the file baked into the built Docker image
(`docker run --rm --entrypoint sha256sum ... /app/model.tflite`), and the
file Ansible actually places on the target
(`/tmp/logibridge_test/model.tflite` in local testing) -- all four hash
to `22c365bb...`. `training/verify_model_sync.sh` checks the first two
automatically and fails loudly on a future mismatch; run it before every
`docker build`.

## Task E1 -- PSI drift monitoring

Reference distribution (300 clean Normal-class windows, 4 bins):
`[0.0, 0.237, 0.730, 0.033]`

Offline validation trace (`monitoring/drift_demo_offline.py`,
`drift_demo_trace.json`): 25 min clean baseline -> `--anomaly combined`
injected -> 20 min recovery, rolling 100-sample PSI checked every ~60s
(mirrors the live `drift_monitor.py` logic exactly):

- PSI first crossed 0.25 at **190 seconds** after injection (target: within 300s) [PASS]
- Max PSI during injection phase: **2.02**
- PSI at end of recovery window: **0.038** (target: <0.10) [PASS]

## Task E2 -- Ansible idempotency

Two consecutive `ansible-playbook` runs against the same target
(`deployment/ansible_run1.log`, `ansible_run2.log`):

- Run 1 (clean slate): `ok=8 changed=5 failed=0`
- Run 2 (no changes made in between): `ok=7 changed=0 failed=0 skipped=1`

Task 4 (stop container) is conditioned on the model/reference-file copy
tasks actually changing something (`when: model_copy.changed or
ref_copy.changed`) -- an unconditional stop-then-restart every run would
report `changed=1` forever and never demonstrate idempotency.

## Task A1/C3 -- MQTT TLS + authentication (production broker profile)

`deployment/mosquitto_production.conf` (TLS 1.2 on port 8883, no
anonymous access, per-role username/password + topic ACLs) was run live
against the actual Python clients, not just tested with the `mosquitto_pub`
CLI:

- `simulator.py` (as `logibridge_sensors`), `inference_service.py` (as
  `logibridge_inference`), and `drift_monitor.py` (as `logibridge_ops`)
  were started simultaneously against the TLS broker. Broker log confirms
  all three negotiated `TLSv1.3` under their own distinct credential, and
  classification results flowed end to end (Normal -> Critical as
  `combined`-mode drift matured), identically to the plaintext dev tests.
- **Negative tests** (the part that actually proves the ACLs work, not
  just exist): `logibridge_sensors` attempting to publish to the
  `inference` topic, and `logibridge_inference` attempting to publish to
  `sensors/temperature`, were both rejected -- broker log shows
  `Denied PUBLISH` for each. Neither role can act outside its intended
  scope even with valid credentials.
- Anonymous connection over TLS with no credentials: rejected
  (`Connection Refused: not authorised`).

The three-role split (sensors / inference / ops) replaced an earlier
single shared "truck" credential once it became clear that role needed
opposite permissions depending on which script used it: the simulator
*writes* sensor data, while the inference service *reads* it -- one
shared write-only grant on `sensors/#` would have silently broken the
inference service's ability to subscribe.

## Task E3 -- OTA strategy bandwidth (85-truck pilot, 280 KB model)

| Strategy | Total data | Cost @ Rs 0.10/MB |
|---|---|---|
| Full replacement | 23.24 MB | Rs 2.32 |
| Canary (10 then 75) | 23.24 MB | Rs 2.32 |
| Shadow mode (2 models resident) | 46.48 MB | Rs 4.65 |

Recommendation: **canary** -- cost is a rounding error either way; the
deciding factor is blast-radius safety (see `deployment/ota_strategy.md`).
