# Task D3 -- 10-Stage Edge ML Pipeline Mapping (evidence draft)

> Rewrite in your own words for Final Report Section 3.

1. **Data collection** -- `data_pipeline/simulator.py` publishes temperature
   (1 Hz), vibration (0.5 Hz) and door events over MQTT, standing in for
   the physical sensors on the refrigerated compartment.
2. **Data labelling** -- `training/generate_dataset.py` runs the simulator
   in three scenario modes (`none`/`temp_drift`/`combined`) for fixed
   durations and labels every window from a run with the class that
   scenario represents (Task D1).
3. **Preprocessing** -- `data_pipeline/preprocessing.py` applies a 5-sample
   moving average, then extracts a 6-value feature vector per 30s window
   (10s step): temp mean/std/rate-of-change, vibration RMS/peak/kurtosis.
4. **Feature normalisation** -- stats computed once from clean Normal-class
   windows only, frozen to `training_stats.npy`, and loaded (never
   recomputed) at inference time.
5. **Model training** -- `training/train_model.py` trains a 2-hidden-layer
   MLP (32/16 units, ReLU) to >88% held-out validation accuracy with 100%
   Critical-class recall.
6. **Model conversion/optimisation** -- `training/convert_ptq.py` and
   `training/prune_quantise.py` produce the FP32 baseline, INT8 PTQ, and
   35%-pruned+INT8 TFLite variants (Task F1).
7. **Packaging** -- `inference/Dockerfile` containerises the preprocessing
   + inference pipeline, with `MODEL_PATH` swappable via environment
   variable without a rebuild.
8. **Deployment** -- `deployment/logibridge_deploy.yml` (Ansible) copies
   the model/reference files to `/opt/logibridge`, pulls the image from
   the registry, and (re)starts the container -- idempotently.
9. **Serving/inference** -- `inference/inference_service.py` runs on the
   truck's edge node, classifying each 30s window in real time and
   publishing results to `logibridge/trucks/{id}/inference`; Warning/
   Critical results are also written to a durable local alert log that
   survives connectivity gaps.
10. **Monitoring/feedback** -- `monitoring/drift_monitor.py` computes
    rolling PSI on inference confidence scores against a frozen
    `reference_dist.json`, alerting operations when PSI > 0.25 -- closing
    the loop back to stage 6 (re-optimisation/retraining) via the OTA
    strategy in `deployment/ota_strategy.md`.
