# Task E3 -- OTA Strategy Selection

**Given:** model updates every 6 weeks, model file 280 KB (INT8 TFLite),
M2M SIM at ₹0.10/MB, 85-truck pilot fleet.

## Bandwidth cost per update cycle

| Strategy | Trucks updated | Data/truck | Total data | Cost @ ₹0.10/MB |
|---|---|---|---|---|
| Full replacement | 85 (all, at once) | 280 KB | 23,800 KB (23.24 MB) | **₹2.32** |
| Canary (10 first, then 75) | 10, then 75 if canary passes | 280 KB | 23,800 KB (23.24 MB), same total, staged over ~1-2 days | **₹2.32** |
| Shadow mode (new model runs alongside old, no cutover yet) | 85 (shadow model shipped to all, both models resident) | 560 KB (old + new model both present) | 47,600 KB (46.48 MB) | **₹4.65** |

At 280 KB/truck, bandwidth cost is trivial under *any* strategy (₹2-5 per
6-week cycle for the whole pilot fleet) -- this is the payoff of shipping a
quantised 280 KB model instead of the ~1.6 GB Docker image (see Task D2's
layer-cache demo: full-image OTA would cost **~₹13,700 per truck per
update**, ~4,700x more). Cost is therefore not the deciding factor between
the three strategies; safety and rollback risk are.

## Recommendation: Canary (10 trucks first)

- **For:** Cold-chain cargo is safety-critical -- a bad model pushed
  fleet-wide could silently degrade Class 2 (Critical) recall and cause a
  repeat of the ₹28 lakh spoilage incident, this time *caused by* an update
  rather than caught by one. Canary deployment (10 of 85 trucks, ~12% of
  fleet) limits blast radius: if the new model's Critical recall regresses
  on real field data, only 10 trucks are exposed before rollback, not 85.
  Given the seven documented rural connectivity gaps, canary trucks should
  be selected from routes with *good* connectivity so their drift/alert
  telemetry reaches the ops centre quickly enough to validate the rollout
  within the 6-week cycle.

- **Against full replacement:** Zero rollback safety net. A regression is
  discovered only after all 85 trucks are affected, and -- given rural
  connectivity gaps -- the ops centre may not even find out for up to 90
  minutes per affected truck. Cost savings versus canary are negligible
  (both ~₹2.32/cycle) so there is no economic argument for accepting this
  risk.

- **Against shadow mode:** Shadow mode (running old and new models
  side-by-side, comparing outputs, without acting on the new model's
  predictions) is the safest option in principle, but it means the *old*
  model is still the one issued alerts during the entire evaluation window
  -- so it provides no benefit if the trigger for the update was a known
  bug or missed-detection pattern in the old model. It also roughly doubles
  Flash usage on a memory-constrained edge node (two INT8 models resident
  instead of one) for no operational benefit once the canary approach
  already provides a real-world validation signal at low blast radius.
  Shadow mode is worth reserving for a *major* architecture change (e.g.
  swapping the MLP for a different feature set entirely), not a routine
  6-week retraining cycle.

**Rollout plan:** canary (10 trucks, ~3-5 days) -> monitor Class 2 recall
and PSI drift on canary trucks via `monitoring/drift_monitor.py` -> if
stable, roll out to remaining 75 trucks via the same
`deployment/logibridge_deploy.yml` playbook, batched by route to keep
any single Ansible run's blast radius manageable.
