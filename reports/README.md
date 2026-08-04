# Reports folder

This folder is where `phase1_report.pdf`, `phase2_report.pdf`, and
`final_report.pdf` belong for submission (Section 5, Deliverable 3).

## What's here now vs. what you need to add

- `report_data_appendix.md` -- every number in this file came from actually
  running the code in this repo (training runs, benchmarks, the drift
  demo, the Ansible runs, etc.). It's real, measured evidence, not
  estimates.
- `pipeline_mapping.md` -- a first-pass Task D3 mapping.
- The `.md` drafts in `scenario_architecture/`, `hardware/`, and
  `deployment/` are similarly evidence-backed first drafts for their
  respective report sections.

**None of these are the Final Report.** The assignment is explicit that:

> "Generating the Final Report exclusively through an AI writing tool
> without substantial original analysis and measured data" constitutes
> academic misconduct, and you must be able to explain any component of
> your submission in a follow-up viva.

The data in this appendix is genuine (every model was actually trained,
every benchmark actually run, every Ansible playbook actually executed
twice) -- that's the "measured data" half of the requirement. The
"substantial original analysis" half is yours to write: the 5-section,
2,500-3,000 word Final Report needs your own explanation of *why* these
numbers matter, your own phrasing, and in particular Section 5's "one
genuine technical difficulty" and "one architectural change you would
make" are personal reflections that only you can write honestly.

Two difficulties you may want to draw on for that section, since they're
real ones this build surfaced:

1. The assignment's linear temperature-drift rate (0.08 C/reading) run
   for the full 15-minute Warning-scenario duration accumulates to a
   physically implausible ~70 C above setpoint without a cap, which would
   put most of a "Warning"-labelled run's windows into Critical territory
   under the Section 2 class thresholds. This repo resolves it by capping
   simulator drift at a physically plausible 8 C
   (`data_pipeline/simulator.py::MAX_TEMP_DRIFT_C`) while keeping the
   assignment's mode-based labelling convention -- worth explaining in
   your own words, and a legitimate candidate for "an architectural
   change you would make" (e.g. moving to per-window instantaneous
   labelling in a v2, which would need a redesigned drift-duration
   schedule to still yield enough Warning-class samples).

2. Task F1 says "structured filter pruning (PolynomialDecay schedule),"
   but `tensorflow_model_optimization`'s standard API
   (`prune_low_magnitude`) performs *unstructured* per-weight pruning on
   a schedule that ramps a masking fraction, not the network's actual
   shape -- it zeros individual weights inside a dense matrix of
   unchanged shape, which gets no real size/latency benefit on a
   non-sparse-aware TFLite kernel, and isn't the technique the task
   names. There's also no direct notion of a "filter" in a Dense/MLP
   architecture the way there is in a CNN. `training/prune_quantise.py`
   resolves both parts directly: each hidden layer's units are ranked by
   the L2 norm of their incoming weight vector, and a real
   PolynomialDecay curve (power 3, matching tfmot's own formula) decides
   how many units to *physically remove* at each scheduled step,
   re-ranking on the network's current fine-tuned weights every time the
   target width drops (803 -> 400 parameters by the end of the
   schedule). Worth explaining in your own words why "the same formula,
   applied to unit count instead of a weight mask" is a materially
   different (and, for this architecture, more correct) implementation.
