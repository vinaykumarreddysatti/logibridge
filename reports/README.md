# Reports folder

This folder is where `final_report.pdf` belongs for submission. The
assignment's suggested repo layout (Deliverable 1) also lists
`phase1_report.pdf`/`phase2_report.pdf` alongside it, but Section 7
("Submission Requirements") is explicit that the actual graded
submission is exactly three items -- GitHub repo URL, Demo Video,
**Final Report PDF** (singular, filename `GROUPNO_LogiEdge_Final.pdf`) --
with no mention of separate phase deliverables. Phase 1/2, if they
exist, were earlier Continuous Evaluation checkpoints submitted
separately during the course, not part of this final repo submission.

## Status

- `report_data_appendix.md` -- finalized. Every number in this file came
  from actually running the code in this repo (training runs, benchmarks,
  the drift demo, the Ansible runs, etc.) and has been reconciled against
  the current `optimisation/results/benchmark_results.csv`.
- `pipeline_mapping.md` -- finalized Task D3 mapping.
- The supporting `.md` files in `scenario_architecture/`, `hardware/`, and
  `deployment/` (constraint analysis, hardware justification, system
  architecture, OTA strategy) are finalized, evidence-backed technical
  documents -- ready to submit as part of the repo.
- `final_report_draft.md` / `final_report.pdf` -- **still a draft.** This
  is the one file that isn't submission-ready. The assignment is explicit
  that:

  > "Generating the Final Report exclusively through an AI writing tool
  > without substantial original analysis and measured data" constitutes
  > academic misconduct, and you must be able to explain any component of
  > your submission in a follow-up viva.

  Sections 1-4 are close to final (condensed from the finalized supporting
  documents above, with real measured numbers). What's still outstanding:
  the frontmatter placeholders (`GROUPNO`, name(s), submission date), and
  Section 5's two `[DRAFT -- personalize before submission]` reflections
  -- "one genuine technical difficulty" and "one architectural change you
  would make" are personal reflections that only you can write honestly,
  and the ones currently in the draft are written in a draft voice, not
  yours. Delete the DRAFT NOTICE banner at the top only once that's done.

Two difficulties you may want to draw on for Section 5, since they're
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
