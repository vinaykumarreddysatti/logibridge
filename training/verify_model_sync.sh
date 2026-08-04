#!/usr/bin/env bash
# Guards against exactly the bug this fixes: prune_quantise.py's
# fine-tuning loop isn't seeded, so re-running it produces a
# same-size-but-different-bytes model.tflite. If inference/model.tflite
# was copied from an earlier run, it silently stops matching the file
# optimisation/benchmark.py actually measured. Run this before
# `docker build` -- it fails loudly instead of deploying a model that
# doesn't match the reported benchmark numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

CANONICAL="training/models/model_pruned_int8.tflite"
DEPLOYED="inference/model.tflite"

if [ ! -f "$CANONICAL" ] || [ ! -f "$DEPLOYED" ]; then
  echo "missing model file(s) -- run training/prune_quantise.py first" >&2
  exit 1
fi

CANONICAL_SHA=$(shasum -a 256 "$CANONICAL" | cut -d' ' -f1)
DEPLOYED_SHA=$(shasum -a 256 "$DEPLOYED" | cut -d' ' -f1)

if [ "$CANONICAL_SHA" != "$DEPLOYED_SHA" ]; then
  echo "MISMATCH: $DEPLOYED does not match $CANONICAL" >&2
  echo "  $CANONICAL_SHA  $CANONICAL" >&2
  echo "  $DEPLOYED_SHA  $DEPLOYED" >&2
  echo "run: cp $CANONICAL $DEPLOYED" >&2
  exit 1
fi

echo "OK: $DEPLOYED matches $CANONICAL ($CANONICAL_SHA)"
