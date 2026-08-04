#!/usr/bin/env bash
# One-time (and safely re-runnable) project setup for LogiEdge.
#
# Creates the venv, installs dependencies, and -- the part that actually
# breaks when this repo is cloned to a different path or a different
# machine -- rewrites deployment/inventory.ini's ansible_python_interpreter
# to the *current* machine's absolute venv path, instead of leaving
# whatever path was hardcoded by whoever committed it last.
#
# Usage:
#   ./setup.sh              # venv + deps + fix inventory.ini
#   ./setup.sh --with-broker  # also install/start the local mosquitto broker

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(pwd)"

echo "== LogiEdge setup =="
echo "repo root: $REPO_ROOT"

# 1. venv
if [ ! -d ".venv" ]; then
  echo "-- creating venv (.venv)"
  python3.11 -m venv .venv
else
  echo "-- .venv already exists, reusing"
fi

VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: expected venv interpreter not found at $VENV_PYTHON" >&2
  exit 1
fi

# Deliberately NOT using `source .venv/bin/activate` + bare `pip`/
# `ansible-playbook` here. If this venv was ever created under a
# different path (e.g. the repo folder was renamed/moved after `python3
# -m venv .venv` ran), every installed entry-point script -- pip,
# ansible-playbook, ansible-galaxy, anything with a `#!/path/to/python`
# shebang baked in at install time -- still points at the OLD path and
# silently fails or falls back to some other Python found on PATH
# (conda, system python, ...) instead of this venv. Invoking pip via
# "$VENV_PYTHON" -m pip sidesteps that: python3 itself is a symlink, not
# a shebang script, so it always resolves correctly regardless of any
# past rename. Reinstalling with --force-reinstall afterwards also
# regenerates every entry-point script's shebang against the CURRENT
# path, fixing direct `pip`/`ansible-playbook` invocations too.
echo "-- installing requirements"
"$VENV_PYTHON" -m pip install --quiet --upgrade pip
"$VENV_PYTHON" -m pip install --quiet --force-reinstall --no-deps -r requirements.txt
"$VENV_PYTHON" -m pip install --quiet -r requirements.txt

echo "-- verifying entry-point scripts point at this machine's venv"
for bin in pip ansible-playbook ansible-galaxy; do
  bin_path=".venv/bin/$bin"
  if [ -f "$bin_path" ]; then
    shebang="$(head -1 "$bin_path")"
    case "$shebang" in
      "#!$VENV_PYTHON"*) echo "   ok: $bin" ;;
      *) echo "   WARNING: $bin shebang ($shebang) does not match $VENV_PYTHON -- run this script again or recreate .venv" >&2 ;;
    esac
  fi
done

# 2. Fix deployment/inventory.ini's ansible_python_interpreter to match
#    THIS machine's actual venv path, regardless of whatever path was
#    committed by whoever last edited the file.
INVENTORY="deployment/inventory.ini"
if [ -f "$INVENTORY" ]; then
  echo "-- syncing ansible_python_interpreter in $INVENTORY -> $VENV_PYTHON"
  # -i.bak works on both GNU and BSD/macOS sed; # as delimiter since paths contain /
  sed -i.bak -E "s#(ansible_python_interpreter=)\S+#\1${VENV_PYTHON}#" "$INVENTORY"
  rm -f "${INVENTORY}.bak"
  grep ansible_python_interpreter "$INVENTORY"
else
  echo "WARNING: $INVENTORY not found, skipping interpreter path fix" >&2
fi

# 3. Optional: local Mosquitto broker
if [[ "${1:-}" == "--with-broker" ]]; then
  if ! command -v mosquitto >/dev/null 2>&1; then
    echo "-- installing mosquitto via Homebrew"
    brew install mosquitto
  fi
  if pgrep -x mosquitto >/dev/null 2>&1; then
    echo "-- mosquitto already running, leaving it as-is"
  else
    echo "-- starting local dev broker (deployment/mosquitto_local.conf)"
    "$(brew --prefix mosquitto)/sbin/mosquitto" -c deployment/mosquitto_local.conf -d
  fi
fi

echo "== setup complete =="
echo "Activate the venv in new terminals with: source .venv/bin/activate"
