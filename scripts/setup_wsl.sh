#!/usr/bin/env bash
# Engineered by uncoalesced.
#
# One-shot environment setup for Dipteran on Linux/WSL2. Covers the two
# failure modes that otherwise take several rounds of manual debugging to
# hit: installing into a venv on a DrvFs-mounted Windows drive (uv's
# hardlink-then-copy fallback dies with "Operation not permitted" on a
# random package, every run), and git's "dubious ownership" refusal when
# cloning across that same mount. Safe to re-run.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Refuse to run from a Windows-mounted drive under WSL2. This has to be
#    a hard stop, not a warning -- the failure it prevents doesn't show up
#    until minutes into `uv pip install`, on whichever package happens to
#    lose the race, which makes it look like a random flaky error instead
#    of what it is.
if [[ "$REPO_DIR" == /mnt/* ]]; then
  cat >&2 <<EOF
This repo is at $REPO_DIR -- a Windows drive mounted into WSL2 (DrvFs).
Installing dependencies there fails unpredictably; that's a DrvFs
limitation, not a bug in this script or in uv. Move the repo onto WSL2's
own filesystem, then run this script again from there:

    git config --global --add safe.directory '*'
    git clone $REPO_DIR ~/dipteran
    cd ~/dipteran
    bash scripts/setup_wsl.sh

EOF
  exit 1
fi

cd "$REPO_DIR"
echo "[setup] repo: $REPO_DIR"

# 2. venv (idempotent -- leaves an existing one alone)
if [[ ! -d .venv ]]; then
  echo "[setup] creating .venv"
  uv venv --python 3.12 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Dependencies. requirements.txt first, then torch separately from the
#    CUDA 12.8 index -- see the comment at the top of requirements.txt for
#    why it can't be merged into the same install.
echo "[setup] installing requirements.txt"
uv pip install -r requirements.txt

echo "[setup] installing torch (CUDA 12.8 index)"
if ! UV_HTTP_TIMEOUT=300 uv pip install torch==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128; then
  echo "[setup] torch install timed out once (large download, pypi.nvidia.com" >&2
  echo "[setup] has been flaky) -- retrying with a longer timeout" >&2
  UV_HTTP_TIMEOUT=600 uv pip install torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi

cat <<EOF

[setup] Python environment ready.
[setup] Next steps (not automated -- SUMO needs sudo, the others need a GPU
[setup] and download several GB):

    sudo apt-get install sumo sumo-tools sumo-doc   # SUMO, not pip-installable
    python -m dipteran.roads                        # builds the road network
    python -m dipteran.connectome                   # fetches the FlyWire connectome

    python main.py                                  # or just try mock mode now --
                                                      # no GPU/SUMO/connectome needed
EOF
