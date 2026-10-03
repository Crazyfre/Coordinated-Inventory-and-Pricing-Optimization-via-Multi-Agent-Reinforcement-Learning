#!/usr/bin/env bash
# Resumable installer for CUDA PyTorch (cu130, Python 3.14, Windows).
# Survives connection drops: re-launching this script resumes the partial download.
#
# Usage:  bash scripts/install_cuda_torch.sh
set -u

WHEEL_URL="https://download.pytorch.org/whl/cu130/torch-2.14.1%2Bcu130-cp314-cp314-win_amd64.whl"
EXPECTED_SIZE=1998854405
# NOTE: pip requires the canonical wheel filename; do not rename this.
WHEEL_FILE=".venv/torch-2.14.1+cu130-cp314-cp314-win_amd64.whl"
LOG_FILE=".venv/curl.log"

mkdir -p .venv

download() {
    echo "[$(date '+%H:%M:%S')] downloading wheel (resumes if partially downloaded) ..."
    curl -L --retry 5 --retry-delay 5 --retry-all-errors -C - -sS \
        -o "$WHEEL_FILE" "$WHEEL_URL"
}

# Keep retrying until the wheel is fully downloaded.
attempt=1
while true; do
    size=$(stat -c %s "$WHEEL_FILE" 2>/dev/null || echo 0)
    if [ "$size" -ge "$EXPECTED_SIZE" ]; then
        echo "[$(date '+%H:%M:%S')] download complete: $size / $EXPECTED_SIZE bytes"
        break
    fi
    if [ "$size" -gt 0 ]; then
        pct=$((size * 100 / EXPECTED_SIZE))
        echo "[$(date '+%H:%M:%S')] attempt $attempt: have $size / $EXPECTED_SIZE bytes (${pct}%)"
    fi
    download
    attempt=$((attempt + 1))
    sleep 5
done

# Sanity check: wheel must be a valid zip (use Python - Git Bash has no unzip).
if ! .venv/Scripts/python -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$WHEEL_FILE') else 1)"; then
    echo "ERROR: downloaded wheel is corrupt - delete $WHEEL_FILE and re-run" >&2
    exit 1
fi
echo "wheel integrity OK"

# Install into the project venv (replaces the CPU build).
echo "installing torch (this replaces the CPU build) ..."
.venv/Scripts/python -m pip install --force-reinstall --no-deps "$WHEEL_FILE" || exit 1

# Verify CUDA availability.
.venv/Scripts/python - <<'EOF'
import torch
print("torch      ", torch.__version__)
print("CUDA build :", torch.version.cuda)
print("CUDA avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("GPU matmul :", (x @ x).sum().item() != 0)
EOF

echo "done."
