#!/bin/bash
# Wrapper script to set up CUDA library paths for faster-whisper GPU support

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the nvidia library paths from pip-installed packages
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
CUDNN_LIB="$PYTHON_SITE/nvidia/cudnn/lib"
CUBLAS_LIB="$PYTHON_SITE/nvidia/cublas/lib"

# Set LD_LIBRARY_PATH if the libraries exist
if [ -d "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:$LD_LIBRARY_PATH"
fi

exec python3 "$SCRIPT_DIR/dictate.py" "$@"
