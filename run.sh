#!/usr/bin/env bash
# Wrapper to run safely on macOS M3

# Activate venv
source ~/.venv_audio_understanding/bin/activate

# Disable Hugging Face caching allocator warmup (prevents giant 9+ GiB MPS allocations)
export HF_HUB_DISABLE_CACHING_ALLOCATOR_WARMUP=1

# Optional: if MPS keeps crashing, uncomment this to force CPU fallback
# export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run script with all given args
python3 main.py "$@"
