#!/usr/bin/env bash
# Run a vllm-based python script in this project's uv .venv on the current
# single-GPU WSL2 server. Encapsulates the fresh-server environment fixes so
# experiments are reproducible:
#   - VLLM_WSL2_ENABLE_PIN_MEMORY=1  vllm disables pin_memory on WSL2 by default;
#     its V1 UvaBuffer then hard-fails ("UVA is not available"). Pinned memory
#     works on this box (kernel 6.18), so enable it.
#   - CC=gcc                          Triton JIT-compiles CUDA utils at runtime.
#   - CUDA_HOME + PATH                nvcc from the pip nvidia-cuda-nvcc-cu13 wheel
#     (pip torch ships the CUDA runtime but not nvcc).
# Also requires (apt, one-time): ffmpeg (torchcodec/libavutil), build-essential,
# python3-dev (Python.h for Triton).
#
# Usage:  bash pilot/scripts/run_vllm.sh pilot/scripts/<script>.py [args...]
# Detached long run:
#   setsid bash pilot/scripts/run_vllm.sh <script>.py --n 200 > /tmp/run.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/../.."            # -> repo root (/home/max/thesis)
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export CC=gcc
# flashinfer JIT (sampler/attention kernels) fails to build here (cccl header vs
# nvcc 13.2 mismatch); flashinfer is uninstalled and Triton attention is forced.
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
# .venv/bin on PATH so subprocess JIT builders (flashinfer -> `ninja`) resolve.
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
exec .venv/bin/python "$@"
