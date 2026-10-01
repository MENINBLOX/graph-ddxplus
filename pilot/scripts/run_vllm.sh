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
# flashinfer is uninstalled (JIT build fails here: cccl header vs nvcc mismatch).
# Two independent flashinfer entry points must both be disabled, else vLLM 0.25
# hard-imports the missing module and EngineCore dies:
#   - attention: force a non-flashinfer backend (TORCH_SDPA needs no JIT and is proven
#     working on this box; TRITON_ATTN also works but requires the Triton JIT toolchain).
#   - sampler: VLLM_USE_FLASHINFER_SAMPLER=0 — greedy (temp=0) never needs its top-p/top-k
#     kernel; without this flag the Sampler still hard-imports flashinfer at init.
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
# .venv/bin on PATH so subprocess JIT builders (flashinfer -> `ninja`) resolve.
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
exec .venv/bin/python "$@"
