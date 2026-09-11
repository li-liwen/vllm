#!/usr/bin/env bash
# Phase 5: 1M context via TP2xPP4 (plan fallback #4) — partition 12,12,12,9.
set -euo pipefail
docker rm -f vllm-glm53f 2>/dev/null || true
docker run -d --name vllm-glm53f \
  --ipc=host --cpuset-cpus=0-23 --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd \
  $(for i in $(seq 128 135); do echo --device /dev/dri/renderD$i; done) \
  --restart unless-stopped \
  -v /home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local:/model:ro \
  -v /home/ubuntu/glm-5.3-flash/artifacts/vllm-compile-cache:/root/.cache/vllm \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  -e HSA_ENABLE_SVM=0 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY in env}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  -e VLLM_PP_LAYER_PARTITION=12,12,12,9 \
  -p 8006:8000 \
  glm53f:latest serve /model \
  --served-model-name glm5.3-flash-autoround \
  --tensor-parallel-size 2 --pipeline-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 1048576 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --load-format safetensors \
  --model-loader-extra-config '{"safetensors_use_index": true}' \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --trust-remote-code
echo "1M TP2xPP4 profile started"
