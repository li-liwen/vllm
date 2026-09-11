#!/usr/bin/env bash
# Phase 4 boot: full model, 8K context, eager, no MTP, TP4xPP2.
# Run on the host; uses glm53f:latest from scripts/build.sh.
set -euo pipefail
docker rm -f vllm-glm53f 2>/dev/null || true
docker run -d --name vllm-glm53f \
  --ipc=host --cpuset-cpus=0-23 --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd \
  $(for i in $(seq 128 135); do echo --device /dev/dri/renderD$i; done) \
  -v /home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local:/model:ro \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e VLLM_API_KEY="${VLLM_API_KEY:?set VLLM_API_KEY in env}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -p 8006:8000 \
  glm53f:latest serve /model \
  --served-model-name glm5.3-flash-autoround \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --enforce-eager \
  --load-format safetensors \
  --model-loader-extra-config '{"safetensors_use_index": true}' \
  --trust-remote-code
echo "container vllm-glm53f started; logs: docker logs -f vllm-glm53f"
