#!/usr/bin/env bash
# Quick correctness smoke test against the running server.
set -euo pipefail
HOST=${1:-http://localhost:8006}
KEY=${VLLM_API_KEY:?}
curl -s -H "Authorization: Bearer $KEY" "$HOST/v1/models" | python3 -m json.tool | head -8
echo "--- text completion:"
curl -s -H "Authorization: Bearer $KEY" "$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm5.3-flash-autoround","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":16,"temperature":0}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
