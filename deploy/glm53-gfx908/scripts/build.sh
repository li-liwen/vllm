#!/usr/bin/env bash
# Build the glm53f wheel + serving images from the feature branch.
# Usage: ./build.sh [commit]   (default: HEAD of feat/glm53-flash-gfx908)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
commit=${1:-$(git -C "$repo" rev-parse HEAD)}
ctx=/tmp/glm53-build
tag_source="glm53f-build:$(echo "$commit" | cut -c1-12)"

rm -rf "$ctx"
mkdir -p "$ctx/vllm-src/deploy/glm53-gfx908/docker"
git -C "$repo" archive "$commit" | tar -x -C "$ctx/vllm-src"
# Dockerfile must be inside the build context
cp "$repo/deploy/glm53-gfx908/docker/Dockerfile.glm53" \
    "$ctx/vllm-src/deploy/glm53-gfx908/docker/"

echo "== stage1: build wheel ($commit)"
docker build -f "$ctx/vllm-src/deploy/glm53-gfx908/docker/Dockerfile.glm53" \
    -t "$tag_source" --target build "$ctx"

echo "== serve image"
cat > "$ctx/Dockerfile.serve" <<EOF
FROM $tag_source AS serve
LABEL org.opencontainers.image.source="li-liwen/vllm feat/glm53-flash-gfx908" \\
      org.opencontainers.image.revision="$commit"
ENV VLLM_GFX908_HIP_BUILD_DIR=/opt/vllm-gfx908-ext \\
    HF_HOME=/huggingface \\
    HF_HUB_OFFLINE=1 \\
    TRANSFORMERS_OFFLINE=1
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/vllm"]
CMD ["serve", "/model"]
EOF
docker build -t glm53f:latest -f "$ctx/Dockerfile.serve" "$ctx"
echo "built: $tag_source and glm53f:latest"
