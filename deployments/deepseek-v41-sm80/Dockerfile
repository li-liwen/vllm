FROM lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3 AS vendor
FROM public.ecr.aws/docker/library/ubuntu@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-dev build-essential ninja-build ca-certificates libnuma1 numactl libgomp1 \
    libibverbs1 librdmacm1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=vendor /usr/local /usr/local
COPY --from=vendor /opt /opt
COPY --from=vendor /vllm-workspace /vllm-workspace
RUN ln -s /usr/local/cuda-13.0 /etc/alternatives/cuda && \
    ln -s /usr/local/cuda-13.0 /etc/alternatives/cuda-13
ENV PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
    CUDA_HOME=/usr/local/cuda NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY patches /opt/deepseek-sm80/snapshot/patches
COPY install.py entrypoint.py /opt/deepseek-sm80/snapshot/
RUN uv venv --python /usr/bin/python3 --system-site-packages /opt/deepseek-sm80/.venv && \
    /opt/deepseek-sm80/.venv/bin/python /opt/deepseek-sm80/snapshot/install.py
ENV VLLM_DEEPINFER_SM80_MHC=1
WORKDIR /vllm-workspace
ENTRYPOINT ["/opt/deepseek-sm80/.venv/bin/python", "/opt/deepseek-sm80/snapshot/entrypoint.py"]
