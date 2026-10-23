# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject devices outside the archived eight-GPU SM80 deployment contract."""

import os
import sys

import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    raise RuntimeError("This deployment requires NVIDIA CUDA SM80")
if torch.accelerator.device_count() != 8:
    raise RuntimeError("This deployment is qualified for eight SM80 GPUs")
for device in range(torch.accelerator.device_count()):
    if current_platform.get_device_capability(device) != (8, 0):
        raise RuntimeError("This deployment supports exactly CUDA capability 8.0")

os.execv("/usr/local/bin/vllm", ["vllm", *sys.argv[1:]])
