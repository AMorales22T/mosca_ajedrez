from __future__ import annotations

import os
import platform
import time

import torch


def report_device() -> dict:
    info = {
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    print(info)
    return info
