"""Check the actual Python, GPU and disk configuration used by this project."""
import importlib.metadata
import json
import shutil
import sys

import torch


def main():
    report = {"python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "chess", "requests", "zstandard", "threadpoolctl")},
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "free_disk_gb": round(shutil.disk_usage('.').free / 1e9, 2)}
    if torch.cuda.is_available():
        report.update({"gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)})
        # Execute a real GPU operation, not just a device enumeration.
        probe = torch.ones(16, device="cuda").sum().item()
        assert probe == 16
    print(json.dumps(report, indent=2))
    if not report["cuda_available"]:
        raise SystemExit("CUDA unavailable. Install a CUDA-enabled PyTorch build before training.")


if __name__ == "__main__":
    main()
