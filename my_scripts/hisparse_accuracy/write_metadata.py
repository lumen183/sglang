#!/usr/bin/env python3
"""Write reproducibility metadata for one accuracy run."""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hisparse-config", default="")
    parser.add_argument("--num-examples", type=int, required=True)
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--num-shots", type=int, required=True)
    args = parser.parse_args()

    payload = {
        "hardware_profile": args.hardware,
        "mode": args.mode,
        "model_path": args.model,
        "dataset_path": args.dataset,
        "hisparse_config": args.hisparse_config,
        "num_examples": args.num_examples,
        "num_threads": args.num_threads,
        "num_shots": args.num_shots,
        "git_sha": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "gpu_inventory": command(
            "nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "transfer_backend": os.environ.get("TRANSFER_BACKEND", "nixl"),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
