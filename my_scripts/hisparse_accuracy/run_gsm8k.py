#!/usr/bin/env python3
"""Run the repository GSM8K evaluator and collect its report artifacts."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from sglang.test.run_eval import run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--num-shots", type=int, default=20)
    args = parser.parse_args()

    dataset = (
        args.data_path
        if args.data_path.is_file()
        else args.data_path / "test.jsonl"
    )
    eval_args = SimpleNamespace(
        base_url=args.base_url.rstrip("/"),
        host=None,
        port=None,
        model=args.model,
        eval_name="gsm8k",
        api="completion",
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        num_shots=args.num_shots,
        gsm8k_data_path=str(dataset),
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        repeat=1,
        return_latency=False,
    )
    metrics = run_eval(eval_args)
    (args.output_dir / "gsm8k-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )

    stem = f"gsm8k_{args.model.replace('/', '_')}"
    for suffix in ("html", "json"):
        source = Path("/tmp") / f"{stem}.{suffix}"
        shutil.copy2(source, args.output_dir / f"gsm8k-report.{suffix}")


if __name__ == "__main__":
    main()
