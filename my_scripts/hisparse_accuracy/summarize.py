#!/usr/bin/env python3
"""Build a compact Markdown result and enforce the accuracy floor."""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-score", type=float, default=0.93)
    args = parser.parse_args()

    metadata = load(args.run_dir / "metadata.json")
    metrics = load(args.run_dir / "gsm8k-metrics.json")
    coverage = load(args.run_dir / "coverage-summary.json")
    score = float(metrics.get("score", metrics.get("mean_score", float("nan"))))
    score_ok = score >= args.min_score
    passed = score_ok and bool(coverage["passed"])
    report = f"""# DSV4 HiSparse accuracy result

| Field | Value |
|---|---|
| Hardware | `{metadata['hardware_profile']}` |
| Mode | `{metadata['mode']}` |
| Git SHA | `{metadata['git_sha']}` |
| GSM8K examples | {metadata['num_examples']} |
| GSM8K shots | {metadata['num_shots']} |
| GSM8K threads | {metadata['num_threads']} |
| GSM8K score | {score:.4f} |
| Required score | {args.min_score:.4f} |
| Coverage passed | {coverage['passed']} |
| Overall passed | {passed} |

## HiSparse coverage

```json
{json.dumps(coverage, indent=2, sort_keys=True)}
```

## GPU inventory

```text
{metadata['gpu_inventory']}
```
"""
    (args.run_dir / "report.md").write_text(report)
    print(report)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
