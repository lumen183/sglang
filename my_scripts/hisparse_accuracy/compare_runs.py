#!/usr/bin/env python3
"""Compare two completed accuracy runs, including deterministic prompts."""

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def response_text(record: dict):
    return record.get("response", {}).get("text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-score-delta", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ref_metrics = load_json(args.reference / "gsm8k-metrics.json")
    cand_metrics = load_json(args.candidate / "gsm8k-metrics.json")
    ref_score = float(ref_metrics.get("score", ref_metrics.get("mean_score")))
    cand_score = float(cand_metrics.get("score", cand_metrics.get("mean_score")))
    ref_fixed = load_jsonl(args.reference / "fixed-prompts.jsonl")
    cand_fixed = load_jsonl(args.candidate / "fixed-prompts.jsonl")
    fixed_diffs = []
    for ref, cand in zip(ref_fixed, cand_fixed):
        if response_text(ref) != response_text(cand):
            fixed_diffs.append(ref["index"])

    failures = []
    if len(ref_fixed) != len(cand_fixed):
        failures.append("fixed prompt counts differ")
    if fixed_diffs:
        failures.append(f"fixed prompt output differs at indices {fixed_diffs}")
    if abs(cand_score - ref_score) > args.max_score_delta:
        failures.append(
            f"score delta {cand_score - ref_score:+.4f} exceeds "
            f"{args.max_score_delta:.4f}"
        )
    payload = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_score": ref_score,
        "candidate_score": cand_score,
        "score_delta": cand_score - ref_score,
        "fixed_prompt_differences": fixed_diffs,
        "passed": not failures,
        "failures": failures,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
