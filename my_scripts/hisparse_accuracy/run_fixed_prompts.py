#!/usr/bin/env python3
"""Run a small deterministic prompt set and preserve complete responses."""

import argparse
import json
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
        raise ValueError("--prompts must contain a JSON list of strings")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for index, prompt in enumerate(prompts):
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "max_new_tokens": args.max_new_tokens,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 0,
            }
            started = time.perf_counter()
            response = requests.post(
                args.base_url.rstrip("/") + "/generate",
                json=payload,
                timeout=args.timeout,
            )
            latency = time.perf_counter() - started
            response.raise_for_status()
            body = response.json()
            record = {
                "index": index,
                "prompt": prompt,
                "latency_seconds": latency,
                "response": body,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"fixed prompt {index + 1}/{len(prompts)} complete")


if __name__ == "__main__":
    main()
