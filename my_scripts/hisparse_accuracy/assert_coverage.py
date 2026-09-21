#!/usr/bin/env python3
"""Assert that a correctness run exercised its intended HiSparse path."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EVENT_RE = re.compile(r"HiSparse hybrid event=([a-z_]+)")
FREED_RE = re.compile(r"freed_c4_slots=(\d+)")


def analyze(mode: str, text: str) -> dict:
    counts = Counter(EVENT_RE.findall(text))
    freed_slots = sum(int(value) for value in FREED_RE.findall(text))
    failures: list[str] = []

    if mode == "hybrid-resident":
        if counts["resident_admit"] == 0:
            failures.append("no resident_admit event")
        if counts["resident_topk"] == 0:
            failures.append("no resident_topk event")
        if counts["demote"] != 0:
            failures.append(f"unexpected demote events: {counts['demote']}")
        if counts["host_topk"] != 0:
            failures.append(f"unexpected host_topk events: {counts['host_topk']}")
    elif mode == "hybrid-evict":
        for event in ("resident_admit", "mirror_complete", "demote", "host_topk"):
            if counts[event] == 0:
                failures.append(f"no {event} event")
        if freed_slots <= 0:
            failures.append("demotion did not report any freed C4 slots")
        positions = {
            event: text.find(f"event={event}")
            for event in ("resident_admit", "mirror_complete", "demote", "host_topk")
        }
        if all(position >= 0 for position in positions.values()):
            ordered = [positions[name] for name in positions]
            if ordered != sorted(ordered):
                failures.append(
                    "expected resident_admit -> mirror_complete -> demote -> host_topk"
                )
    elif mode not in ("baseline", "native"):
        failures.append(f"unsupported mode: {mode}")

    return {
        "mode": mode,
        "events": dict(sorted(counts.items())),
        "freed_c4_slots": freed_slots,
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--decode-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.decode_log.read_text(errors="replace")
    payload = analyze(args.mode, text)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
