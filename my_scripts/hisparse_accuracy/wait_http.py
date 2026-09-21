#!/usr/bin/env python3
"""Wait for an HTTP endpoint while also watching a launcher PID."""

import argparse
import os
import time
import urllib.error
import urllib.request


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if not process_alive(args.pid):
            raise SystemExit(f"process {args.pid} exited before {args.url} became ready")
        try:
            with urllib.request.urlopen(args.url, timeout=5) as response:
                if response.status == 200:
                    print(f"ready: {args.url}")
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise SystemExit(f"timeout waiting for {args.url}: {last_error}")


if __name__ == "__main__":
    main()
