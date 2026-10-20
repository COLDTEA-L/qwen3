"""Bounded log summary only: no Torch imports, requests, signals or mutations."""
import argparse
import re
from pathlib import Path


def summarize(text, request_id=None):
    registered = re.findall(r"Registering KV cache: ptr=(0x[0-9a-fA-F]+)", text)
    bad = sum(int(ptr, 16) % (2 * 1024 * 1024) != 0 for ptr in registered)
    block = re.search(r"--block-size\s+(\d+)", text)
    capacity = re.search(r"--num-gpu-blocks-override\s+(\d+)", text)
    messages = [line for line in text.splitlines() if request_id is None or request_id in line]
    successes = set()
    failures = set()
    for line in messages:
        if "KV cache transfer for request " in line and " took " in line:
            match = re.search(r"KV cache transfer for request (\S+) took", line)
            if match:
                successes.add(match[1])
        if "Mooncake transfer failed for request " in line:
            match = re.search(r"Mooncake transfer failed for request (\S+)", line)
            if match:
                failures.add(match[1])
        if "CAM82_KV_TRANSFER_FAILED request=" in line:
            match = re.search(r"CAM82_KV_TRANSFER_FAILED request=(\S+)", line)
            if match:
                failures.add(match[1])
    return [
        f"logged_flags: block_size={block[1] if block else 'missing'} "
        f"num_blocks_override={capacity[1] if capacity else 'missing'}",
        f"registration_records={len(registered)} unaligned_records={bad} "
        f"guard_pass_records={text.count('CAM82_KV_REGIONS_OK')} "
        f"allocation_pass_records={text.count('CAM82_KV_ALLOCATION_OK')}",
        f"request_ids_with_transfer_success={len(successes)} "
        f"request_ids_with_transfer_failure={len(failures)} "
        f"fail_stop_seen={'CAM82_KV_FAIL_STOP' in text}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=("228", "238"))
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--request-id")
    args = parser.parse_args()
    for role in (("p", "d228") if args.host == "228" else ("d238",)):
        logs = list(args.results.glob(role + "-off-*.log"))
        if not logs:
            print(f"{role}: no off log found")
            continue
        # Names contain the launch timestamp. An older process appending to its
        # log must not outrank a newer startup failure by mtime.
        log = max(logs, key=lambda path: path.name)
        print(f"{role}: {log.name} (latest launch filename, not a live-process health check)")
        for line in summarize(log.read_text(errors="replace"), args.request_id):
            print("  " + line)


if __name__ == "__main__":
    main()
