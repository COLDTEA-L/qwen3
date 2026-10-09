"""Fixed-arrival decode background plus long-prefix restore bursts.

This records client-side metrics. Cache hits are NOT proof of a remote pool hit;
correlate cached_tokens_details, MemCache metrics and device traces separately.
"""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path


def arrival_plan(duration, background_rps, burst_every, burst_size):
    if duration <= 0 or background_rps <= 0 or burst_every <= 0 or burst_size < 1:
        raise ValueError("Arrival parameters must be positive")
    requests = [(i / background_rps, "background", i)
                for i in range(math.ceil(duration * background_rps))
                if i / background_rps < duration]
    # First burst follows a short decode-only interval.
    for wave in range(1, math.ceil(duration / burst_every)):
        requests.extend((wave * burst_every, "restore", wave * burst_size + i)
                        for i in range(burst_size))
    return sorted(requests)


async def execute(args):
    import aiohttp
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    material = tokenizer.encode(
        "An agent maintains a long reasoning history, tool observations and task state. ",
        add_special_tokens=False,
    )
    prefix = (material * (args.prefix_tokens // len(material) + 1))[:args.prefix_tokens]
    background = material * (128 // len(material) + 1)
    plan = arrival_plan(args.duration, args.background_rps, args.burst_every, args.burst_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Output already exists; choose a new file for each run")
    semaphore = asyncio.Semaphore(args.max_concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    wall_start, start = time.time(), time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def request(arrival, kind, index):
            await asyncio.sleep(max(0, start + arrival - time.perf_counter()))
            rid = f"cam-{args.run_id}-{kind}-{index}"
            suffix = tokenizer.encode(f"Task {index}: continue your analysis.", add_special_tokens=False)
            tokens = prefix + suffix if kind == "restore" else suffix + background[:128]
            record = {"rid": rid, "kind": kind, "scheduled_s": arrival,
                      "wall_start_s": wall_start, "input_tokens": len(tokens)}
            async with semaphore:
                submitted = time.perf_counter()
                record["client_queue_ms"] = max(0, (submitted - start - arrival) * 1000)
                completion = 0
                first = None
                previous = None
                gaps = []
                meta = {}
                try:
                    payload = {"input_ids": tokens, "rid": rid, "stream": True,
                               "sampling_params": {"temperature": 0, "ignore_eos": True,
                                   "max_new_tokens": args.background_output_tokens
                                   if kind == "background" else args.restore_output_tokens}}
                    async with session.post(args.base_url.rstrip("/") + "/generate", json=payload) as response:
                        response.raise_for_status()
                        async for raw in response.content:
                            raw = raw.strip()
                            if not raw or raw.startswith(b":"):
                                continue
                            if raw.startswith(b"data:"):
                                raw = raw[5:].strip()
                            if raw == b"[DONE]":
                                continue
                            data = json.loads(raw)
                            meta = data.get("meta_info") or {}
                            now = time.perf_counter()
                            count = int(meta.get("completion_tokens", completion))
                            if count > completion:
                                if first is None:
                                    first = now
                                elif previous is not None:
                                    gaps.append((now - previous) * 1000 / (count - completion))
                                previous, completion = now, count
                    if first is None or completion < 1:
                        raise RuntimeError("No completion tokens received")
                    record.update(success=True, output_tokens=completion,
                                  ttft_ms=(first - submitted) * 1000,
                                  e2e_ms=(time.perf_counter() - submitted) * 1000,
                                  tpot_ms=(previous - first) * 1000 / (completion - 1)
                                  if completion > 1 else None,
                                  streamed_gap_per_token_ms=gaps,
                                  cached_tokens=meta.get("cached_tokens"),
                                  cached_tokens_details=meta.get("cached_tokens_details"))
                except Exception as error:
                    record.update(success=False, error=str(error),
                                  e2e_ms=(time.perf_counter() - submitted) * 1000)
                # Single event loop: this small synchronous append cannot interleave.
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        await asyncio.gather(*(request(*item) for item in plan))
    print(f"Saved {len(plan)} requests to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--background-rps", type=float, default=0.5)
    parser.add_argument("--burst-every", type=float, default=10)
    parser.add_argument("--burst-size", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--background-output-tokens", type=int, default=512)
    parser.add_argument("--restore-output-tokens", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if min(args.prefix_tokens, args.background_output_tokens, args.restore_output_tokens,
           args.max_concurrency, args.timeout) <= 0:
        raise ValueError("Token counts, concurrency and timeout must be positive")
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
