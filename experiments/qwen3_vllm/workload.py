"""OpenAI /v1/completions workload; client metrics are NOT link-conflict proof."""

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path


def sse_json(line):
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    data = line[5:].strip()
    return None if data == b"[DONE]" else json.loads(data)


def summarize_rows(rows):
    result = {"requests": len(rows), "success": sum(bool(x.get("success")) for x in rows)}
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = sorted(x[metric] for x in rows if x.get("success") and x.get(metric) is not None)
        if values:
            result[metric] = {"p50": values[math.ceil(len(values) * .5) - 1],
                              "p99": values[math.ceil(len(values) * .99) - 1], "mean": statistics.mean(values)}
    if len(rows) < 200:
        result["warning"] = "Fewer than 200 requests; P99 is not a stable tail estimate"
    return result


def summary(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = summarize_rows(rows)
    result["by_kind"] = {kind: summarize_rows([x for x in rows if x.get("kind", "unknown") == kind])
                         for kind in sorted({x.get("kind", "unknown") for x in rows})}
    return result


async def run(args):
    import aiohttp
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    material = tokenizer.encode("An agent resumes its reasoning history and tool observations. ", add_special_tokens=False)
    prefix = (material * (args.prefix_tokens // len(material) + 1))[:args.prefix_tokens]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against another run using the same filename.
    with output.open("x"):
        pass
    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async def one(i):
            await asyncio.sleep(max(0, start + i / args.rps - time.perf_counter()))
            async with sem:
                sent = time.perf_counter()
                row = {"id": i, "kind": "long-prefix" if i % 4 == 0 else "background",
                       "client_queue_ms": max(0, (sent - start - i / args.rps) * 1000)}
                prompt = (prefix if i % 4 == 0 else material * 4) + tokenizer.encode(f" Task {i}: continue.", add_special_tokens=False)
                # The upstream proxy's recompute path concatenates strings.
                # Re-tokenization can change counts; reported usage is authoritative.
                payload = {"model": "qwen3", "prompt": tokenizer.decode(prompt, skip_special_tokens=True), "max_tokens": args.output_tokens,
                           "temperature": 0, "ignore_eos": True, "stream": True,
                           "stream_options": {"include_usage": True}}
                first = last = None
                usage = {}
                done = finished = False
                try:
                    async with session.post(args.base_url.rstrip("/") + "/v1/completions", json=payload,
                                            headers={"X-Request-Id": f"qwen-test-{i}-{time.time_ns()}"}) as response:
                        response.raise_for_status()
                        # readline preserves SSE lines across arbitrary network chunks.
                        while raw := await response.content.readline():
                            if raw.strip() == b"data: [DONE]":
                                done = True
                            event = sse_json(raw)
                            if event is None:
                                continue
                            if event.get("error"):
                                raise RuntimeError(str(event["error"]))
                            if event.get("usage"):
                                usage = event["usage"]
                            if any(x.get("finish_reason") is not None for x in event.get("choices", [])):
                                finished = True
                            if any(x.get("text") for x in event.get("choices", [])):
                                last = time.perf_counter()
                                if first is None:
                                    first = last
                    if first is None:
                        raise RuntimeError("No generated text received")
                    if not done or not finished:
                        raise RuntimeError("Incomplete SSE response; do not count it as successful inference")
                    count = usage.get("completion_tokens")
                    row.update(success=True, ttft_ms=(first - sent) * 1000,
                        e2e_ms=(time.perf_counter() - sent) * 1000, output_tokens=count,
                        tpot_ms=(last - first) * 1000 / (count - 1) if count and count > 1 else None,
                        reported_usage=usage)
                except Exception as exc:
                    row.update(success=False, error=str(exc), e2e_ms=(time.perf_counter() - sent) * 1000)
                with output.open("a") as out:
                    out.write(json.dumps(row) + "\n")
        await asyncio.gather(*(one(i) for i in range(args.requests)))
    print(json.dumps(summary(output), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summarize")
    p.add_argument("--base-url", default="http://10.155.205.228:30080")
    p.add_argument("--model-path", default="/ssd_2/yuanwenliu/models/Qwen3-30B-A3B")
    p.add_argument("--output")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--rps", type=float, default=.2)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--prefix-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=32)
    args = p.parse_args()
    if args.summarize:
        print(json.dumps(summary(args.summarize), indent=2))
        return
    if not args.output or min(args.requests, args.rps, args.concurrency, args.prefix_tokens, args.output_tokens) <= 0:
        p.error("Specify a new output path and positive load parameters")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
