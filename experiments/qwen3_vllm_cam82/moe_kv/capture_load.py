"""Send bounded concurrent native workloads to EACH D API for metadata capture.

Client concurrency is not a guarantee of the engine's actual decode batch size.
Validate and inspect the captured collective shapes before replaying them.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import urllib.request


def request(url, tokens, index=0, gate=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    prompt = "Write a detailed explanation of prefill and decode separation. "
    if index:
        prompt = f"Task {index}: " + prompt
    payload = {"model": "qwen3", "prompt": prompt,
               "max_tokens": tokens, "ignore_eos": True, "temperature": 0, "seed": 1024,
               "stream": False}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    if gate is not None:
        gate.wait(timeout=60)
    with opener.open(req, timeout=600) as response:
        data = json.load(response)
    if data.get("error") or not data.get("choices"):
        raise RuntimeError(f"Capture workload failed: {data}")
    record = {"url": url, "index": index, "id": data.get("id"), "usage": data.get("usage")}
    print(json.dumps(record), flush=True)
    return record


def collect(tokens=128, requests_per_dp=1):
    if not 16 <= tokens <= 512 or not 1 <= requests_per_dp <= 32:
        raise ValueError("Use 16..512 tokens and 1..32 requests per DP")
    total = 2 * requests_per_dp
    gate = threading.Barrier(total)
    with ThreadPoolExecutor(max_workers=total) as pool:
        futures = [pool.submit(request, f"http://10.155.205.{host}:30011", tokens, i, gate)
                   for host in (228, 238) for i in range(requests_per_dp)]
        return [future.result() for future in futures]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--requests-per-dp", type=int, default=1,
                   help="Concurrent requests to each D API; not measured engine batch size")
    a = p.parse_args()
    if not 16 <= a.tokens <= 512 or not 1 <= a.requests_per_dp <= 32:
        p.error("Use 16..512 tokens and 1..32 requests per DP")
    results = collect(a.tokens, a.requests_per_dp)
    print("CAPTURE_LOAD_COMPLETE", json.dumps({"requests": len(results),
          "requests_per_dp": a.requests_per_dp, "output_tokens_requested": a.tokens}), flush=True)


if __name__ == "__main__":
    main()
