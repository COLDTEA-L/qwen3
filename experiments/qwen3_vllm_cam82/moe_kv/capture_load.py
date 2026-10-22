"""Send one bounded native decode workload to EACH D API for metadata capture."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request


def request(url, tokens):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {"model": "qwen3", "prompt": "Write a detailed explanation of prefill and decode separation. ",
               "max_tokens": tokens, "ignore_eos": True, "temperature": 0, "seed": 1024,
               "stream": False}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=600) as response:
        data = json.load(response)
    if data.get("error") or not data.get("choices"):
        raise RuntimeError(f"Capture workload failed: {data}")
    print(json.dumps({"url": url, "id": data.get("id"), "usage": data.get("usage")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=128)
    a = p.parse_args()
    if not 16 <= a.tokens <= 512:
        p.error("Use 16..512 tokens")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, f"http://10.155.205.{host}:30011", a.tokens) for host in (228, 238)]
        for future in futures:
            future.result()
