"""Small deterministic output regression, not a model-quality benchmark."""

import argparse
import json
import urllib.request
from pathlib import Path

PROMPTS = (
    "Explain what a KV cache is in one sentence.",
    "Compute 17 + 25. Give only the answer.",
    "List three benefits of prefill and decode separation.",
)


def collect(url, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output exists; choose a new run filename")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    for prompt in PROMPTS:
        payload = {"model": "qwen3", "prompt": prompt, "temperature": 0,
                   "seed": 1024, "max_tokens": 32, "stream": False}
        request = urllib.request.Request(url.rstrip("/") + "/v1/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=300) as response:
            body = json.load(response)
        choice = body["choices"][0]
        if not choice.get("text") or choice.get("finish_reason") not in ("stop", "length"):
            raise RuntimeError("Incomplete/empty generation")
        rows.append({"prompt": prompt, "text": choice["text"],
                     "finish_reason": choice["finish_reason"], "usage": body.get("usage")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(rows, stream, indent=2)
    print("OUTPUT_SNAPSHOT_READY", output)


def compare(native, alltoall):
    with Path(native).open() as stream:
        before = json.load(stream)
    with Path(alltoall).open() as stream:
        after = json.load(stream)
    if len(before) != len(PROMPTS) or len(after) != len(PROMPTS):
        raise ValueError("Incomplete snapshot")
    matched = True
    for prompt, left, right in zip(PROMPTS, before, after):
        if left["prompt"] != prompt or right["prompt"] != prompt:
            raise ValueError("Prompt/order mismatch")
        equal = (left["text"], left["finish_reason"]) == (right["text"], right["finish_reason"])
        matched = matched and equal
        print("MATCH" if equal else "REVIEW_REQUIRED", prompt)
    print("OUTPUT_SMOKE_MATCH" if matched else "OUTPUT_SMOKE_DIFFERENCE: inspect logits/output before performance tests")
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "compare"))
    parser.add_argument("--url", default="http://10.155.205.228:30009")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--alltoall", type=Path)
    args = parser.parse_args()
    if args.action == "collect":
        if args.output is None:
            parser.error("collect requires --output")
        collect(args.url, args.output)
    else:
        if args.native is None or args.alltoall is None:
            parser.error("compare requires --native and --alltoall")
        if not compare(args.native, args.alltoall):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
