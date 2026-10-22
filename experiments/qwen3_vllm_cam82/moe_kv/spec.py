"""CPU-only validation of native communication captures and experiment plans."""
import hashlib
import json
import re
from pathlib import Path

EP = 8
ALIGN = 2 * 1024 * 1024
CONDITIONS = ("M", "K", "MK", "SERIAL")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate(traces):
    if len(traces) != EP or [x["ep_rank"] for x in traces] != list(range(EP)):
        raise ValueError("Need exactly ep0.json through ep7.json")
    count = len(traces[0]["pairs"])
    if not 1 <= count <= 192:
        raise ValueError("Capture must contain 1..192 complete pairs")
    for r, trace in enumerate(traces):
        if trace.get("schema") != 1 or trace.get("origin") != "native_decode_capture" or trace.get("ep_size") != EP:
            raise ValueError("Require native EP8 decode captures, not synthetic splits")
        if (trace["host"], trace["physical_device"]) != (("228", r + 4) if r < 4 else ("238", r - 4)):
            raise ValueError("Unexpected EP rank to physical-device mapping")
        for key in ("source_hashes", "global_ranks", "skip_pairs"):
            if trace[key] != traces[0][key]:
                raise ValueError(f"Capture ranks disagree on {key}")
        if len(trace["global_ranks"]) != EP or len(set(trace["global_ranks"])) != EP:
            raise ValueError("Invalid native EP membership")
        hashes = trace["source_hashes"]
        if set(hashes) != {"vllm_ascend.ops.moe.token_dispatcher", "vllm_ascend.ops.moe.comm_utils"} or any(
                not re.fullmatch("[0-9a-f]{64}", v) for v in hashes.values()):
            raise ValueError("Missing native source fingerprints")
        if len(trace["pairs"]) != count:
            raise ValueError("Capture incomplete across ranks")
        for pair in trace["pairs"]:
            if [(x["kind"], x["phase"]) for x in pair] != [
                ("all_gather", "dispatch"), ("all_to_all", "dispatch"), ("all_to_all", "combine")]:
                raise ValueError("Expected native count AllGather, dispatch A2AV, combine A2AV")
            if pair[0]["shape"] != [128] or pair[0]["dtype"] not in ("float32", "int32", "int64"):
                raise ValueError("Unexpected Qwen expert-count gather")
            for op in pair[1:]:
                if op["dtype"] != "bfloat16" or len(op["shape"]) != 2 or op["shape"][1] != 2048:
                    raise ValueError("Require unquantized Qwen hidden_size=2048")
                for key in ("input_splits", "output_splits"):
                    splits = op[key]
                    if not isinstance(splits, list) or len(splits) != EP or any(type(x) is not int or x < 0 for x in splits):
                        raise ValueError("Invalid all-to-all split vector")
                if sum(op["input_splits"]) != op["shape"][0]:
                    raise ValueError("Split sum does not equal input rows")
            if (pair[1]["input_splits"] != pair[2]["output_splits"] or
                    pair[1]["output_splits"] != pair[2]["input_splits"]):
                raise ValueError("Dispatch/combine are not inverse split layouts")
    cross_bytes = 0
    for i in range(count):
        if len({trace["pairs"][i][0]["dtype"] for trace in traces}) != 1:
            raise ValueError("Metadata collective dtype mismatch across ranks")
        for j in (1, 2):
            for src in range(EP):
                for dst in range(EP):
                    send = traces[src]["pairs"][i][j]["input_splits"][dst]
                    recv = traces[dst]["pairs"][i][j]["output_splits"][src]
                    if send != recv:
                        raise ValueError(f"Mismatched collective sequence/splits: pair={i} op={j} {src}->{dst}")
                    if src // 4 != dst // 4:
                        cross_bytes += send * 2048 * 2
    if not cross_bytes:
        raise ValueError("No cross-host MoE payload in capture")
    return {"trace_sha256": digest(traces), "pairs": count,
            "cross_host_payload_bytes_per_window": cross_bytes}


def load(root):
    traces = [json.loads((Path(root) / f"ep{r}.json").read_text()) for r in range(EP)]
    return traces, validate(traces)


def kv_layout(prefix_tokens):
    if prefix_tokens <= 0 or prefix_tokens > 2048:
        raise ValueError("Require 1..2048 prefix tokens")
    rounded = ((prefix_tokens + 63) // 64) * 64
    # 48 layers * K/V, one KV head per TP rank, dim128 BF16.
    per_region = rounded * 128 * 2
    return {"regions": 96, "region_bytes": per_region, "stride_bytes": ALIGN,
            "bytes_per_rank": 96 * per_region, "bytes_per_request": 4 * 96 * per_region}


def overlap(start, end, intervals):
    return any(start < stop and end > begin for begin, stop in intervals)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(load(args.trace_dir)[1], indent=2))
