"""Produce per-rank and worst-rank latency ratios without GPU dependencies."""

import argparse
import json
import statistics
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        raise ValueError("Empty latency samples")
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def summarize(records):
    groups = {}
    for record in records:
        # Do not combine modes, payloads, destinations or ranks.
        key = tuple(record[k] for k in ("mode", "kv_target", "tp_mib", "moe_mib", "kv_mib", "rank", "flow", "case"))
        groups.setdefault(key, []).extend(record["samples_ms"])
    output = []
    for key, samples in sorted(groups.items()):
        baseline = groups.get((*key[:-1], key[-2]))
        if baseline is None:
            raise ValueError(f"Missing standalone baseline for {key}")
        output.append(dict(zip(("mode", "kv_target", "tp_mib", "moe_mib", "kv_mib", "rank", "flow", "case"), key),
                           samples=len(samples), p50_ms=percentile(samples, 0.50),
                           p95_ms=percentile(samples, 0.95), p99_ms=percentile(samples, 0.99),
                           p50_slowdown=percentile(samples, .50) / percentile(baseline, .50),
                           p95_slowdown=percentile(samples, .95) / percentile(baseline, .95),
                           note="transport replay; device timeline and link counters still required"))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for folder in args.inputs:
        for filename in Path(folder).glob("rank*.jsonl"):
            records.extend(json.loads(line) for line in filename.read_text().splitlines() if line.strip())
    if not records:
        raise SystemExit("No replay records found")
    result = summarize(records)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for row in result:
        if "+" in row["case"]:
            print(f"rank={row['rank']} {row['case']:24} {row['flow']:10} p95 ratio={row['p95_slowdown']:.3f}")


if __name__ == "__main__":
    main()
