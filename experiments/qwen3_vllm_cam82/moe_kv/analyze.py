"""Strict, CPU-only paired analysis. Never infer physical congestion from timings."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from spec import CONDITIONS, digest, overlap


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "mean": statistics.mean(values), "p50": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1],
            "p99": values[math.ceil(.99 * len(values)) - 1]}


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def read_run(root):
    headers, rows = [], {}
    for rank in range(12):
        file = Path(root) / f"rank{rank:02d}.jsonl"
        data = [json.loads(line) for line in file.read_text().splitlines() if line]
        if (not data or data[0].get("type") != "header" or data[0].get("rank") != rank
                or data[-1] != {"type": "complete", "ok": True}
                or any(x.get("type") == "error" for x in data)):
            raise ValueError(f"Incomplete/failed run: {file}")
        header = data[0]
        if header["schema"] != 1 or header["host"] != ("228" if rank < 8 else "238"):
            raise ValueError("Unexpected log schema or rank mapping")
        if header["signature"] != digest({"config": header["config"], "trace": header["trace"]}):
            raise ValueError("Manifest signature does not match its contents")
        if any(x.get("wall_ms", 0) < 0 or x.get("stream_ms", 0) < 0 for x in data):
            raise ValueError("Invalid duration")
        pre = [x for x in data if x["type"] == "preflight"]
        if len(pre) != 1 or pre[0]["moe_payload_ok"] != (rank >= 4) or pre[0]["kv_payload_ok"] != (rank >= 8):
            raise ValueError("Missing data-correctness preflight")
        headers.append(header)
        rows[rank] = data[1:-1]
    if len({x["signature"] for x in headers}) != 1:
        raise ValueError("Mixed runs/configurations/captures")
    config, trace = headers[0]["config"], headers[0]["trace"]
    expected_keys = {(i, p, op) for i in range(config["iterations"])
                     for p in range(trace["pairs"]) for op in range(3)}
    for rank, data in rows.items():
        expected_phases = {(r, c) for r in range(config["repeats"]) for c in CONDITIONS}
        phases = [x for x in data if x["type"] == "phase"]
        if len(phases) != len(expected_phases) or {(x["repeat"], x["condition"]) for x in phases} != expected_phases:
            raise ValueError(f"Missing/duplicate phases rank{rank}")
        for r, condition in expected_phases:
            selected = [x for x in data if x.get("repeat") == r and x.get("condition") == condition]
            phase = next(x for x in selected if x["type"] == "phase")
            if not phase["payload_ok"]:
                raise ValueError("Phase payload check failed")
            ms = [x for x in selected if x["type"] == "moe"]
            wanted = expected_keys if rank >= 4 and condition != "K" else set()
            actual = {(x["iteration"], x["pair"], x["op"]) for x in ms}
            if len(ms) != len(wanted) or actual != wanted:
                raise ValueError(f"Incomplete/duplicate MoE samples rank{rank} {r}/{condition}")
            ks = [x for x in selected if x["type"] == "kv"]
            want_k = config["kv_transfers"] if rank >= 8 and condition != "M" else 0
            if len(ks) != want_k or {x["transfer"] for x in ks} != set(range(want_k)):
                raise ValueError(f"Incomplete/duplicate KV samples rank{rank} {r}/{condition}")
            for x in ms + ks:
                if not phase["start_ns"] <= x["start_ns"] < x["end_ns"] <= phase["work_end_ns"]:
                    raise ValueError("Sample outside its local phase interval")
    return headers[0], rows


def analyze(root):
    header, by_rank = read_run(root)
    buckets = defaultdict(list)
    for rank, rows in by_rank.items():
        for row in rows:
            if row["type"] in ("moe", "kv", "phase"):
                buckets[(row["repeat"], row["condition"])].append({**row, "rank": rank})
    phases, paired, warnings = [], [], []
    for repeat in range(header["config"]["repeats"]):
        critical = {}
        overlaps = set()
        for condition in CONDITIONS:
            data = buckets[repeat, condition]
            kv = [x for x in data if x["type"] == "kv"]
            intervals = merged([(x["start_ns"], x["end_ns"]) for x in kv])
            # All KV records originate on 238: monotonic timestamps are comparable
            # across those local processes, never against host228's clock.
            ops = defaultdict(list)
            for x in data:
                if x["type"] != "moe":
                    continue
                key = (x["iteration"], x["pair"], x["op"])
                ops[key].append(x)
                if condition == "MK" and x["rank"] >= 8 and overlap(x["start_ns"], x["end_ns"], intervals):
                    overlaps.add(key)
                if condition == "SERIAL" and x["rank"] >= 8 and overlap(x["start_ns"], x["end_ns"], intervals):
                    raise ValueError("SERIAL contains local host submission overlap")
            critical[condition] = {key: max(x["wall_ms"] for x in values) for key, values in ops.items()}
            phase_row = {"repeat": repeat, "condition": condition,
                "moe_critical_wall_ms": stats(list(critical[condition].values())),
                "kv_rank_batch_ms": stats([x["wall_ms"] for x in kv]),
                "phase_barrier_upper_ms": max(x["phase_ms"] for x in data if x["type"] == "phase")}
            for index, name in enumerate(("metadata", "dispatch", "combine")):
                phase_row[name + "_critical_wall_ms"] = stats([v for key, v in critical[condition].items() if key[2] == index])
                phase_row[name + "_critical_stream_ms"] = stats([
                    max(x["stream_ms"] for x in values) for key, values in ops.items() if key[2] == index])
            if kv:
                seconds = (max(x["end_ns"] for x in kv) - min(x["start_ns"] for x in kv)) / 1e9
                phase_row["kv_aggregate_GiB_s_including_gaps"] = sum(x["bytes"] for x in kv) / seconds / 2**30
            phases.append(phase_row)
        row = {"repeat": repeat, "host_submission_overlap_fraction": len(overlaps) / len(critical["MK"])}
        for op, name in enumerate(("metadata", "dispatch", "combine")):
            keys = [key for key in critical["M"] if key[2] == op]
            baseline = statistics.median(critical["M"][k] for k in keys)
            concurrent = statistics.median(critical["MK"][k] for k in keys)
            serial = statistics.median(critical["SERIAL"][k] for k in keys)
            if baseline <= 0:
                raise ValueError("Zero baseline duration")
            row[name + "_slowdown_pct"] = (concurrent / baseline - 1) * 100
            row[name + "_serial_change_pct"] = (serial / baseline - 1) * 100
            matched = [k for k in keys if k in overlaps]
            row[name + "_overlap_matched_slowdown_pct"] = (
                (statistics.median(critical["MK"][k] for k in matched) /
                 statistics.median(critical["M"][k] for k in matched) - 1) * 100 if matched else None)
        phase_lookup = {x["condition"]: x for x in phases if x["repeat"] == repeat}
        row["kv_slowdown_pct"] = (phase_lookup["MK"]["kv_rank_batch_ms"]["p50"] /
                                  phase_lookup["K"]["kv_rank_batch_ms"]["p50"] - 1) * 100
        paired.append(row)
        if row["host_submission_overlap_fraction"] < .5:
            warnings.append(f"repeat {repeat}: less than 50% host submission overlap; increase transfer count or inspect trace")
    if header["config"]["profile"]:
        warnings.append("Profile enabled: diagnostic run; do not mix with unprofiled performance runs")
    if len(paired) < 5:
        warnings.append("Fewer than five independent rounds: smoke test only")
    warnings.append("Host submission overlap does not prove device overlap or shared-link congestion; inspect device trace/topology")
    summary = {key: stats([r[key] for r in paired if r.get(key) is not None])
               for key in paired[0] if key != "repeat"}
    return {"manifest": header, "rounds": paired, "phases": phases, "summary": summary, "warnings": warnings}


def markdown(result):
    lines = ["# MoE / KV 通信重放结果", "", "只报告实测变化，不自动判定物理链路拥塞。百分比按每轮配对计算。", "",
             "| 指标 | 各轮变化率中位数 | 最小～最大 |", "|---|---:|---:|"]
    for key, name in (("dispatch_slowdown_pct", "Dispatch 完成耗时"),
                      ("combine_slowdown_pct", "Combine 完成耗时"),
                      ("kv_slowdown_pct", "每 rank KV 批次耗时"),
                      ("dispatch_serial_change_pct", "错峰 Dispatch 相对 M"),
                      ("combine_serial_change_pct", "错峰 Combine 相对 M")):
        values = [x[key] for x in result["rounds"]]
        lines.append(f"| {name} | {statistics.median(values):+.2f}% | {min(values):+.2f}% ～ {max(values):+.2f}% |")
    lines += ["", "正数表示变慢，负数表示变快；需结合各轮波动判断，不能预设必须发生劣化。", "",
              "stream_ms 包含该 stream 上的依赖等待，不等于纯链路传输耗时。critical 为同一调用八个 rank 耗时的最大值。", "",
              "## 重叠样本（主机提交时间口径）", "",
              "仅比较重叠样本与 M 组同一 iteration/pair/op；不是设备重叠或物理拥塞证明。", "",
              "| 轮次 | 主机提交重叠率 | 配对 Dispatch 变化 | 配对 Combine 变化 |",
              "|---|---:|---:|---:|"]
    def percent(value):
        return "无样本" if value is None else f"{value:+.2f}%"
    for row in result["rounds"]:
        lines.append(f"| {row['repeat']} | {row['host_submission_overlap_fraction'] * 100:.2f}% | "
                     f"{percent(row['dispatch_overlap_matched_slowdown_pct'])} | "
                     f"{percent(row['combine_overlap_matched_slowdown_pct'])} |")
    trace = result["manifest"]["trace"]
    lines += ["", f"每对 Dispatch+Combine 的跨机载荷均值："
              f"{trace['cross_host_payload_bytes_per_window'] / trace['pairs'] / 1024:.2f} KiB"
              "（全 EP8、双向合计，不含计数 AllGather）。", "", "## 检查事项", ""]
    lines.extend("- " + w for w in result["warnings"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    result = analyze(args.run_dir)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=False)
    (target / "analysis.json").write_text(json.dumps(result, indent=2))
    with (target / "paired_rounds.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(result["rounds"][0]))
        writer.writeheader()
        writer.writerows(result["rounds"])
    (target / "report.md").write_text(markdown(result))
    print(markdown(result))
    print(f"ANALYSIS_READY {target}")


if __name__ == "__main__":
    main()
