"""Controlled physical-traffic replay on 2x8 A2 NPUs, WITHOUT model weights.

MoE payload is emulated by two HCCL EP8 All-to-Alls; KV by HCCL P2P.
These are explicitly transport surrogates, NOT DeepEP/MemFabric performance.
Rank 0..3=P, 4..11=D, 12..15=Pool. Separate streams and threads permit overlap.
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["cam-detour", "hccl-a2a"], default="cam-detour")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tp-mib", type=int, default=8)
    parser.add_argument("--moe-mib", type=int, default=8)
    parser.add_argument("--kv-mib", type=int, default=64)
    parser.add_argument("--kv-target", choices=["228", "238"], default="228")
    args = parser.parse_args()
    if min(args.iterations, args.tp_mib, args.moe_mib, args.kv_mib) <= 0 or args.warmup < 0:
        raise ValueError("Invalid benchmark sizes/iterations")
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from sglang.srt.distributed.device_communicators.cam_detour import CamConfig, CamRuntime

    rank, device = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    dist.init_process_group("gloo")
    if dist.get_world_size() != 16:
        raise ValueError("Use --nnodes=2 --nproc_per_node=8")
    selected_tp = None
    for members in (list(range(4)), list(range(4, 8)), list(range(8, 12))):
        pg = dist.new_group(members, backend="hccl")
        if rank in members:
            selected_tp = SimpleNamespace(ranks=members, rank_in_group=members.index(rank), device_group=pg)
    ep = dist.new_group(list(range(4, 12)), backend="hccl")
    kv_pairs = []
    destination = 4 if args.kv_target == "228" else 8
    own_pair = None
    for i in range(4):
        members = [destination + i, 12 + i]
        pg = dist.new_group(members, backend="hccl")
        if rank in members:
            own_pair = (pg, members)
        kv_pairs.append(pg)
    host_start = rank // 8 * 8
    cfg = CamConfig.from_dict({
        "process_rank_to_cam": {str(r): r - host_start for r in range(host_start, host_start + 8)},
        "groups": [list(range(host_start, host_start + 4)), list(range(host_start + 4, host_start + 8))],
    })
    runtime = CamRuntime(cfg, args.mode)
    tp_buffer = torch.full((args.tp_mib * 1024 * 1024 // 2,), (device + 1) * 0.001,
                           dtype=torch.bfloat16, device="npu") if selected_tp else None
    moe_send = torch.full((args.moe_mib * 1024 * 1024 // 2,), 0.125,
                          dtype=torch.bfloat16, device="npu") if 4 <= rank < 12 else None
    moe_recv = torch.empty_like(moe_send) if moe_send is not None else None
    kv_buffer = torch.ones(args.kv_mib * 1024 * 1024 // 2, dtype=torch.bfloat16,
                           device="npu") if own_pair else None
    # Materialize each native HCCL group before concurrent stream replay.
    if selected_tp:
        dist.all_reduce(torch.zeros(1, device="npu"), group=selected_tp.device_group)
    if moe_send is not None:
        dist.all_reduce(torch.zeros(1, device="npu"), group=ep)
    if own_pair:
        dist.all_reduce(torch.zeros(1, device="npu"), group=own_pair[0])
    torch.npu.synchronize()
    dist.barrier()

    def run(kind, start_gate):
        torch.npu.set_device(device)
        stream = torch.npu.Stream()
        durations = []
        start_gate.wait()
        with torch.npu.stream(stream):
            for iteration in range(args.warmup + args.iterations):
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                begin.record(stream)
                with torch.profiler.record_function("REPLAY::" + kind):
                    if kind == "multipath":
                        # Avoid repeatedly summing the previous iteration's output.
                        runtime.all_reduce(tp_buffer.clone(), selected_tp)
                    elif kind == "moe":
                        dist.all_to_all_single(moe_recv, moe_send, group=ep)
                        dist.all_to_all_single(moe_send, moe_recv, group=ep)
                    elif kind == "kv":
                        pg, members = own_pair
                        if rank == members[1]:
                            dist.send(kv_buffer, dst=members[0], group=pg)
                        else:
                            dist.recv(kv_buffer, src=members[1], group=pg)
                end.record(stream)
                end.synchronize()
                if iteration >= args.warmup:
                    durations.append(begin.elapsed_time(end))
        return kind, durations

    cases = ["multipath", "moe", "kv", "multipath+moe", "multipath+kv",
             "moe+kv", "multipath+moe+kv"]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        enabled = case.split("+")
        tasks = [k for k in enabled if (k == "multipath" and selected_tp is not None)
                 or (k == "moe" and moe_send is not None) or (k == "kv" and own_pair is not None)]
        dist.barrier()
        gate = threading.Event()
        with ThreadPoolExecutor(max_workers=3) as executor:
            jobs = [executor.submit(run, kind, gate) for kind in tasks]
            gate.set()
            records = [job.result() for job in jobs]
        torch.npu.synchronize()
        with (output_dir / f"rank{rank}.jsonl").open("a") as out:
            for kind, durations in records:
                out.write(json.dumps({"case": case, "flow": kind, "rank": rank,
                                      "mode": args.mode, "kv_target": args.kv_target,
                                      "samples_ms": durations, "transport_surrogate": True,
                                      "tp_mib": args.tp_mib, "moe_mib": args.moe_mib,
                                      "kv_mib": args.kv_mib}) + "\n")
        dist.barrier()
    if kv_buffer is not None:
        torch.testing.assert_close(kv_buffer, torch.ones_like(kv_buffer))
    print(f"REPLAY_PASS rank={rank} mode={args.mode}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
