"""Standalone native EP8 communication replay + Ascend Mooncake KV reads.

12 processes: 228 physical0..3 KV sources, physical4..7 EP0..3;
238 physical0..3 EP4..7 and KV receivers. No models, CAM, TP conversion.
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import threading
import time

from spec import ALIGN, CONDITIONS, digest, kv_layout, load


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", choices=("228", "238"), required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--master-addr", default="10.155.205.228")
    p.add_argument("--master-port", type=int, default=29480)
    p.add_argument("--prefix-tokens", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=20, help="Captured windows per condition")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--kv-transfers", type=int, default=200, help="Read batches per destination rank")
    p.add_argument("--kv-gap-ms", type=float, default=0, help="Delay after each completed read; not a rate guarantee")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--profile", action="store_true", help="Separate short profiling run, not comparable to unprofiled timing")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if min(a.iterations, a.warmup, a.kv_transfers, a.repeats, a.timeout) <= 0 or a.kv_gap_ms < 0:
        p.error("Counts must be positive and gap nonnegative")
    if not 1024 <= a.master_port <= 65535 or not a.run_id or "/" in a.run_id:
        p.error("Invalid port or run-id")
    kv_layout(a.prefix_tokens)
    return a


class KV:
    def __init__(self, torch, device, source, ip, layout):
        from mooncake.engine import TransferEngine
        self.torch, self.device, self.source, self.layout = torch, device, source, layout
        self.engine = TransferEngine()
        # Match the pinned native connector's HCCL Ascend transport hostname.
        rc = self.engine.initialize(f"{ip}:0:npu_{device}", "P2PHANDSHAKE", "ascend", "")
        if rc != 0:
            raise RuntimeError(f"Mooncake initialize failed: {rc}")
        size = layout["regions"] * ALIGN
        self.owner = torch.empty(size + ALIGN, dtype=torch.uint8, device=f"npu:{device}")
        offset = (-self.owner.data_ptr()) % ALIGN
        self.buffer = self.owner[offset:offset + size]
        self.buffer.fill_(device + 1 if source else 0)
        torch.npu.synchronize()
        self.ptrs = [self.buffer.data_ptr() + i * ALIGN for i in range(layout["regions"])]
        self.lengths = [layout["region_bytes"]] * layout["regions"]
        self.registered = []
        for ptr in self.ptrs:
            if ptr % ALIGN or ptr + ALIGN > self.buffer.data_ptr() + size:
                raise RuntimeError("KV registration bounds/alignment failure")
            rc = self.engine.register_memory(ptr, ALIGN)
            if rc != 0:
                raise RuntimeError(f"Mooncake register failed: {rc}")
            self.registered.append(ptr)
        self.metadata = {"session": f"{ip}:{self.engine.get_rpc_port()}", "ptrs": self.ptrs}

    def reset(self):
        if not self.source:
            self.buffer.zero_()
            self.torch.npu.synchronize()

    def transfer(self, peer):
        rc = self.engine.batch_transfer_sync_read(peer["session"], self.ptrs, peer["ptrs"], self.lengths)
        if rc != 0:
            raise RuntimeError(f"Mooncake transfer failed: {rc}")

    def verify(self):
        # Full transferred payload verification, outside all timed sections.
        for i in range(self.layout["regions"]):
            data = self.buffer[i * ALIGN:i * ALIGN + self.layout["region_bytes"]]
            if not bool((data == self.device + 1).all().item()):
                raise RuntimeError(f"KV payload mismatch at region {i}")

    def close(self):
        for ptr in self.registered:
            rc = self.engine.unregister_memory(ptr)
            if rc != 0:
                raise RuntimeError(f"Mooncake unregister failed: {rc}")


class Replay:
    def __init__(self, torch, group, ep_rank, trace):
        from vllm_ascend.ops.moe.comm_utils import async_all_to_all, gather_from_sequence_parallel_region
        self.torch, self.group, self.rank = torch, group, ep_rank
        self.a2a, self.gather = async_all_to_all, gather_from_sequence_parallel_region
        self.ops = []
        for pair_index, pair in enumerate(trace["pairs"]):
            for op_index, spec in enumerate(pair):
                tensor = torch.full(spec["shape"], ep_rank + 1,
                                    dtype=getattr(torch, spec["dtype"]), device="npu")
                self.ops.append((pair_index, op_index, spec, tensor))
        torch.npu.synchronize()

    def invoke(self, spec, tensor):
        if spec["kind"] == "all_gather":
            return self.gather(tensor, self.group)
        _, output, handle = self.a2a(tensor, spec["output_splits"], spec["input_splits"], self.group)
        handle.wait()
        return output

    def verify(self):
        for _, _, spec, tensor in self.ops:
            output = self.invoke(spec, tensor)
            self.torch.npu.current_stream().synchronize()
            splits = ([128] * 8 if spec["kind"] == "all_gather" else spec["output_splits"])
            start = 0
            for src, rows in enumerate(splits):
                if rows and not bool((output[start:start + rows] == src + 1).all().item()):
                    raise RuntimeError(f"MoE replay data mismatch from EP{src}")
                start += rows

    def run(self, iterations, measured=True):
        rows = []
        for iteration in range(iterations):
            for pair, op, spec, tensor in self.ops:
                # Stream event completion does not globally synchronize the KV stream.
                begin = self.torch.npu.Event(enable_timing=True)
                end = self.torch.npu.Event(enable_timing=True)
                t0 = time.monotonic_ns()
                with (self.torch.profiler.record_function(f"moe_replay/{iteration}/{pair}/{op}")
                      if getattr(self, "profile", False) else contextlib.nullcontext()):
                    begin.record()
                    output = self.invoke(spec, tensor)
                    end.record()
                end.synchronize()
                t1 = time.monotonic_ns()
                if measured:
                    rows.append({"type": "moe", "iteration": iteration, "pair": pair, "op": op,
                        "phase": "metadata" if op == 0 else spec["phase"],
                        "start_ns": t0, "end_ns": t1, "wall_ms": (t1 - t0) / 1e6,
                        "stream_ms": begin.elapsed_time(end)})
                del output
        return rows


def worker(local, args):
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from qwen_cam82.policy import check_installation
    root = check_installation()
    traces, info = load(args.trace_dir)
    rank = local if args.host == "228" else local + 8
    ep_rank = rank - 4
    device = local
    torch.npu.set_device(device)
    if torch.npu.device_count() != 8:
        raise RuntimeError("Require all eight physical NPUs visible on each host")
    for key, expected in traces[0]["source_hashes"].items():
        file = root / (key.removeprefix("vllm_ascend.").replace(".", "/") + ".py")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Captured/replay native source mismatch: {file}")
    dist.init_process_group("gloo", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                            rank=rank, world_size=12, timeout=datetime.timedelta(seconds=args.timeout))
    group = dist.new_group(list(range(4, 12)), backend="hccl",
                           timeout=datetime.timedelta(seconds=args.timeout))
    config = {k: v for k, v in vars(args).items() if k not in ("host", "trace_dir", "output_dir", "dry_run")}
    signature = digest({"config": config, "trace": info})
    signatures = [None] * 12
    dist.all_gather_object(signatures, signature)
    if len(set(signatures)) != 1:
        raise RuntimeError("Hosts have different traces or experiment arguments")
    path = Path(args.output_dir) / args.run_id / f"rank{rank:02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.open("x")
    def write(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
    write({"type": "header", "schema": 1, "rank": rank, "host": args.host,
           "device": device, "config": config, "trace": info, "signature": signature,
           "measurement": "native communication replay; no expert GEMM/TP/permutation; host overlap is not device/link proof"})
    kv = KV(torch, device, rank < 4, f"10.155.205.{args.host}", kv_layout(args.prefix_tokens)) if rank < 4 or rank >= 8 else None
    peers = [None] * 12
    dist.all_gather_object(peers, kv.metadata if kv else None)
    peer = peers[local] if rank >= 8 else None
    replay = Replay(torch, group, ep_rank, traces[ep_rank]) if rank >= 4 else None
    if replay:
        replay.verify()
        replay.run(args.warmup, measured=False)
    dist.barrier()
    if rank >= 8:
        for _ in range(3):
            kv.transfer(peer)
        kv.verify()
    dist.barrier()
    write({"type": "preflight", "moe_payload_ok": bool(replay), "kv_payload_ok": rank >= 8})
    profiler = None
    if args.profile and replay:
        replay.profile = True
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path.parent / f"profile-rank{rank:02d}")))
        profiler.start()
    try:
        for repeat in range(args.repeats):
            order = list(CONDITIONS)
            random.Random(args.seed + repeat).shuffle(order)
            for condition in order:
                if rank >= 8:
                    kv.reset()
                dist.barrier()
                start = time.monotonic_ns()
                kv_rows, errors = [], []
                ready = threading.Event()
                def transfers():
                    try:
                        torch.npu.set_device(device)
                        ready.set()
                        for n in range(args.kv_transfers):
                            begin = time.monotonic_ns()
                            kv.transfer(peer)
                            end = time.monotonic_ns()
                            kv_rows.append({"type": "kv", "transfer": n, "start_ns": begin,
                                "end_ns": end, "wall_ms": (end - begin) / 1e6,
                                "bytes": kv.layout["bytes_per_rank"]})
                            if args.kv_gap_ms and n + 1 < args.kv_transfers:
                                time.sleep(args.kv_gap_ms / 1000)
                    except BaseException as exc:
                        errors.append(repr(exc))
                        ready.set()
                thread = None
                if condition == "MK" and rank >= 8:
                    thread = threading.Thread(target=transfers, daemon=True)
                    thread.start()
                    if not ready.wait(args.timeout):
                        raise RuntimeError("KV background thread did not start")
                moe_rows = replay.run(args.iterations) if replay and condition != "K" else []
                if condition == "SERIAL":
                    dist.barrier()  # All eight D ranks finish before any KV read starts.
                if condition in ("K", "SERIAL") and rank >= 8:
                    transfers()
                if thread:
                    thread.join(args.timeout)
                    if thread.is_alive():
                        raise RuntimeError("KV background read timed out; this run is invalid")
                if errors:
                    raise RuntimeError(f"KV transfer failure: {errors}")
                # Save actual local work span, not time spent waiting at the end barrier.
                work_end = time.monotonic_ns()
                dist.barrier()
                phase_end = time.monotonic_ns()
                if rank >= 8 and condition != "M":
                    kv.verify()
                for row in moe_rows + kv_rows:
                    write({**row, "repeat": repeat, "condition": condition})
                write({"type": "phase", "repeat": repeat, "condition": condition,
                       "start_ns": start, "work_end_ns": work_end, "end_ns": phase_end,
                       "phase_ms": (phase_end - start) / 1e6, "payload_ok": True})
                if profiler:
                    profiler.step()
                if rank == 0:
                    print(f"MOE_KV_PHASE_DONE repeat={repeat} condition={condition}", flush=True)
        if profiler:
            profiler.stop()
            profiler = None
        if replay:
            replay.verify()
        dist.barrier()
        if kv:
            kv.close()
        dist.barrier()
        write({"type": "complete", "ok": True})
        print(f"MOE_KV_COMPLETE rank={rank} file={path}", flush=True)
    except BaseException as exc:
        write({"type": "error", "error": repr(exc)})
        raise
    finally:
        out.close()
    dist.destroy_process_group()


def main():
    args = arguments()
    _, info = load(args.trace_dir)
    print(json.dumps({"capture": info, "kv": kv_layout(args.prefix_tokens),
                      "host": args.host, "processes": 8 if args.host == "228" else 4}, indent=2), flush=True)
    if args.dry_run:
        return
    if os.environ.get("QWEN_CAM82_HOST") != args.host or os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("Source env.sh for this host; keep all eight NPUs visible")
    if os.getenv("QWEN_MOE_CAPTURE_DIR"):
        raise RuntimeError("Unset QWEN_MOE_CAPTURE_DIR before standalone replay")
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(args,), nprocs=8 if args.host == "228" else 4, join=True)


if __name__ == "__main__":
    main()
