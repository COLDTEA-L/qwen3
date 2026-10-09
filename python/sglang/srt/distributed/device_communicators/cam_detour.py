# SPDX-License-Identifier: Apache-2.0
"""Opt-in, eager-only CAM experiment for Qwen3 Attention TP collectives.

MoE dispatch/combine and PD transport deliberately remain native. This adapter
is not a Fabric scheduler. A host-local CAM domain includes relay memory owners
which can belong to another inference engine or a dedicated KV pool process.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
MODES = {"off", "hccl-a2a", "cam-detour"}
IPC_DATA_OFFSET = 2 * 1024 * 1024


@dataclass(frozen=True)
class CamConfig:
    rank_map: dict[int, int]
    groups: tuple[tuple[int, ...], ...]
    server_addr: str = "127.0.0.1:10067"
    comm_id: int = 7
    physical_size: int = 8
    chunk_bytes: int = 8 * 1024 * 1024
    buffer_mb: int = 512

    @classmethod
    def from_dict(cls, data: dict):
        known = {
            "process_rank_to_cam", "groups", "server_addr", "comm_id",
            "physical_size", "chunk_bytes", "buffer_mb",
        }
        if unknown := set(data) - known:
            raise ValueError(f"Unknown CAM config keys: {sorted(unknown)}")
        mapping = {int(k): int(v) for k, v in data["process_rank_to_cam"].items()}
        groups = tuple(tuple(int(r) for r in g) for g in data["groups"])
        result = cls(mapping, groups, **{k: v for k, v in data.items() if k not in
                                       {"process_rank_to_cam", "groups"}})
        if result.physical_size != 8 or not 0 <= result.comm_id < 64:
            raise ValueError("This experiment requires physical_size=8 and comm_id in [0,64)")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Two process ranks cannot own the same CAM rank")
        if any(not 0 <= v < 8 for v in mapping.values()):
            raise ValueError("CAM ranks must be in [0,8)")
        if not groups or any(len(g) != 4 or len(set(g)) != 4 for g in groups):
            raise ValueError("Only ordered, four-rank local TP groups are supported")
        if any(r not in mapping for g in groups for r in g):
            raise ValueError("Every selected group rank needs a process_rank_to_cam mapping")
        if len({r for g in groups for r in g}) != sum(map(len, groups)):
            raise ValueError("Selected groups must be disjoint")
        if not 1 <= result.buffer_mb <= 2000 or result.chunk_bytes < 4 * 640 * 4:
            raise ValueError("Invalid buffer_mb or chunk_bytes")
        # CAM divides each rank window into 8*8 cells, each with two slots.
        # A slot reserves 2 MiB for flags BEFORE its payload.
        available = result.buffer_mb * 1024 * 1024 // 64 // 2 - IPC_DATA_OFFSET
        worst_slice = math.ceil(result.chunk_bytes / 4 / 5) + 128 * 4
        if available <= 0 or worst_slice > available:
            raise ValueError("CAM payload does not fit: increase buffer_mb or reduce chunk_bytes")
        return result

    def cam_group(self, ranks):
        ranks = tuple(ranks)
        if ranks not in self.groups:
            return None
        return tuple(self.rank_map[r] for r in ranks)


def padded_shard_elements(elements: int, element_size: int, chunk_bytes: int):
    """Align all five detour slices to >=256 bytes (128 elements)."""
    capacity = chunk_bytes // (4 * element_size)
    alignment = 5 * 128
    capacity = capacity // alignment * alignment
    if elements < 0 or capacity <= 0:
        raise ValueError("Invalid shard/chunk size")
    return min(elements, capacity), capacity


class CamRuntime:
    def __init__(self, config: CamConfig, mode: str):
        import torch
        import torch.distributed as dist

        self.config, self.mode = config, mode
        self.lock = threading.Lock()
        self.calls = 0
        self.trace_path = os.environ.get("SGLANG_CAM_DETOUR_TRACE")
        self.rank = dist.get_rank()
        if self.rank not in config.rank_map:
            raise ValueError(f"Process rank {self.rank} has no CAM mapping")
        self.cam_rank = config.rank_map[self.rank]
        self.args = None
        self.op = None
        self.rank_tensors = {}
        if torch.npu.current_device() != self.cam_rank:
            raise ValueError("Physical device must equal CAM rank; do not hide/reindex cards")
        if mode == "cam-detour":
            self.args, self.op = bootstrap(config, self.cam_rank)
        logger.warning("CAM_READY mode=%s process_rank=%s cam_rank=%s groups=%s",
                       mode, self.rank, self.cam_rank, config.groups)

    def _validate(self, x, group):
        import torch

        if self.config.cam_group(group.ranks) is None:
            return False
        if x.device.type != "npu" or x.dtype not in (
            torch.float16, torch.bfloat16, torch.float32
        ):
            raise ValueError("Selected CAM group requires NPU fp16/bf16/fp32 tensors")
        if torch.compiler.is_compiling():
            raise RuntimeError("CAM detour experiment does not support torch.compile")
        capturing = getattr(torch.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("Disable device graph capture for CAM detour")
        if tuple(group.ranks)[group.rank_in_group] != self.rank:
            raise ValueError("Group order/rank mismatch")
        return True

    def _exchange(self, packed, group):
        import torch
        import torch.distributed as dist

        if self.mode == "hccl-a2a":
            result = torch.empty_like(packed)
            dist.all_to_all_single(result, packed, group=group.device_group)
            return result
        key = tuple(group.ranks)
        if key not in self.rank_tensors:
            self.rank_tensors[key] = torch.tensor(
                self.config.cam_group(key), dtype=torch.int32, device=packed.device
            )
        return self.op.all2_all_detour(
            packed, self.rank_tensors[key], self.args, self.config.comm_id
        )

    def _reduced_shards(self, x, group):
        import torch

        flat = x.contiguous().view(-1)
        shard_size = flat.numel() // 4
        _, capacity = padded_shard_elements(
            shard_size, x.element_size(), self.config.chunk_bytes
        )
        shards = flat.view(4, shard_size)
        for start in range(0, shard_size, capacity):
            count = min(capacity, shard_size - start)
            padded = math.ceil(count / 640) * 640
            packed = torch.zeros((4, padded), dtype=x.dtype, device=x.device)
            packed[:, :count].copy_(shards[:, start:start + count])
            with torch.profiler.record_function("CAM_DETOUR::attention_tp_exchange"):
                received = self._exchange(packed, group)
            # CAM receive order is source rank order, matching the selected TP group.
            reduced = received.view(4, padded).float().sum(dim=0).to(x.dtype)[:count]
            yield start, count, reduced

    def all_reduce(self, x, group):
        import torch
        import torch.distributed as dist

        if not self._validate(x, group):
            return None
        if x.numel() == 0:
            return x
        original_count = x.numel()
        padded_count = math.ceil(original_count / 4) * 4
        flat = torch.zeros(padded_count, dtype=x.dtype, device=x.device)
        flat[:original_count].copy_(x.contiguous().view(-1))
        output = torch.empty_like(flat).view(4, -1)
        with self.lock, torch.profiler.record_function("CAM_DETOUR::attention_tp_all_reduce"):
            for start, count, reduced in self._reduced_shards(flat, group):
                gathered = torch.empty(4 * count, dtype=x.dtype, device=x.device)
                # This gather is native HCCL; ONLY the exchange leg is CAM.
                dist.all_gather_into_tensor(gathered, reduced.contiguous(), group=group.device_group)
                output[:, start:start + count].copy_(gathered.view(4, count))
            x.copy_(output.view(-1)[:original_count].view_as(x))
            self._record("all_reduce", original_count * x.element_size(), group)
        return x

    def reduce_scatter(self, output, x, group):
        import torch
        import torch.distributed as dist

        if not self._validate(x, group):
            return False
        if x.numel() != 4 * output.numel() or output.dtype != x.dtype or output.device != x.device:
            raise ValueError("ReduceScatter requires matching dtype/device and equal shards")
        if x.numel() == 0:
            return True
        with self.lock, torch.profiler.record_function("CAM_DETOUR::attention_tp_reduce_scatter"):
            target = torch.empty(output.numel(), dtype=x.dtype, device=x.device)
            for start, count, reduced in self._reduced_shards(x, group):
                target[start:start + count].copy_(reduced)
                # One control-only barrier per chunk prevents ping-pong reuse
                # while a slower peer is still reading the previous slot.
                # Baseline uses the same barrier. This is NOT a performance optimization.
                dist.barrier(group=group.device_group)
            output.copy_(target.view_as(output))
            self._record("reduce_scatter", x.numel() * x.element_size(), group)
        return True

    def _record(self, kind, nbytes, group):
        import time

        self.calls += 1
        if self.calls <= 3 or self.calls % 100 == 0:
            logger.warning("CAM_CALL mode=%s op=%s rank=%s calls=%s bytes=%s",
                           self.mode, kind, self.rank, self.calls, nbytes)
        if self.trace_path:
            # Host submission timestamps, NOT device execution timestamps.
            path = Path(self.trace_path.replace("{rank}", str(self.rank)))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"clock": "host_submit", "time_ns": time.time_ns(),
                                         "op": kind, "bytes": nbytes,
                                         "rank": self.rank, "cam_group": self.config.cam_group(group.ranks),
                                         "mode": self.mode, "call": self.calls}) + "\n")


def bootstrap(config: CamConfig, cam_rank: int):
    """Called by all eight physical memory owners, including KV pool sidecars."""
    import torch
    import umdk_cam_op_lib

    if torch.npu.current_device() != cam_rank:
        raise ValueError("CAM rank must match the physical NPU ID")
    if os.environ.get("CAM_COMM_ID") not in (None, config.server_addr):
        raise ValueError("Unset CAM_COMM_ID or make it equal to config.server_addr")
    os.environ["CAM_BUFFSIZE"] = str(config.buffer_mb)
    args = umdk_cam_op_lib.cam_get_comm(config.comm_id, cam_rank, 8, config.server_addr)
    torch.npu.synchronize()
    return args, umdk_cam_op_lib


_runtime = None


def initialize_qwen3_cam():
    global _runtime
    mode = os.environ.get("SGLANG_CAM_DETOUR_MODE", "off")
    if mode not in MODES:
        raise ValueError(f"Unknown SGLANG_CAM_DETOUR_MODE={mode}")
    if mode == "off" or _runtime is not None:
        return
    if os.environ.get("SGLANG_CAM_DETOUR_ACK_EAGER") != "1":
        raise RuntimeError("Set SGLANG_CAM_DETOUR_ACK_EAGER=1 only with graphs, compile and TBO disabled")
    import torch
    if not hasattr(torch, "npu"):
        raise RuntimeError("CAM experiment requires torch_npu")
    filename = os.environ.get("SGLANG_CAM_DETOUR_CONFIG")
    if not filename:
        raise ValueError("SGLANG_CAM_DETOUR_CONFIG must point to the role's JSON file")
    config = CamConfig.from_dict(json.loads(Path(filename).read_text()))
    _runtime = CamRuntime(config, mode)


def maybe_cam_all_reduce(x, group):
    return None if _runtime is None else _runtime.all_reduce(x, group)


def maybe_cam_reduce_scatter(output, x, group):
    return False if _runtime is None else _runtime.reduce_scatter(output, x, group)


def trace_moe_dispatcher(experts):
    """Name native MoE regions for profiling without replacing their transport."""
    if os.environ.get("SGLANG_CAM_TRACE_MOE") != "1":
        return
    from functools import wraps
    import torch

    dispatcher = getattr(experts, "dispatcher", None)
    if dispatcher is None or getattr(dispatcher, "_cam_trace_installed", False):
        return

    def wrap(original, name):
        @wraps(original)
        def traced(*args, **kwargs):
            with torch.profiler.record_function(f"NATIVE_MOE::{name}"):
                return original(*args, **kwargs)
        return traced

    for name in ("dispatch", "combine"):
        if hasattr(dispatcher, name):
            setattr(dispatcher, name, wrap(getattr(dispatcher, name), name))
    dispatcher._cam_trace_installed = True
