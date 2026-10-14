"""All TP payloads use CAM A2A decomposition; no HCCL payload gather.

HCCL barriers synchronize slot reuse, not data. No silent transport fallback.
The specialized CAM AG/RS kernels are deliberately not invoked here.
"""
import logging
import math
import os
import threading

from .policy import BUFFER_MB, capacity

logger = logging.getLogger(__name__)


def bootstrap(device):
    import torch
    import umdk_cam_op_lib as op
    if os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("Keep all eight physical devices visible; role offset is set by worker")
    if torch.npu.current_device() != device:
        raise RuntimeError("Physical CAM/device rank mismatch")
    os.environ["CAM_BUFFSIZE"] = str(BUFFER_MB)
    os.environ.pop("CAM_COMM_ID", None)
    args = op.cam_get_comm(7, device, 8, "127.0.0.1:10067")
    torch.npu.synchronize()
    return args, op


class CamTP:
    def __init__(self, group, base, mode):
        import torch
        self.group, self.mode = group, mode
        self.lock, self.calls = threading.Lock(), 0
        if group.world_size != 4 or torch.npu.current_device() != base + group.rank_in_group:
            raise RuntimeError("Expected one local TP4 with explicit physical mapping")
        self.ranks = torch.tensor(list(range(base, base + 4)), device="npu", dtype=torch.int32)
        self.args = self.op = None
        if mode == "cam-detour":
            self.args, self.op = bootstrap(torch.npu.current_device())
        logger.warning("CAM_TP_READY mode=%s physical=%s global_tp_ranks=%s cam_tp_ranks=%s",
                       mode, torch.npu.current_device(), group.ranks, list(range(base, base + 4)))

    def exchange(self, packed):
        import torch
        import torch.distributed as dist
        if self.mode == "cam-detour":
            result = self.op.all2_all_detour(packed, self.ranks, self.args, 7)
        elif self.mode == "hccl-a2a":
            result = torch.empty_like(packed)
            dist.all_to_all_single(result, packed, group=self.group.device_group)
        else:
            raise RuntimeError("Off mode must not enter CAM payload")
        # Keep peers from reusing a ping-pong cell while a slower rank still
        # reads it. Eager correctness baseline; NOT an optimization claim.
        dist.barrier(group=self.group.device_group)
        return result

    def validate(self, x):
        import torch
        if x.device.type != "npu" or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("CAM requires NPU floating-point input")
        if torch.compiler.is_compiling():
            raise RuntimeError("CAM is eager-only")
        capture = getattr(torch.npu, "is_current_stream_capturing", None)
        if capture and capture():
            raise RuntimeError("CAM does not support captured device graphs")

    def record(self, name, x):
        self.calls += 1
        if self.calls <= 3 or self.calls % 100 == 0:
            logger.warning("CAM_TP_CALL mode=%s op=%s calls=%s bytes=%s",
                           self.mode, name, self.calls, x.numel() * x.element_size())

    def reduced_shard(self, shards):
        import torch
        n = shards.shape[1]
        out = torch.empty(n, dtype=shards.dtype, device=shards.device)
        for start in range(0, n, capacity(shards.element_size())):
            count = min(capacity(shards.element_size()), n - start)
            padded = math.ceil(count / 640) * 640
            packed = torch.zeros((4, padded), dtype=shards.dtype, device=shards.device)
            packed[:, :count].copy_(shards[:, start:start + count])
            received = self.exchange(packed)
            out[start:start + count].copy_(received.view(4, padded).float().sum(0).to(shards.dtype)[:count])
        return out

    def gathered_shards(self, flat):
        import torch
        n = flat.numel()
        out = torch.empty((4, n), dtype=flat.dtype, device=flat.device)
        for start in range(0, n, capacity(flat.element_size())):
            count = min(capacity(flat.element_size()), n - start)
            padded = math.ceil(count / 640) * 640
            # Each destination gets the same source slice, through CAM.
            packed = torch.zeros((4, padded), dtype=flat.dtype, device=flat.device)
            packed[:, :count].copy_(flat[start:start + count].expand(4, count))
            out[:, start:start + count].copy_(self.exchange(packed).view(4, padded)[:, :count])
        return out

    def all_reduce(self, x):
        import torch
        self.validate(x)
        if x.numel() == 0:
            return x.clone()
        with self.lock, torch.profiler.record_function("CAM_TP::all_reduce"):
            flat = torch.zeros(math.ceil(x.numel() / 4) * 4, dtype=x.dtype, device=x.device)
            flat[:x.numel()].copy_(x.contiguous().view(-1))
            result = self.gathered_shards(self.reduced_shard(flat.view(4, -1)))
            self.record("all_reduce", x)
            return result.reshape(-1)[:x.numel()].view_as(x)

    def all_gather(self, x, dim=-1):
        import torch
        self.validate(x)
        if not -x.ndim <= dim < x.ndim:
            raise ValueError("Invalid AllGather dimension")
        dim = dim % x.ndim
        moved = x.movedim(dim, 0).contiguous()
        with self.lock, torch.profiler.record_function("CAM_TP::all_gather"):
            gathered = self.gathered_shards(moved.view(-1))
            shape = (moved.shape[0] * 4, *moved.shape[1:])
            self.record("all_gather", x)
            return gathered.reshape(shape).movedim(0, dim).contiguous()

    def reduce_scatter(self, x, dim=-1):
        import torch
        self.validate(x)
        if not -x.ndim <= dim < x.ndim:
            raise ValueError("Invalid ReduceScatter dimension")
        dim = dim % x.ndim
        moved = x.movedim(dim, 0).contiguous()
        if moved.shape[0] % 4:
            raise ValueError("ReduceScatter requires four equal shards")
        with self.lock, torch.profiler.record_function("CAM_TP::reduce_scatter"):
            shard = self.reduced_shard(moved.reshape(4, moved.numel() // 4))
            shape = (moved.shape[0] // 4, *moved.shape[1:])
            self.record("reduce_scatter", x)
            return shard.reshape(shape).movedim(0, dim).contiguous()
