"""Single-host eight-card CAM gate, two TP4 subgroups active concurrently."""
import argparse
import os
from datetime import timedelta
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cam-detour", "hccl-a2a"), default="cam-detour")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 8 or os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error("Run eight local processes with all eight physical devices visible")
    from qwen_cam82.policy import check_installation
    check_installation()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from qwen_cam82.collectives import CamTP
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=600))
    try:
        groups = [dist.new_group(list(range(i, i + 4)), timeout=timedelta(seconds=600)) for i in (0, 4)]
        base = 0 if rank < 4 else 4
        group = groups[base // 4]
        tp = SimpleNamespace(world_size=4, rank_in_group=rank % 4, ranks=list(range(base, base + 4)), device_group=group)
        runtime = CamTP(tp, base, args.mode)
        for dtype in (torch.bfloat16, torch.float32):
            for n in (1, 7, 641, 2100001):
                x = torch.full((n,), rank % 4 + 1, dtype=dtype, device="npu")
                original = x.clone()
                expected = x.clone()
                dist.all_reduce(expected, group=group)
                actual = runtime.all_reduce(x)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.02)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
            for dim in (0, 1, -1):
                x = torch.full((8, 4), rank % 4 + 1, dtype=dtype, device="npu")
                chunks = [torch.empty_like(x) for _ in range(4)]
                dist.all_gather(chunks, x, group=group)
                torch.testing.assert_close(runtime.all_gather(x, dim), torch.cat(chunks, dim=dim), rtol=0, atol=0)
                expected = x.clone()
                dist.all_reduce(expected, group=group)
                expected = torch.tensor_split(expected, 4, dim=dim)[rank % 4].contiguous()
                torch.testing.assert_close(runtime.reduce_scatter(x, dim), expected, rtol=0.025, atol=0.02)
            x = torch.empty((0, 4), dtype=dtype, device="npu")
            assert runtime.all_reduce(x).shape == x.shape
            assert runtime.all_gather(x, 0).shape == (0, 4)
            assert runtime.reduce_scatter(x, 0).shape == (0, 4)
        dist.barrier()
        if rank == 0:
            print(f"CAM_TP_SMOKE_PASS mode={args.mode}; two TP4 groups, all8 owners, no model/KV", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
