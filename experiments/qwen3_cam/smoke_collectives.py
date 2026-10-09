"""Eight-card local CAM arithmetic test, independent of model weights."""

import argparse
import os
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hccl-a2a", "cam-detour"], default="cam-detour")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from sglang.srt.distributed.device_communicators.cam_detour import CamConfig, CamRuntime
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 8:
        raise ValueError("Use torchrun --nproc_per_node=8 on one host")
    groups = []
    for ranks in (list(range(4)), list(range(4, 8))):
        pg = dist.new_group(ranks, backend="hccl")
        if rank in ranks:
            selected = SimpleNamespace(ranks=ranks, device_group=pg, rank_in_group=ranks.index(rank))
        groups.append(pg)
    cfg = CamConfig.from_dict({"process_rank_to_cam": {str(r): r for r in range(8)},
                               "groups": [list(range(4)), list(range(4, 8))]})
    runtime = CamRuntime(cfg, args.mode)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for numel in (1, 259, 8192, 4 * 640 * 2048 + 4):
            # AllReduce includes unequal-to-four total sizes and chunk boundary cases.
            for iteration in range(args.iterations):
                torch.manual_seed(100 + rank + iteration)
                x = torch.randn(numel, dtype=dtype, device="npu") * 0.01
                expected = x.clone()
                dist.all_reduce(expected, group=selected.device_group)
                actual = runtime.all_reduce(x.clone(), selected)
                torch.npu.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
            if numel % 4 == 0:
                expected_rs = torch.empty(numel // 4, dtype=dtype, device="npu")
                dist.reduce_scatter_tensor(expected_rs, x, group=selected.device_group)
                actual_rs = torch.empty_like(expected_rs)
                runtime.reduce_scatter(actual_rs, x, selected)
                torch.npu.synchronize()
                torch.testing.assert_close(actual_rs, expected_rs, rtol=0.03, atol=0.003)
    print(f"SMOKE_PASS rank={rank} mode={args.mode} calls={runtime.calls}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
