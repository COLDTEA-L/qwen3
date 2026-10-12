"""Keep four 238-host relay windows and optional MemCache HBM pools alive.

Launch on 238 with torchrun --nproc_per_node=4; no torch distributed PG is needed.
Use full physical visibility: LOCAL_RANK 0..3 selects NPU 4..7 here.
"""

import argparse
import json
import os
import signal
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool-config", help="Optional MemCache LocalConfig JSON")
    parser.add_argument("--mode", choices=["off", "hccl-a2a", "cam-detour"], default="cam-detour")
    args = parser.parse_args()
    import torch
    import torch_npu  # noqa: F401
    from sglang.srt.distributed.device_communicators.cam_detour import CamConfig, bootstrap

    device = 4 + int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    config = CamConfig.from_dict(json.loads(Path(args.config).read_text()))
    pool = None
    if args.pool_config:
        from memcache_hybrid import DistributedObjectStore, LocalConfig
        settings = json.loads(Path(args.pool_config).read_text())
        cfg = LocalConfig()
        for key, value in settings.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown MemCache LocalConfig field {key}")
            setattr(cfg, key, value)
        pool = DistributedObjectStore()
        if pool.setup(cfg) != 0 or pool.init(device, True) != 0:
            raise RuntimeError(f"MemCache startup failed on device {device}")
        print(f"KV_POOL_READY physical_device={device}", flush=True)
    comm_args = op = None
    if args.mode == "cam-detour":
        comm_args, op = bootstrap(config, device)
    print(f"RELAY_READY device={device} mode={args.mode} pool={pool is not None}", flush=True)
    stop.wait()
    # Keep all three objects alive until all inference workers have stopped.
    if op is not None:
        torch.npu.synchronize()
        op.cam_free_comm(config.comm_id)


if __name__ == "__main__":
    main()
