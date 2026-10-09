"""Verify actual remote MemCache buffer put/get, not synthetic HCCL KV traffic.

Configure the client with zero HBM and zero DRAM capacity; the four pool
sidecars must be the only positive-capacity services in this metadata domain.
"""

import argparse
import json
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-config", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--mib", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.mib <= 0 or args.iterations <= 0:
        raise ValueError("Payload and iteration count must be positive")
    import torch
    import torch_npu  # noqa: F401
    from memcache_hybrid import DistributedObjectStore, LocalConfig
    torch.npu.set_device(args.device)
    settings = json.loads(Path(args.client_config).read_text())
    if settings.get("hbm_size") not in (0, "0") or settings.get("dram_size") not in (0, "0"):
        raise ValueError("Use a zero-capacity client; otherwise placement is not a dedicated-pool test")
    config = LocalConfig()
    for key, value in settings.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown LocalConfig field {key}")
        setattr(config, key, value)
    store = DistributedObjectStore()
    if store.setup(config) != 0 or store.init(args.device, True) != 0:
        raise RuntimeError("Client startup failed; verify installed MemCache accepts zero-capacity clients")
    count = args.mib * 1024 * 1024 // 4
    source = torch.arange(count, device="npu", dtype=torch.int32)
    target = torch.empty_like(source)
    nbytes = source.numel() * source.element_size()
    for tensor in (source, target):
        if store.register_buffer(tensor.data_ptr(), nbytes) != 0:
            raise RuntimeError("NPU buffer registration failed")
    torch.npu.synchronize()
    key = "cam_pool_probe_" + uuid.uuid4().hex
    try:
        for iteration in range(args.iterations):
            target.zero_()
            torch.npu.synchronize()
            start = time.perf_counter()
            put = store.batch_put_from([key], [source.data_ptr()], [nbytes])
            get = store.batch_get_into([key], [target.data_ptr()], [nbytes])
            torch.npu.synchronize()
            for result in (put, get):
                statuses = list(result) if not isinstance(result, int) else [result]
                if any(int(status) != 0 for status in statuses):
                    raise RuntimeError(f"MemCache put/get failed: {result}")
            torch.testing.assert_close(target, source, rtol=0, atol=0)
            print(json.dumps({"iteration": iteration, "bytes": nbytes, "put_get_ms":
                              (time.perf_counter() - start) * 1000,
                              "clock": "host_completed", "checksum": "PASS"}), flush=True)
    finally:
        # Delete only this probe's unique key, never clear the shared pool.
        if hasattr(store, "remove"):
            store.remove(key)
        if hasattr(store, "close"):
            store.close()


if __name__ == "__main__":
    main()
