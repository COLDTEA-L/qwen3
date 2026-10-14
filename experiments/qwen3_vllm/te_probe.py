"""Opt-in 1MiB NPU-to-NPU Mooncake READ; no model/store/MemFabric process.

Use only idle, allocated experiment cards. Keep the source process alive until
the reader completes. This checks data correctness, not all-rank PD correctness.
"""

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path

from runtime import HERE, HOSTS

BYTES = 1024 * 1024
PATTERN = 37


def checked_descriptor(data):
    host, separator, port = str(data["endpoint"]).rpartition(":")
    if not separator or host != HOSTS["228"] or not 0 < int(port) < 65536:
        raise ValueError("Only the current 228 source endpoint is accepted")
    if data["bytes"] != BYTES or data["pattern"] != PATTERN or int(data["pointer"]) <= 0:
        raise ValueError("Unexpected source allocation descriptor")
    age = time.time() - float(data["created_at"])
    if not -60 <= age < 900:
        raise ValueError("Stale source descriptor; use the live process's current file")
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=("serve", "read"))
    p.add_argument("--descriptor", type=Path, default=HERE / "runtime-config/te-source-228.json")
    args = p.parse_args()
    expected = "228" if args.role == "serve" else "238"
    if os.getenv("QWEN_VLLM_HOST") != expected:
        p.error(f"Source the new vLLM env.sh {expected} first")
    if args.role == "serve" and args.descriptor.exists():
        p.error("Descriptor exists; inspect the old process/file before a new test")
    remote = checked_descriptor(json.loads(args.descriptor.read_text())) if args.role == "read" else None
    # These are physical card 0 on each host, selected before importing Torch.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    import torch
    import torch_npu  # noqa: F401
    from mooncake.engine import TransferEngine

    torch.npu.set_device(0)
    tensor = torch.full((BYTES,), PATTERN if args.role == "serve" else 0,
                        dtype=torch.uint8, device="npu")
    torch.npu.synchronize()
    engine = TransferEngine()
    ret = engine.initialize(HOSTS[expected], "P2PHANDSHAKE", "ascend", "")
    if ret != 0:
        raise RuntimeError(f"Mooncake initialize failed: {ret}")
    ret = engine.register_memory(tensor.data_ptr(), BYTES)
    if ret != 0:
        raise RuntimeError(f"Mooncake register_memory failed: {ret}")
    try:
        if args.role == "serve":
            descriptor = {"endpoint": f"{HOSTS[expected]}:{engine.get_rpc_port()}",
                          "pointer": tensor.data_ptr(), "bytes": BYTES, "pattern": PATTERN,
                          "created_at": time.time()}
            args.descriptor.parent.mkdir(parents=True, exist_ok=True)
            with args.descriptor.open("x") as out:
                os.chmod(out.name, 0o600)
                json.dump(descriptor, out)
            print("TE_SOURCE_READY", json.dumps(descriptor), flush=True)
            stopped = threading.Event()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: stopped.set())
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline and not stopped.wait(1):
                pass
        else:
            start = time.perf_counter()
            ret = engine.transfer_sync_read(remote["endpoint"], tensor.data_ptr(), int(remote["pointer"]), BYTES)
            if ret != 0:
                raise RuntimeError(f"Mooncake NPU READ failed: {ret}")
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Device -> host is deliberate correctness checking, outside inference.
            if not bool((tensor.cpu() == PATTERN).all()):
                raise RuntimeError("NPU READ completed but data is incorrect")
            print("TE_NPU_READ_PASS", f"bytes={BYTES} elapsed_ms={elapsed_ms:.3f}", flush=True)
    finally:
        unregister = getattr(engine, "unregister_memory", None)
        if unregister is not None:
            unregister(tensor.data_ptr())
        # Do not delete the descriptor; it is retained as test evidence.


if __name__ == "__main__":
    main()
