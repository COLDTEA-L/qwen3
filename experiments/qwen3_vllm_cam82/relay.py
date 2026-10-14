"""238 cards 4-7 hold CAM windows; no model, no KV pool, no distributed PG."""
import os
import signal
import threading


def main():
    import torch
    import torch_npu  # noqa: F401
    from qwen_cam82.policy import check_installation
    from qwen_cam82.collectives import bootstrap
    check_installation()
    if os.environ.get("QWEN_CAM82_HOST") != "238":
        raise RuntimeError("Relays are only for 238 cards4-7")
    device = 4 + int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(device)
    args, op = bootstrap(device)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *a: stop.set())
    print(f"CAM_RELAY_READY physical={device} (no KV pool)", flush=True)
    stop.wait()
    torch.npu.synchronize()
    op.cam_free_comm(7)


if __name__ == "__main__":
    main()
