"""Version/source checks; --device imports native libraries, not a runtime certification."""
import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", action="store_true")
    p.add_argument("--model", action="store_true")
    args = p.parse_args()
    if md.version("qwen-vllm-cam82") != "0.1.3":
        raise RuntimeError("Install CAM82 adapter 0.1.3 in BOTH containers; see README 5.5")
    from qwen_cam82.policy import check_installation
    if platform.machine() != "aarch64" or tuple(map(int, platform.python_version_tuple()[:2])) != (3, 11):
        raise RuntimeError("Require A2 arm64 Python3.11 image")
    print("CAM82_SOURCE_OK", check_installation())
    from qwen_cam82.native_kv import ALIGNMENT, BLOCK_SIZE, NUM_BLOCKS
    print(f"CAM82_NATIVE_KV_POLICY block_tokens={BLOCK_SIZE} blocks={NUM_BLOCKS} "
          f"alignment={ALIGNMENT} (actual addresses checked during model startup)")
    cache = Path("/opt/mooncake-build/CMakeCache.txt").read_text()
    for setting in ("USE_ASCEND:BOOL=ON", "USE_ASCEND_DIRECT:BOOL=OFF", "WITH_STORE:BOOL=OFF"):
        if setting not in cache:
            raise RuntimeError("Wrong Mooncake build: " + setting)
    if args.device:
        import torch
        import torch_npu  # noqa: F401
        op = importlib.import_module("umdk_cam_op_lib")
        te = importlib.import_module("mooncake.engine").TransferEngine
        if torch.npu.device_count() != 8 or os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
            raise RuntimeError("Keep all8 physical devices visible")
        for attr in ("cam_get_comm", "cam_free_comm", "all2_all_detour"):
            if not hasattr(op, attr):
                raise RuntimeError("Missing CAM API " + attr)
        for attr in ("initialize", "register_memory", "transfer_sync_write", "get_rpc_port"):
            if not hasattr(te, attr):
                raise RuntimeError("Missing native PD TE API " + attr)
        from launch import check_connector, command
        cmd, _ = command("p-228")
        check_connector(json.loads(cmd[cmd.index("--kv-transfer-config") + 1]))
        print("CAM_NATIVE_IMPORT_OK", op.__file__)
    if args.model:
        path = Path(os.environ["MODEL_PATH"])
        cfg = json.loads((path / "config.json").read_text())
        if cfg.get("num_experts") != 128 or cfg.get("model_type") != "qwen3_moe":
            raise RuntimeError("Wrong model")
        shards = set(json.loads((path / "model.safetensors.index.json").read_text())["weight_map"].values())
        if not all((path / s).is_file() and (path / s).stat().st_size > 0 for s in shards):
            raise RuntimeError("Weights incomplete")
    print("CAM82_CHECK_PASS (not a device communication/model test)")


if __name__ == "__main__":
    main()
