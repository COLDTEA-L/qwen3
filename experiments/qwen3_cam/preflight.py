"""Read-only dependency and configuration preflight; never downloads weights."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--mode", choices=["off", "hccl-a2a", "cam-detour"], default="cam-detour")
    parser.add_argument("--require-pool", action="store_true")
    parser.add_argument("--import-model", action="store_true", help="Import model code without loading weights")
    args = parser.parse_args()
    from sglang.srt.distributed.device_communicators.cam_detour import CamConfig
    config = CamConfig.from_dict(json.loads(Path(args.config).read_text()))
    print("CONFIG_OK", config)
    failed = []
    for name in ("torch", "torch_npu", "sglang", "sgl_kernel_npu", "deep_ep", "memfabric_hybrid"):
        try:
            module = importlib.import_module(name)
            print("IMPORT_OK", name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", ""))
        except Exception as exc:
            failed.append(name)
            print("IMPORT_FAILED", name, type(exc).__name__, str(exc))
    if args.mode == "cam-detour":
        try:
            module = importlib.import_module("umdk_cam_op_lib")
            for name in ("cam_get_comm", "all2_all_detour"):
                if not hasattr(module, name):
                    raise RuntimeError(f"Missing CAM API {name}")
            print("CAM_IMPORT_OK", module.__file__)
        except Exception as exc:
            failed.append("umdk_cam_op_lib")
            print("CAM_IMPORT_FAILED", str(exc))
    if args.import_model:
        try:
            model = importlib.import_module("sglang.srt.models.qwen3_moe")
            print("MODEL_IMPORT_OK", model.__file__)
        except Exception as exc:
            failed.append("model_import")
            print("MODEL_IMPORT_FAILED", type(exc).__name__, str(exc))
    if args.require_pool:
        try:
            from memcache_hybrid import DistributedObjectStore, LocalConfig, MetaConfig, MetaService
            cfg = LocalConfig()
            for name in ("world_size", "protocol", "dram_size", "hbm_size", "max_hbm_size",
                         "meta_service_url", "config_store_url"):
                if not hasattr(cfg, name):
                    raise RuntimeError(f"Missing LocalConfig field {name}")
            for name in ("setup", "init", "register_buffer", "batch_put_from", "batch_get_into"):
                if not hasattr(DistributedObjectStore, name):
                    raise RuntimeError(f"Missing MemCache API {name}")
            print("POOL_API_OK", importlib.metadata.version("memcache-hybrid"),
                  importlib.metadata.version("memfabric-hybrid"))
        except Exception as exc:
            failed.append("memcache_hybrid")
            print("POOL_API_FAILED", type(exc).__name__, str(exc))
    try:
        import torch
        if not torch.npu.is_available() or torch.npu.device_count() != 8:
            raise RuntimeError("Require all eight local physical NPUs visible")
        print("DEVICES_OK", torch.npu.device_count())
    except Exception as exc:
        failed.append("NPU")
        print("NPU_FAILED", str(exc))
    if os.environ.get("ASCEND_MF_TRANSFER_PROTOCOL", "device_rdma") == "device_rdma":
        try:
            from memfabric_hybrid import TransferEngine
            if not hasattr(TransferEngine.TransDataOpType, "DEVICE_RDMA"):
                raise RuntimeError("Installed MemFabric has no DEVICE_RDMA transport; do not accept SDMA fallback")
        except Exception as exc:
            failed.append("device_rdma")
            print("RDMA_FAILED", str(exc))
    if args.model_path:
        cfg = json.loads((Path(args.model_path) / "config.json").read_text())
        expected = {"model_type": "qwen3_moe", "num_hidden_layers": 48,
                    "num_experts": 128, "num_experts_per_tok": 8,
                    "num_key_value_heads": 4, "head_dim": 128}
        if any(cfg.get(k) != v for k, v in expected.items()):
            failed.append("model_config")
        print("MODEL_CONFIG", {k: cfg.get(k) for k in expected})
        shards = list(Path(args.model_path).glob("*.safetensors"))
        print("WEIGHT_SHARDS", len(shards))
        if not shards:
            failed.append("weights")
        index = Path(args.model_path) / "model.safetensors.index.json"
        if index.is_file():
            names = set(json.loads(index.read_text())["weight_map"].values())
            if any(not (Path(args.model_path) / name).is_file() for name in names):
                failed.append("missing_weight_shard")
    print("VISIBLE_DEVICES", os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not reindexed"))
    if failed:
        raise SystemExit("PREFLIGHT_FAILED " + ",".join(failed))
    print("PREFLIGHT_PASS (does not certify kernel correctness or PD compatibility)")


if __name__ == "__main__":
    main()
