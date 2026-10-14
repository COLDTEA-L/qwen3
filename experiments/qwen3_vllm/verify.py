"""Read-only image/runtime checks. Imports do not certify NPU/RoCE correctness."""

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from runtime import SHAS


def check():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", action="store_true")
    p.add_argument("--model", action="store_true")
    p.add_argument("--source", action="store_true")
    p.add_argument("--image-only", action="store_true", help="Metadata/provenance only; no device-library imports")
    p.add_argument("--moe-adapter", action="store_true", help="Check the opt-in All-to-All package and pinned source fingerprints")
    args = p.parse_args()
    failures = []
    print("PYTHON", platform.python_version(), "ARCH", platform.machine())
    if platform.machine() != "aarch64":
        failures.append("Require A2 aarch64 image")
    expected = {"vllm": "0.22.1", "vllm-ascend": "0.22.1rc1", "torch": "2.10.0",
                "torch-npu": "2.10.0", "triton-ascend": "3.2.1"}
    for name, version in expected.items():
        try:
            actual = md.version(name)
            print("PACKAGE", name, actual)
            if actual.split("+")[0] != version:
                failures.append(f"{name}: expected {version}, got {actual}")
        except md.PackageNotFoundError:
            failures.append(f"Missing {name}")
    cann_infos = list(Path("/usr/local/Ascend").glob("**/ascend_toolkit_install.info"))
    cann_ok = any("version=9.0.0" in f.read_text(errors="replace").splitlines() for f in cann_infos)
    print("CANN_9_0_METADATA", cann_ok)
    if not cann_ok:
        failures.append("CANN9.0 toolkit metadata missing")
    modules = () if args.image_only else ("torch_npu", "vllm_ascend", "mooncake.engine", "mooncake.store")
    for module in modules:
        try:
            mod = importlib.import_module(module)
            print("MODULE", module, getattr(mod, "__file__", None))
            if module == "mooncake.engine":
                for attr in ("initialize", "register_memory", "get_rpc_port", "transfer_sync_read"):
                    if not hasattr(mod.TransferEngine, attr):
                        failures.append("Missing TransferEngine API " + attr)
            if module == "mooncake.store" and not hasattr(mod, "MooncakeDistributedStore"):
                failures.append("Missing MooncakeDistributedStore")
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    print("MASTER", shutil.which("mooncake_master"))
    if not shutil.which("mooncake_master"):
        failures.append("Mooncake master binary missing")
    # Source-built Mooncake may not have pip metadata. Verify its build provenance.
    mc_source = Path("/vllm-workspace/Mooncake")
    if (mc_source / ".git").exists():
        sha = subprocess.check_output(["git", "-C", str(mc_source), "rev-parse", "HEAD"], text=True).strip()
        print("MOONCAKE_SOURCE", sha)
        if sha != SHAS["Mooncake"]:
            failures.append("Image Mooncake source is not pinned v0.3.9")
    else:
        print("MOONCAKE_SOURCE_UNKNOWN: preserve official image digest; report binary provenance")
        failures.append("Cannot verify Mooncake v0.3.9 source provenance")
    if args.device:
        import ctypes
        import torch
        try:
            ctypes.CDLL("libatb.so")
            print("ATB_LOAD_OK (not an ABI or operator correctness test)")
        except OSError as exc:
            failures.append("NNAL/ATB library: " + str(exc))
        if not torch.npu.is_available() or torch.npu.device_count() != 8:
            failures.append("Need 8 visible NPUs before role-specific selection")
        print("VISIBLE_NPUS", torch.npu.device_count())
        if not Path("/etc/hccn.conf").is_file():
            failures.append("/etc/hccn.conf is not mounted")
        proxies = [x for x in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY") if os.getenv(x)]
        if proxies:
            failures.append("Inference proxy still present (values omitted)")
    if args.model:
        root = Path(os.environ.get("MODEL_PATH", "/ssd_2/yuanwenliu/models/Qwen3-30B-A3B"))
        try:
            cfg = json.loads((root / "config.json").read_text())
            if cfg.get("model_type") != "qwen3_moe" or cfg.get("num_experts") != 128:
                failures.append("Not the target Qwen3 MoE configuration")
            index = json.loads((root / "model.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values())
            missing = [x for x in shards if not (root / x).is_file()]
            print("MODEL_SHARDS", len(shards), "MISSING", missing)
            if missing:
                failures.append("Missing model shards")
        except Exception as exc:
            failures.append("Local model: " + str(exc))
    if args.source:
        source = Path(os.environ.get("VLLM_ASCEND_SOURCE", "/ssd_2/yuanwenliu/vllm-ascend-qwen3"))
        sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        print("PROXY_SOURCE", sha)
        if sha != SHAS["vllm-ascend"]:
            failures.append("Proxy source is not pinned v0.22.1rc1")
    if args.moe_adapter:
        try:
            from qwen_moe_adapter.policy import check_installation
            print("MOE_ADAPTER_VERSION", md.version("qwen-vllm-moe-adapter"))
            print("MOE_ADAPTER_SOURCE_OK", check_installation())
        except Exception as exc:
            failures.append("MoE adapter: " + str(exc))
    if failures:
        raise SystemExit("QUALIFICATION_FAILED\n" + "\n".join(failures))
    print("QUALIFICATION_PASS (not an NPU transfer or inference test)")


if __name__ == "__main__":
    check()
