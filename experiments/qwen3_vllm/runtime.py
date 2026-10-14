"""Native vLLM-Ascend baseline commands; no SGLang/CAM imports or installs.

This is a migration gate, not an implementation of the three-flow controller.
Upstream references and the HBM-pool limitation are recorded in the README.
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/ssd_2/yuanwenliu"
HOSTS = {"228": "10.155.205.228", "238": "10.155.205.238"}
SHAS = {
    "vllm": "0decac0d96c42b49572498019f0a0e3600f50398",
    "vllm-ascend": "5f6faa0cb8830f667266f3b8121cd1383606f2a1",
    "Mooncake": "a00f75742469d6cc408bc5393b73c501e92dc74a",
}
IMAGE = "quay.io/ascend/vllm-ascend@sha256:638fc04eaa3654fcf14688096ed4e9d88ea0d905fa8685eed4b36d5fffe8fd8d"
ROLES = ("solo-228", "p-228", "d-238", "d8-228", "d8-238")


def connector(role, layout, pool=False):
    is_prefill = role == "p-228"
    if role == "solo-228":
        raise ValueError("Solo has no PD connector")
    kv_role = "kv_producer" if is_prefill else "kv_consumer"
    port = 20010 if is_prefill else (20020 if layout == "pd4" else 20040)
    p2p = {
        "kv_connector": "MooncakeConnectorV1", "kv_role": kv_role,
        "engine_id": "qwen-" + role, "kv_buffer_device": "npu", "kv_port": port,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 1, "tp_size": 4},
            "decode": {"dp_size": 1 if layout == "pd4" else 2, "tp_size": 4},
        },
    }
    if not pool:
        return p2p
    # Qualification only: A2 Mooncake's stock segment allocator uses host DRAM.
    store = {
        "kv_connector": "AscendStoreConnector", "kv_role": kv_role,
        "engine_id": "qwen-store-" + role,
        "kv_connector_extra_config": {
            "backend": "mooncake", "lookup_rpc_port": role,
        },
    }
    return {
        "kv_connector": "MultiConnector", "kv_role": kv_role,
        "engine_id": "qwen-multi-" + role, "kv_port": port,
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"connectors": [p2p, store]},
    }


def build_command(role, layout="pd4", pool=False, model=None,
                  max_len=2048, max_seqs=4, memory=0.75, moe_comm="native"):
    if role not in ROLES or layout not in ("pd4", "pd8"):
        raise ValueError("Unknown role/layout")
    if role == "solo-228" and layout != "pd4":
        raise ValueError("Solo is only the initial pd4 qualification stage")
    if role.startswith("d8-") != (layout == "pd8") and role != "p-228":
        if role != "solo-228":
            raise ValueError("d-238 uses pd4; d8-228/d8-238 use pd8")
    if max_len < 128 or max_seqs < 1 or not 0 < memory < 1:
        raise ValueError("Invalid token/concurrency/memory budget")
    if moe_comm not in ("native", "alltoall"):
        raise ValueError("MoE mode must be native or alltoall")
    if pool and role == "solo-228":
        raise ValueError("Pool qualification requires a PD role")
    model = model or f"{ROOT}/models/Qwen3-30B-A3B"
    if not Path(model).is_absolute():
        raise ValueError("Model must be an absolute local path")
    host = role.rsplit("-", 1)[1]
    devices = "4,5,6,7" if role == "d8-228" else "0,1,2,3"
    http_port = 30009 if role == "solo-228" else (30010 if role == "p-228" else 30011)
    env = {
        "ASCEND_RT_VISIBLE_DEVICES": devices,
        "VLLM_HOST_IP": HOSTS[host], "HCCL_IF_IP": HOSTS[host],
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
           "--served-model-name", "qwen3", "--host", "0.0.0.0", "--port", str(http_port),
           "--dtype", "bfloat16", "--tensor-parallel-size", "4",
           "--enable-expert-parallel", "--distributed-executor-backend", "mp",
           "--enforce-eager", "--max-model-len", str(max_len), "--max-num-seqs", str(max_seqs),
           "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", str(memory),
           "--seed", "1024", "--block-size", "128",
           "--additional-config", json.dumps({"enable_flashcomm1": False,
               "enable_matmul_allreduce": False, "enable_fused_mc2": 0,
               "enable_cpu_binding": False, "multistream_overlap_gate": False,
               "multistream_overlap_shared_expert": False, "enable_shared_expert_dp": False,
               "enable_flashcomm2_parallel_size": 0, "qwen_moe_comm": moe_comm}, separators=(",", ":"))]
    if moe_comm == "alltoall":
        cmd += ["--worker-cls", "qwen_moe_adapter.worker.QwenAllToAllWorker"]
    # External DP load balancing exposes one API endpoint per host. EP spans DP*TP.
    if role.startswith("d8-"):
        cmd += ["--data-parallel-size", "2", "--data-parallel-size-local", "1",
                "--data-parallel-rank", "0" if host == "228" else "1",
                "--data-parallel-address", HOSTS["228"], "--data-parallel-rpc-port", "29310",
                "--data-parallel-backend", "mp"]
    else:
        cmd += ["--data-parallel-size", "1"]
    if role != "solo-228":
        cmd += ["--kv-transfer-config", json.dumps(connector(role, layout, pool), separators=(",", ":"))]
    # Pool experiment disables in-engine prefix reuse to expose remote lookup.
    # This alone is NOT evidence that a request actually hits the remote store.
    cmd += ["--no-enable-prefix-caching"]
    return cmd, env


def write_configs(directory):
    directory = Path(directory)
    files = {
        "deployment.env": "\n".join([
            f"export MODEL_PATH={ROOT}/models/Qwen3-30B-A3B",
            f"export VLLM_ASCEND_SOURCE={ROOT}/vllm-ascend-qwen3",
            f"export UMDK_PATH={ROOT}/umdk-qwen3-cam", "",
        ]),
        "mooncake-dram-qualification.json": json.dumps({
            "metadata_server": "P2PHANDSHAKE", "protocol": "ascend", "device_name": "",
            "master_server_address": HOSTS["238"] + ":50088",
            "global_segment_size": "64MB", "local_buffer_size": "16MB",
            "preferred_segment": False, "prefer_alloc_in_same_node": False,
        }, indent=2) + "\n",
    }
    if any((directory / name).exists() for name in files):
        raise FileExistsError("Configs already exist; inspect them instead of overwriting")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, content in files.items():
        with (directory / name).open("x") as stream:
            os.chmod(stream.name, 0o600)
            stream.write(content)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("configure", "serve", "proxy", "master"))
    p.add_argument("--role", choices=ROLES)
    p.add_argument("--layout", choices=("pd4", "pd8"), default="pd4")
    p.add_argument("--pool-dram-qualification", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--profile-dir", type=Path, help="Optional absolute NPU profiler output directory")
    p.add_argument("--moe-comm", choices=("native", "alltoall"), default="native",
                   help="native is the control; alltoall is an opt-in A2 experiment, not CAM")
    args = p.parse_args()
    if args.action == "configure":
        if args.dry_run:
            print("Will create private configs in", HERE / "runtime-config")
            return
        write_configs(HERE / "runtime-config")
        print("VLLM_PRIVATE_CONFIG_READY", HERE / "runtime-config")
        return
    env = {}
    if args.action == "serve":
        if args.role is None:
            p.error("serve requires --role")
        try:
            cmd, env = build_command(args.role, args.layout, args.pool_dram_qualification,
                os.getenv("MODEL_PATH"), args.max_model_len, args.max_num_seqs, args.gpu_memory_utilization, args.moe_comm)
        except ValueError as exc:
            p.error(str(exc))
        if args.profile_dir:
            if not args.profile_dir.is_absolute():
                p.error("--profile-dir must be absolute")
            cmd += ["--profiler-config", json.dumps({"profiler": "torch",
                "torch_profiler_dir": str(args.profile_dir), "torch_profiler_with_stack": False})]
        expected_host = args.role.rsplit("-", 1)[1]
    elif args.action == "proxy":
        source = Path(os.getenv("VLLM_ASCEND_SOURCE", f"{ROOT}/vllm-ascend-qwen3"))
        proxy = source / "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
        cmd = [sys.executable, str(proxy), "--host", "0.0.0.0", "--port", "30080",
               "--prefiller-hosts", HOSTS["228"], "--prefiller-ports", "30010",
               "--decoder-hosts"]
        cmd += [HOSTS["238"]] if args.layout == "pd4" else [HOSTS["228"], HOSTS["238"]]
        cmd += ["--decoder-ports"] + (["30011"] if args.layout == "pd4" else ["30011", "30011"])
        expected_host = "228"
        if not args.dry_run and not proxy.is_file():
            p.error("Pinned vLLM-Ascend proxy source is missing")
    else:
        cmd = ["mooncake_master", "--port", "50088"]
        expected_host = "238"
    print("ENV", shlex.join([f"{k}={v}" for k, v in env.items()]))
    print("COMMAND", shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if os.getenv("QWEN_VLLM_HOST") != expected_host:
        p.error(f"Source qwen3_vllm/env.sh {expected_host} in the new vLLM container")
    if args.action == "serve" and os.getenv("QWEN_CAM_MODE", "off") != "off":
        p.error("CAM is not ported to this baseline; only QWEN_CAM_MODE=off is supported")
    if args.action == "serve" and args.moe_comm == "alltoall":
        try:
            from qwen_moe_adapter.policy import check_installation
            check_installation()
        except (ImportError, RuntimeError) as exc:
            p.error(f"MoE adapter qualification failed: {exc}")
    if args.pool_dram_qualification:
        cfg = HERE / "runtime-config/mooncake-dram-qualification.json"
        if not cfg.is_file():
            p.error("Run configure first")
        os.environ["MOONCAKE_CONFIG_PATH"] = str(cfg)
    os.execvpe(cmd[0], cmd, {**os.environ, **env})


if __name__ == "__main__":
    main()
