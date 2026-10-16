"""CANN8.2 P4+D8: native Qwen MoE, opt-in CAM TP. CPU-safe dry-run."""
import argparse
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

ROOT = "/ssd_2/yuanwenliu"
ROLES = {"p-228": ("228", 0), "d8-228": ("228", 4), "d8-238": ("238", 0)}


def command(role, mode="off", sp=True, model=None, max_len=2048, max_seqs=4, memory=0.75):
    if role not in ROLES or mode not in ("off", "hccl-a2a", "cam-detour"):
        raise ValueError("Unknown role/mode")
    if max_len < 128 or not 1 <= max_seqs <= 4 or not 0 < memory < 1:
        raise ValueError("Initial qualification requires context>=128, max-seqs1..4, memory in (0,1)")
    host, base = ROLES[role]
    p = role == "p-228"
    model = model or ROOT + "/models/Qwen3-30B-A3B"
    if not Path(model).is_absolute():
        raise ValueError("Use an absolute local model path")
    kv = {"kv_connector": "MooncakeConnectorV1", "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector",
          "kv_role": "kv_producer" if p else "kv_consumer", "kv_buffer_device": "npu",
          "engine_id": "cam82-" + role, "kv_port": 20010 if p else 20040,
          "kv_connector_extra_config": {"use_ascend_direct": False,
              "prefill": {"tp_size": 4, "dp_size": 1}, "decode": {"tp_size": 4, "dp_size": 2}}}
    cmd = [sys.executable, "-m", "qwen_cam82.entrypoint", "serve", model,
           "--served-model-name", "qwen3", "--host", "0.0.0.0", "--port", "30010" if p else "30011",
           "--dtype", "bfloat16", "--tensor-parallel-size", "4", "--enable-expert-parallel",
           "--distributed-executor-backend", "mp", "--worker-cls", "qwen_cam82.worker.QwenCamWorker",
           "--enforce-eager", "--max-model-len", str(max_len), "--max-num-seqs", str(max_seqs),
           "--max-num-batched-tokens", str(min(1024, max_len)), "--gpu-memory-utilization", str(memory), "--seed", "1024",
           "--block-size", "128", "--no-enable-prefix-caching", "--kv-transfer-config", json.dumps(kv),
           "--compilation-config", json.dumps({"level": 0, "pass_config": {"enable_sequence_parallelism": sp}})]
    if p:
        cmd += ["--data-parallel-size", "1"]
    else:
        cmd += ["--data-parallel-size", "2", "--data-parallel-size-local", "1",
                "--data-parallel-rank", "0" if host == "228" else "1",
                "--data-parallel-address", "10.155.205.228", "--data-parallel-rpc-port", "29310",
                "--data-parallel-backend", "mp"]
    env = {"QWEN_CAM82_ROLE": role, "QWEN_CAM82_MODE": mode,
           "PHYSICAL_DEVICES": ",".join(str(base + i) for i in range(4)),
           "VLLM_HOST_IP": "10.155.205." + host, "HCCL_IF_IP": "10.155.205." + host,
           "VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP": "0", "VLLM_USE_V1": "1"}
    return cmd, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "proxy"))
    parser.add_argument("--role", choices=tuple(ROLES))
    parser.add_argument("--tp-mode", choices=("off", "hccl-a2a", "cam-detour"), default="off")
    parser.add_argument("--no-sp", action="store_true", help="Diagnostic only: disable native prefill SP (RS may then not occur)")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env = {}
    if args.action == "serve":
        if not args.role:
            parser.error("serve requires --role")
        try:
            cmd, env = command(args.role, args.tp_mode, not args.no_sp, os.getenv("MODEL_PATH"),
                               args.max_model_len, args.max_num_seqs, args.gpu_memory_utilization)
        except ValueError as exc:
            parser.error(str(exc))
        host = ROLES[args.role][0]
    else:
        host = "228"
        proxy = ROOT + "/vllm-ascend-cam82/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
        cmd = [sys.executable, proxy, "--host", "0.0.0.0", "--port", "30080",
               "--prefiller-hosts", "10.155.205.228", "--prefiller-ports", "30010",
               "--decoder-hosts", "10.155.205.228", "10.155.205.238", "--decoder-ports", "30011", "30011"]
    print("ENV", shlex.join([f"{k}={v}" for k, v in env.items()]))
    print("COMMAND", shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if os.getenv("QWEN_CAM82_HOST") != host or os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
        parser.error(f"Source cam82/env.sh {host}; do not hide/reindex physical cards")
    if args.action == "serve" and importlib.util.find_spec("qwen_cam82.entrypoint") is None:
        parser.error("Installed CAM82 adapter is old. Follow README 5.1 to install adapter 0.1.1 offline in BOTH containers")
    from qwen_cam82.policy import check_installation
    check_installation()
    os.execvpe(cmd[0], cmd, {**os.environ, **env})


if __name__ == "__main__":
    main()
