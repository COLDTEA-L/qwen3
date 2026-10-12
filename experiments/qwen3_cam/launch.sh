#!/usr/bin/env bash
# Run inside an already-qualified A2 NPU environment; no installation side effects.
set -euo pipefail
role="${1:?usage: bash launch.sh p-228|d-228|d-238 [extra launch_server args]}"
shift
script_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "${script_dir}/../.." && pwd)
export PYTHONPATH="${repo_dir}/python${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_PATH:?Set MODEL_PATH to local Qwen3-30B-A3B BF16 weights on the A2 hosts}"
: "${HOST_228_IP:?Source the current env.sh first}"
: "${ASCEND_MF_STORE_URL:?Set ASCEND_MF_STORE_URL to the reachable memfabric store endpoint}"
export ASCEND_MF_TRANSFER_PROTOCOL="${ASCEND_MF_TRANSFER_PROTOCOL:-device_rdma}"
export SGLANG_CAM_DETOUR_MODE="${SGLANG_CAM_DETOUR_MODE:-off}"
export SGLANG_CAM_DETOUR_ACK_EAGER=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0
unset CAM_COMM_ID
export CAM_BUFFSIZE=512
common=(--model-path "$MODEL_PATH" --device npu --attention-backend ascend
        --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static 0.65
        --context-length 16384 --disable-cuda-graph --disable-overlap-schedule
        --disaggregation-transfer-backend ascend --enable-metrics
        --moe-a2a-backend deepep --deepep-mode normal)
case "$role" in
  p-228)
    export SGLANG_CAM_DETOUR_CONFIG="${script_dir}/configs/p_228.json"
    args=(--tp-size 4 --ep-size 4 --attn-dp-size 1 --nnodes 1 --node-rank 0
          --base-gpu-id 0 --port 30000 --disaggregation-mode prefill
          --disaggregation-bootstrap-port 8998 --max-running-requests 16)
    ;;
  d-228|d-238)
    if [ "$role" = d-228 ]; then
      export SGLANG_CAM_DETOUR_CONFIG="${script_dir}/configs/d_228.json"
      rank=0; base=4
    else
      export SGLANG_CAM_DETOUR_CONFIG="${script_dir}/configs/d_238.json"
      rank=1; base=0
    fi
    args=(--tp-size 8 --ep-size 8 --attn-dp-size 2 --nnodes 2 --node-rank "$rank"
          --base-gpu-id "$base" --dist-init-addr "${HOST_228_IP}:29500" --port 30001
          --disaggregation-mode decode --disaggregation-bootstrap-port 8998
          --max-running-requests 64)
    ;;
  *) echo "Unknown role $role" >&2; exit 2 ;;
esac
exec python -m sglang.launch_server "${common[@]}" "${args[@]}" "$@"
