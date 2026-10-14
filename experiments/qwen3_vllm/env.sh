#!/usr/bin/env bash
# Source in a NEW dedicated vLLM container, not the existing SGLang container.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Use: source experiments/qwen3_vllm/env.sh 228|238" >&2
  exit 2
fi
_qv_host="${1:-}"
case "$_qv_host" in 228|238) ;; *) echo "Specify 228 or 238" >&2; return 2 ;; esac
_qv_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test -f "$_qv_dir/runtime-config/deployment.env" || { echo "Run runtime.py configure first" >&2; return 2; }
source "$_qv_dir/runtime-config/deployment.env" || return
# Remove application injection and old PD configuration before restoring CANN.
unset PYTHONPATH RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset ASCEND_MF_STORE_URL ASCEND_MF_TRANSFER_PROTOCOL ASCEND_RT_VISIBLE_DEVICES
unset CAM_COMM_ID SGLANG_CAM_DETOUR_MODE SGLANG_CAM_DETOUR_CONFIG
unset MOONCAKE_CONFIG_PATH MOONCAKE_GLOBAL_SEGMENT_SIZE ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_BUFFER_POOL
_qv_cann=""
for _qv_file in /usr/local/Ascend/cann-9.0.0/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  if [ -f "$_qv_file" ]; then _qv_cann="$_qv_file"; break; fi
done
test -n "$_qv_cann" || { echo "CANN set_env.sh missing" >&2; return 2; }
source "$_qv_cann" || return
test -f /usr/local/Ascend/nnal/atb/set_env.sh || { echo "NNAL/ATB environment missing" >&2; return 2; }
source /usr/local/Ascend/nnal/atb/set_env.sh || return
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QWEN_VLLM_HOST="$_qv_host"
export VLLM_HOST_IP="10.155.205.$_qv_host" HCCL_IF_IP="10.155.205.$_qv_host"
export GLOO_SOCKET_IFNAME=enp67s0f5 HCCL_SOCKET_IFNAME=enp67s0f5 TP_SOCKET_IFNAME=enp67s0f5
export HCCL_INTRA_ROCE_ENABLE=1
export PHYSICAL_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=1 OMP_PROC_BIND=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 VLLM_ASCEND_ENABLE_FUSED_MC2=0
export QWEN_CAM_MODE=off
unset VLLM_USE_MODELSCOPE
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1,::1,10.155.205.228,10.155.205.238
export no_proxy="$NO_PROXY"
mkdir -p "$_qv_dir/results" "$_qv_dir/profiles"
cd "$_qv_dir/../.." || return
echo "QWEN_VLLM_ENV_READY host=$_qv_host cann=$_qv_cann (native baseline; CAM off)"
unset _qv_host _qv_dir _qv_cann _qv_file
