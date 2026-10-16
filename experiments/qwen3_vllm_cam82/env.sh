#!/usr/bin/env bash
if [ "${BASH_SOURCE[0]}" = "$0" ]; then echo 'Source this file with 228|238' >&2; exit 2; fi
case "${1:-}" in 228|238) ;; *) echo 'Specify 228|238' >&2; return 2 ;; esac
unset PYTHONPATH ASCEND_RT_VISIBLE_DEVICES RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset CAM_COMM_ID MOONCAKE_CONFIG_PATH ASCEND_MF_STORE_URL SGLANG_CAM_DETOUR_MODE QWEN_CAM_MODE
unset PYTORCH_NPU_ALLOC_CONF ASCEND_ENABLE_USE_FABRIC_MEM ASCEND_BUFFER_POOL
source /usr/local/Ascend/ascend-toolkit/set_env.sh || return
source /usr/local/Ascend/nnal/atb/set_env.sh || return
export QWEN_CAM82_HOST="$1" VLLM_HOST_IP="10.155.205.$1" HCCL_IF_IP="10.155.205.$1"
export GLOO_SOCKET_IFNAME=enp67s0f5 HCCL_SOCKET_IFNAME=enp67s0f5 TP_SOCKET_IFNAME=enp67s0f5
# The CAM wrapper dlopens libcust_opapi.so by basename, not an absolute path.
# Register both its API library and operator package in every new shell.
export CAM82_OPP_DIR="${ASCEND_OPP_PATH}/vendors/CAM"
if [ ! -r "${CAM82_OPP_DIR}/op_api/lib/libcust_opapi.so" ]; then
  echo "Missing CAM operator API: ${CAM82_OPP_DIR}/op_api/lib/libcust_opapi.so" >&2
  return 2
fi
export ASCEND_CUSTOM_OPP_PATH="${CAM82_OPP_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
# Keep matched toolkit runtimes before SDK stubs and other CANN copies.
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${CAM82_OPP_DIR}/op_api/lib:/usr/local/lib:/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODEL_PATH=/ssd_2/yuanwenliu/models/Qwen3-30B-A3B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1
export VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP=0 VLLM_ASCEND_ENABLE_PREFETCH_MLP=0 VLLM_ASCEND_ENABLE_FLASHCOMM=0
export PHYSICAL_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 OMP_PROC_BIND=false PYTHONHASHSEED=0
export TASK_QUEUE_ENABLE=1 HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600
# A lone ROCE=0 can fail HCCL environment validation. Use topology defaults;
# these switches do not prove which physical link a collective actually uses.
unset HCCL_INTRA_PCIE_ENABLE HCCL_INTRA_ROCE_ENABLE
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy VLLM_USE_MODELSCOPE
export NO_PROXY=localhost,127.0.0.1,10.155.205.228,10.155.205.238
export no_proxy="$NO_PROXY"
cd /ssd_2/yuanwenliu/sglang || return
mkdir -p experiments/qwen3_vllm_cam82/results experiments/qwen3_vllm_cam82/profiles
echo "CAM82_ENV_READY host=$1 (all8 physically visible, inference offline)"
