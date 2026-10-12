#!/usr/bin/env bash
# Source once in each dedicated experiment terminal; do not run as a subprocess.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Use: source /ssd_2/yuanwenliu/sglang/experiments/qwen3_cam/env.sh 228|238" >&2
  exit 2
fi
_qwen_role="${1:-}"
case "$_qwen_role" in 228|238) ;; *) echo "Specify 228 or 238" >&2; return 2 ;; esac
_qwen_exp_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
test -f "$_qwen_exp_dir/runtime-config/deployment.env" || { echo "Run configure_runtime.py first" >&2; return 2; }
source "$_qwen_exp_dir/runtime-config/deployment.env" || return
if [ -z "${HOST_228_IP:-}" ] || [ -z "${HOST_238_IP:-}" ] || [ -z "${UMDK_PATH:-}" ] || \
  [ -z "${HOST_IF_228:-}" ] || [ -z "${HOST_IF_238:-}" ]; then
  echo "Old/incomplete deployment.env; regenerate with the current configure_runtime.py" >&2
  return 2
fi
_qwen_cann=""
for _qwen_path in /usr/local/Ascend/cann-9.0.0/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  if [ -f "$_qwen_path" ]; then _qwen_cann="$_qwen_path"; break; fi
done
test -n "$_qwen_cann" || { echo "Matching CANN9.0 set_env.sh missing" >&2; return 2; }
source "$_qwen_cann" || return
for _qwen_vendor in customize custom_transformer CAM; do
  _qwen_path="${ASCEND_OPP_PATH:-}/vendors/$_qwen_vendor/bin/set_env.bash"
  if [ -f "$_qwen_path" ]; then source "$_qwen_path" || return; fi
done
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT CAM_COMM_ID
unset ASCEND_RT_VISIBLE_DEVICES
export PYTHONPATH="$_qwen_exp_dir/../../python${PYTHONPATH:+:$PYTHONPATH}"
if [ "$_qwen_role" = 228 ]; then export HCCL_SOCKET_IFNAME="$HOST_IF_228"; else export HCCL_SOCKET_IFNAME="$HOST_IF_238"; fi
export GLOO_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME"
export ASCEND_MF_STORE_URL="tcp://${HOST_228_IP}:7100"
export ASCEND_MF_TRANSFER_PROTOCOL=device_rdma
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0
export SOC_VERSION=ascend910b4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_CAM_DETOUR_MODE="${SGLANG_CAM_DETOUR_MODE:-off}"
mkdir -p "$_qwen_exp_dir/results" "$_qwen_exp_dir/profiles"
cd "$_qwen_exp_dir/../.." || return
echo "QWEN_ENV_READY host=$_qwen_role cann=$_qwen_cann"
unset _qwen_role _qwen_exp_dir _qwen_cann _qwen_path _qwen_vendor
