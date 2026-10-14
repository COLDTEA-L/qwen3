#!/usr/bin/env bash
# New dedicated container only. No broad /var or /home mounts.
set -euo pipefail
image="${1:?Specify built CAM82 image}"; container="${2:?Specify new container name}"
docker image inspect "$image" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container exists; do not replace it" >&2; exit 2
fi
for task_path in /ssd_2/yuanwenliu/sglang /ssd_2/yuanwenliu/vllm-ascend-cam82 \
  /ssd_2/yuanwenliu/models/Qwen3-30B-A3B /etc/hccn.conf /etc/ascend_install.info \
  /usr/local/Ascend/driver /usr/local/dcmi /var/log/npu; do
  test -e "$task_path" || { echo "Missing mount: $task_path" >&2; exit 2; }
done
npu_cli=$(command -v npu-smi); npu_cli=$(readlink -f "$npu_cli")
cmd=(docker run -itd --name "$container" --privileged --runtime ascend --network host --ipc host
  --ulimit memlock=-1:-1 --workdir /ssd_2/yuanwenliu/sglang
  -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
for task_npu in {0..7}; do test -e "/dev/davinci${task_npu}"; cmd+=(--device "/dev/davinci${task_npu}"); done
cmd+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v "$npu_cli:/usr/local/bin/npu-smi:ro"
  -v /var/log/npu:/usr/slog:rw -v /etc/hccn.conf:/etc/hccn.conf:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /ssd_2/yuanwenliu/sglang:/ssd_2/yuanwenliu/sglang:rw
  -v /ssd_2/yuanwenliu/vllm-ascend-cam82:/ssd_2/yuanwenliu/vllm-ascend-cam82:ro
  -v /ssd_2/yuanwenliu/models/Qwen3-30B-A3B:/ssd_2/yuanwenliu/models/Qwen3-30B-A3B:ro)
cmd+=(--entrypoint /bin/bash "$image" -lc 'exec sleep infinity')
case "${3:-}" in
  --print-only) printf '%q ' "${cmd[@]}"; printf '\n' ;;
  "") exec "${cmd[@]}" ;;
  *) echo 'Only --print-only is supported' >&2; exit 2 ;;
esac
