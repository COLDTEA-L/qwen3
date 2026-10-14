#!/usr/bin/env bash
# Only creates a new container; never replaces/stops existing containers.
build_command() {
  local image="$1" container="$2" npu_cli="$3" number
  container_command=(docker run -itd --name "$container" --privileged --network host
    --ipc host --runtime ascend --ulimit memlock=-1:-1
    --workdir /ssd_2/yuanwenliu/sglang -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
  for number in {0..7}; do container_command+=(--device "/dev/davinci${number}"); done
  container_command+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro -v "$npu_cli:/usr/local/bin/npu-smi:ro"
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro -v /etc/hccn.conf:/etc/hccn.conf:ro
    -v /var/log/npu:/usr/slog:rw
    -v /ssd_2/yuanwenliu/sglang:/ssd_2/yuanwenliu/sglang:rw
    -v /ssd_2/yuanwenliu/vllm-ascend-qwen3:/ssd_2/yuanwenliu/vllm-ascend-qwen3:ro
    -v /ssd_2/yuanwenliu/models/Qwen3-30B-A3B:/ssd_2/yuanwenliu/models/Qwen3-30B-A3B:ro)
  if [ -d /usr/local/Ascend/firmware ]; then
    container_command+=(-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro)
  fi
  container_command+=(--entrypoint /bin/bash "$image" -lc 'exec sleep infinity')
}
if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi
set -euo pipefail
image="${1:?Specify local baseline image}"; container="${2:?Specify new container name}"
case "${3:-}" in ""|--print-only) ;; *) echo "Only --print-only is supported" >&2; exit 2 ;; esac
docker image inspect "$image" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container exists; do not overwrite: $container" >&2; exit 2
fi
runtime_info=$(docker info --format '{{json .Runtimes}}')
case "$runtime_info" in *'"ascend"'*) ;; *) echo "Ascend runtime missing" >&2; exit 2 ;; esac
for path in /etc/hccn.conf /etc/ascend_install.info /usr/local/Ascend/driver /usr/local/dcmi /var/log/npu \
  /ssd_2/yuanwenliu/sglang /ssd_2/yuanwenliu/vllm-ascend-qwen3 /ssd_2/yuanwenliu/models/Qwen3-30B-A3B; do
  test -e "$path" || { echo "Required mount missing: $path" >&2; exit 2; }
done
npu_cli=$(command -v npu-smi || true)
test -n "$npu_cli" || { echo "Host npu-smi missing" >&2; exit 2; }
npu_cli=$(readlink -f "$npu_cli")
for number in {0..7}; do test -e "/dev/davinci${number}"; done
for path in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do test -e "$path"; done
build_command "$image" "$container" "$npu_cli"
if [ "${3:-}" = --print-only ]; then printf '%q ' "${container_command[@]}"; printf '\n'; exit 0; fi
exec "${container_command[@]}"
