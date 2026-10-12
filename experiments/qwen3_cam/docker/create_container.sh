#!/usr/bin/env bash
# New dedicated container only. Follows the user's known A2 launch profile.
# --privileged is intentional for this experiment, NOT production isolation.
build_container_command() {
  local image="$1" container="$2" npu_cli="$3" firmware="${4:-}"
  local number device
  container_command=(docker run -itd --name "$container" --privileged
    --network host --ipc host --runtime ascend
    --ulimit memlock=-1:-1 -e ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
  for number in {0..7}; do
    container_command+=(--device "/dev/davinci${number}")
  done
  for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    container_command+=(--device "$device")
  done
  container_command+=(
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v "$npu_cli:/usr/local/bin/npu-smi:ro"
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /var/log/npu:/usr/slog:rw
    -v /ssd_2/sglang:/ssd_2/sglang:rw
    -v /ssd_2/umdk:/ssd_2/umdk:rw
    -v /ssd_2/models/Qwen3-30B-A3B:/ssd_2/models/Qwen3-30B-A3B:rw
    -v /ssd_2/qwen3-download-venv:/ssd_2/qwen3-download-venv:rw)
  if [ -n "$firmware" ]; then
    container_command+=(-v "$firmware:/usr/local/Ascend/firmware:ro")
  fi
  container_command+=(--entrypoint /bin/bash "$image" -lc 'exec sleep infinity')
}
# Allow CPU tests to inspect argument construction without invoking Docker.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi
set -euo pipefail
image="${1:?usage: bash create_container.sh IMAGE NEW_CONTAINER_NAME}"
container="${2:?Specify a new container name}"
action="${3:-run}"
case "$action" in run|--print-only) ;; *) echo "Optional third argument: --print-only" >&2; exit 2 ;; esac
runtime_info=$(docker info --format '{{json .Runtimes}}')
case "$runtime_info" in
  *'"ascend"'*) ;;
  *) echo "Docker runtime 'ascend' is not registered; ask the host administrator" >&2; exit 2 ;;
esac
docker image inspect "$image" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container already exists: $container; inspect it, do not overwrite" >&2
  exit 2
fi
for directory in /ssd_2/sglang /ssd_2/umdk /ssd_2/models/Qwen3-30B-A3B \
  /ssd_2/qwen3-download-venv /usr/local/Ascend/driver /usr/local/dcmi /var/log/npu; do
  test -d "$directory" || { echo "Required host directory missing: $directory" >&2; exit 2; }
done
if [ -f /usr/local/bin/npu-smi ]; then
  npu_cli=/usr/local/bin/npu-smi
else
  npu_cli=$(command -v npu-smi || true)
fi
test -n "$npu_cli" || { echo "Host npu-smi is not in PATH" >&2; exit 2; }
npu_cli=$(readlink -f "$npu_cli")
test -f "$npu_cli"
for number in {0..7}; do
  test -e "/dev/davinci${number}"
done
for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  test -e "$device"
done
firmware=""
if [ -d /usr/local/Ascend/firmware ]; then firmware=/usr/local/Ascend/firmware; fi
build_container_command "$image" "$container" "$npu_cli" "$firmware"
if [ "$action" = --print-only ]; then
  printf '%q ' "${container_command[@]}"
  printf '\n'
  exit 0
fi
exec "${container_command[@]}"
