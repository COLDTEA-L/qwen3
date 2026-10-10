#!/usr/bin/env bash
# New dedicated container only. Never stop, replace or commit an existing one.
set -euo pipefail
image="${1:?usage: bash create_container.sh IMAGE NEW_CONTAINER_NAME}"
container="${2:?Specify a new container name}"
docker image inspect "$image" >/dev/null
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container already exists: $container; inspect it, do not overwrite" >&2
  exit 2
fi
test -d /ssd_2/sglang
test -d /usr/local/Ascend/driver
test -d /usr/local/Ascend/firmware
npu_cli=$(command -v npu-smi || true)
test -n "$npu_cli" || { echo "Host npu-smi is not in PATH" >&2; exit 2; }
npu_cli=$(readlink -f "$npu_cli")
test -f "$npu_cli"
devices=()
for number in {0..7}; do
  test -e "/dev/davinci${number}"
  devices+=(--device "/dev/davinci${number}")
done
for device in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  test -e "$device"
  devices+=(--device "$device")
done
exec docker run -d --name "$container" --network host --ipc host --runtime runc \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK "${devices[@]}" \
  -v /ssd_2:/ssd_2 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v "$npu_cli:/usr/local/sbin/npu-smi:ro" \
  --entrypoint /bin/bash "$image" -lc 'exec sleep infinity'
