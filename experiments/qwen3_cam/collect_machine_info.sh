#!/usr/bin/env bash
# Read-only inventory. Run on each A2 HOST, not on the cardless development host.
# Does not install packages, alter NICs, configure HCCN, or start/stop containers.
set -uo pipefail

query() {
  local label="$1"
  shift
  printf '\n[%s]\n' "$label"
  if command -v timeout >/dev/null 2>&1; then
    timeout 30 "$@"
  else
    "$@"
  fi
  local status=$?
  if [ "$status" -ne 0 ]; then
    printf 'Query unavailable/failed (exit %s); preserve this message in the report.\n' "$status"
  fi
  return 0
}

query Hostname hostname
query Identity id
query CPU-architecture uname -m
query OS cat /etc/os-release
query Host-memory free -h
query SSD-filesystem df -hT /ssd_2
query SSD-mount findmnt -T /ssd_2 -o TARGET,SOURCE,FSTYPE
query Host-shared-memory df -h /dev/shm
query Docker-client-server docker version
query Docker-storage docker info --format 'ServerVersion={{.ServerVersion}} Architecture={{.Architecture}} StorageDriver={{.Driver}} DockerRootDir={{.DockerRootDir}}'
query Docker-images docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'
query Running-containers docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
query NPU-tool-path bash -c 'command -v npu-smi'
query NPU-status npu-smi info
query NPU-identifiers npu-smi info -l
query NPU-topology npu-smi info -t topo
query NPU-board-0 npu-smi info -t board -i 0 -c 0
query Device-nodes ls -l /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
for filename in /usr/local/Ascend/driver/version.info /usr/local/Ascend/firmware/version.info; do
  if [ -r "$filename" ]; then
    query "$filename" cat "$filename"
  fi
done
query Ascend-install-files find /usr/local/Ascend -maxdepth 4 -type f \( \
  -name set_env.sh -o -name version.cfg -o -name version.info \
  -o -name ascend_toolkit_install.info \) -print
query Host-link ip -br link
query Host-IPv4 ip -br -4 addr
query Host-routes ip -4 route
query Listening-TCP ss -ltn

if [ -n "${PEER_HOST_IP:-}" ]; then
  query Route-to-peer ip route get "$PEER_HOST_IP"
fi
if [ -n "${HOST_IFNAME:-}" ]; then
  query Selected-link ip -details link show dev "$HOST_IFNAME"
  query Selected-NIC ethtool "$HOST_IFNAME"
  query Selected-NIC-driver ethtool -i "$HOST_IFNAME"
fi
if command -v rdma >/dev/null 2>&1; then
  query Host-RDMA-links rdma link show
fi

hccn_query=""
if command -v hccn_tool >/dev/null 2>&1; then
  hccn_query=$(command -v hccn_tool)
elif [ -x /usr/local/Ascend/driver/tools/hccn_tool ]; then
  hccn_query=/usr/local/Ascend/driver/tools/hccn_tool
elif [ -x /usr/local/Ascend/driver/tools/hccn_tool/hccn_tool ]; then
  hccn_query=/usr/local/Ascend/driver/tools/hccn_tool/hccn_tool
fi
if [ -n "$hccn_query" ]; then
  for device in {0..7}; do
    query "NPU-${device}-IP" "$hccn_query" -i "$device" -ip -g
    query "NPU-${device}-link" "$hccn_query" -i "$device" -link -g
    query "NPU-${device}-health" "$hccn_query" -i "$device" -net_health -g
  done
else
  printf '\n[HCCN] Tool not found; report the installed driver/tool location.\n'
fi
