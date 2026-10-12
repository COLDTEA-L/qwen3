#!/usr/bin/env bash
# Default: inventory only. Explicit name: diagnostic exec on that RUNNING
# container, after confirming ownership/permission. Never start/stop/install.
# Usage: bash collect_container_info.sh [container-name] [CANN-set_env.sh]
set -uo pipefail

query() {
  local label="$1"
  shift
  printf '\n[%s]\n' "$label"
  if command -v timeout >/dev/null 2>&1; then timeout 30 "$@"; else "$@"; fi
  local status=$?
  [ "$status" -eq 0 ] || printf 'Query unavailable/failed (exit %s); NOT a compatibility verdict.\n' "$status"
  return 0
}

query Host hostname
query All-containers docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}'
query Images docker image ls --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}} {{.ID}} {{.Size}}'
if [ "$#" -eq 0 ]; then
  printf '\nSelect an owned/authorized RUNNING candidate and rerun with its exact name.\n'
  exit 0
fi
if [ "$#" -gt 2 ] || [[ "$1" == -* ]]; then
  printf 'Usage: bash collect_container_info.sh [container-name] [CANN-set_env.sh]\n' >&2
  exit 2
fi
candidate="$1"
cann_env="${2:-}"
container_id=$(docker container inspect --format '{{.Id}}' "$candidate") || exit 1
query Container-config docker container inspect "$container_id" --format 'Name={{.Name}} ImageTag={{.Config.Image}} ImageID={{.Image}} State={{.State.Status}} User={{.Config.User}} Network={{.HostConfig.NetworkMode}} IPC={{.HostConfig.IpcMode}} Privileged={{.HostConfig.Privileged}} Runtime={{.HostConfig.Runtime}}'
query Device-mapping docker container inspect "$container_id" --format '{{json .HostConfig.Devices}}'
query Device-requests docker container inspect "$container_id" --format '{{json .HostConfig.DeviceRequests}}'
query Mounts docker container inspect "$container_id" --format '{{range .Mounts}}{{println .Source "->" .Destination "RW=" .RW}}{{end}}'
image_id=$(docker container inspect "$container_id" --format '{{.Image}}') || exit 1
query Image-baseline docker image inspect "$image_id" --format 'ID={{.Id}} Architecture={{.Architecture}} OS={{.Os}} Tags={{json .RepoTags}} Digests={{json .RepoDigests}}'
state=$(docker container inspect "$container_id" --format '{{.State.Status}}') || exit 1
if [ "$state" != running ]; then
  printf '\nContainer is %s: metadata only. Original start command may launch a service; do not start blindly.\n' "$state"
  exit 0
fi

# No full env/inspect/command lines (may contain credentials), no login shell,
# no tensors, engines, pools, process groups or model load.
docker exec -i "$container_id" bash -s -- "$cann_env" <<'CONTAINER_PROBE'
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
query() {
  local label="$1"
  shift
  printf '\n[%s]\n' "$label"
  if command -v timeout >/dev/null 2>&1; then timeout 30 "$@"; else "$@"; fi
  local status=$?
  [ "$status" -eq 0 ] || printf 'Query unavailable/failed (exit %s); preserve output.\n' "$status"
  return 0
}
if [ -n "${1:-}" ]; then
  if [ -f "$1" ]; then
    # Explicitly selected, trusted, image-matching vendor environment script.
    set +u
    source "$1"
    set -u
  else
    printf 'CANN environment script does not exist: %s\n' "$1" >&2
    exit 1
  fi
fi
query CPU-architecture uname -m
query OS cat /etc/os-release
query CANN-env-paths find /usr/local/Ascend -maxdepth 4 -name set_env.sh -print
query Device-nodes ls -l /dev/davinci{0..7} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
query NPU-status npu-smi info
query SSD-mount ls -ld /ssd_2/yuanwenliu
query Shared-memory df -h /dev/shm
query CANN-version-files bash -c '
  find /usr/local/Ascend -maxdepth 4 -type f \( -name version.cfg -o -name ascend_toolkit_install.info \) -print -exec cat {} \;
  for f in /usr/local/Ascend/ascend-toolkit/latest/version.info /usr/local/Ascend/driver/version.info; do
    if [ -r "$f" ]; then printf "\n%s\n" "$f"; cat "$f"; fi
  done
'
for tool in python python3 gcc g++ cmake ccec; do
  query "Tool-${tool}" bash -c 'command -v "$1"' _ "$tool"
done
python_query=""
if command -v python >/dev/null 2>&1; then
  python_query=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
  python_query=$(command -v python3)
fi
if [ -z "$python_query" ]; then
  printf '\nNo Python found; report the candidate venv/conda activation method.\n'
  exit 0
fi
query Python-inventory "$python_query" - <<'PY'
import importlib
import importlib.metadata as md
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

print("PYTHON", sys.executable, sys.version.replace("\n", " "))
for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
    print("SELECTED_ENV", name, os.environ.get(name, "UNSET"))
print("PYTHON_HEADERS", (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file())
for name in ("torch", "torch-npu", "sglang", "sglang-router", "sgl-kernel-npu",
             "deep-ep", "deep-ep-npu", "memfabric-hybrid", "memcache-hybrid",
             "setuptools", "wheel", "pybind11"):
    try:
        print("PACKAGE", name, md.version(name))
    except md.PackageNotFoundError:
        print("PACKAGE_NOT_FOUND", name)
for name in ("torch", "torch_npu", "sglang", "sglang_router", "sgl_kernel_npu",
             "deep_ep", "memfabric_hybrid", "memcache_hybrid", "umdk_cam_op_lib"):
    try:
        spec = importlib.util.find_spec(name)
        print("MODULE", name, spec.origin if spec else "NOT_FOUND")
    except Exception as exc:
        print("MODULE_QUERY_FAILED", name, type(exc).__name__, str(exc))
try:
    import torch
    import torch_npu
    print("TORCH", torch.__version__, "TORCH_NPU", torch_npu.__version__)
    print("NPU_AVAILABLE", torch.npu.is_available(), "NPU_COUNT", torch.npu.device_count())
except Exception as exc:
    print("TORCH_NPU_QUERY_FAILED", type(exc).__name__, str(exc))
for name in ("sgl_kernel_npu", "deep_ep", "memfabric_hybrid", "memcache_hybrid"):
    try:
        module = importlib.import_module(name)
        print("IMPORT_OK", name, getattr(module, "__file__", None))
        if name == "memfabric_hybrid":
            engine = getattr(module, "TransferEngine", None)
            print("DEVICE_RDMA_API", hasattr(getattr(engine, "TransDataOpType", None), "DEVICE_RDMA"))
        if name == "memcache_hybrid":
            print("MEMCACHE_API", all(hasattr(module, n) for n in (
                "DistributedObjectStore", "LocalConfig", "MetaService", "MetaConfig")))
    except Exception as exc:
        print("IMPORT_FAILED", name, type(exc).__name__, str(exc))
print("INVENTORY_COMPLETE: not an A2/CAM/EP8/PD correctness certification")
PY
CONTAINER_PROBE
