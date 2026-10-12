#!/usr/bin/env bash
# Dedicated container only; matching CANN/Python/Torch ABI is required.
set -euo pipefail
action="${1:?usage: bash install_cam.sh build|install}"
root="${UMDK_PATH:?Source the current env.sh first}"
soc="${SOC_VERSION:?Source env.sh first}"
case "$action" in
  build) bash "$root/build/cam/comm_operator/build_qwen3_all2all_detour.sh" "$soc" ;;
  install)
    shopt -s nullglob
    packages=("$root/output/cam/comm_operator/run/CAM_${soc}_all_to_all_detour_"*.run)
    test "${#packages[@]}" -eq 1 || { echo "Expected one CAM .run; inspect generated output" >&2; exit 2; }
    bash "${packages[0]}" --install
    ;;
  *) echo "Action must be build or install" >&2; exit 2 ;;
esac
shopt -s nullglob
wheels=("$root/output/cam/qwen3-all2all/wheel/"*.whl)
test "${#wheels[@]}" -eq 1 || { echo "Expected one generated CAM wheel; inspect output" >&2; exit 2; }
python -m pip install --no-deps --force-reinstall "${wheels[0]}"
python -c 'import umdk_cam_op_lib as c; assert hasattr(c,"cam_get_comm") and hasattr(c,"all2_all_detour"); print("CAM_API_OK", c.__file__)'
