# shellcheck shell=bash
# Source this helper, do not execute it: SDK exports must reach the build shell.
# CANN 8.2's env scripts contain non-matching grep probes. Only while sourcing
# the SDK, disable errexit and the inherited ERR trap for ordinary-shell probe
# semantics (Bash versions differ on source inside an if). Restore strict mode
# and diagnostics before checking the final SDK status or running build steps.
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "[cam82_env] ASCEND_HOME_PATH is not set" >&2
    return 1
fi
if [ ! -r "${ASCEND_HOME_PATH}/bin/setenv.bash" ]; then
    echo "[cam82_env] missing readable ${ASCEND_HOME_PATH}/bin/setenv.bash" >&2
    return 1
fi
echo "[cam82_env] loading ${ASCEND_HOME_PATH}/bin/setenv.bash"
_cam82_saved_err_trap=$(trap -p ERR)
trap - ERR
set +e
_cam82_sdk_status=0
# shellcheck disable=SC1091
if source "${ASCEND_HOME_PATH}/bin/setenv.bash"; then
    :
else
    _cam82_sdk_status=$?
fi
set -e
if [ -n "${_cam82_saved_err_trap}" ]; then
    eval "${_cam82_saved_err_trap}"
fi
unset _cam82_saved_err_trap
if [ "${_cam82_sdk_status}" -ne 0 ]; then
    echo "[cam82_env] CANN environment returned ${_cam82_sdk_status}; stopping" >&2
    return "${_cam82_sdk_status}"
fi
unset _cam82_sdk_status

# Check useful postconditions, not merely that a source command returned zero.
if [ ! -r "${ASCEND_HOME_PATH:-}/include/acl/acl.h" ]; then
    echo "[cam82_env] missing ACL SDK header under ${ASCEND_HOME_PATH:-<unset>}" >&2
    return 1
fi
if [ ! -d "${ASCEND_OPP_PATH:-}" ]; then
    echo "[cam82_env] missing ASCEND_OPP_PATH directory: ${ASCEND_OPP_PATH:-<unset>}" >&2
    return 1
fi
for _cam82_sdk_tool in msopgen cmake; do
    if ! command -v "${_cam82_sdk_tool}" >/dev/null 2>&1; then
        echo "[cam82_env] missing build tool: ${_cam82_sdk_tool}" >&2
        return 1
    fi
done
unset _cam82_sdk_tool
# The SDK may reorder paths. Preserve the already verified toolkit-first order.
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:${ASCEND_HOME_PATH}/aarch64-linux/devlib"
echo CAM82_CANN_ENV_READY
