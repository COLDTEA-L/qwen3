"""Scoped startup policy: keep physical IDs before vLLM spawns a DP engine."""
import contextlib
import functools
import inspect
import os

from .policy import MODES, ROLES, validate_config

WORKER_CLASS = "qwen_cam82.worker.QwenCamWorker"


def prepare_environment():
    role = os.environ.get("QWEN_CAM82_ROLE", "")
    mode = os.environ.get("QWEN_CAM82_MODE", "")
    if role not in ROLES or mode not in MODES:
        raise ValueError(f"Explicit CAM82 role/mode required: role={role!r}, mode={mode!r}")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible:
        raise ValueError(f"CAM82 requires all8 physical NPUs before process creation; visible={visible!r}")
    expected = ",".join(str(ROLES[role][1] + i) for i in range(4))
    if os.environ.get("PHYSICAL_DEVICES") != expected:
        raise ValueError(f"PHYSICAL_DEVICES must be {expected!r} for {role}")
    # Run before importing the CLI/Torch, and again in each custom worker.
    # This experiment uses topology defaults consistently in all TP modes.
    os.environ.pop("HCCL_INTRA_PCIE_ENABLE", None)
    os.environ.pop("HCCL_INTRA_ROCE_ENABLE", None)
    return role, mode


def install_dp_visibility_policy(engine_utils):
    """Replace only our worker's DP mask context, never patch installed files.

    The context is entered by CoreEngineProcManager in the API parent BEFORE
    proc.start(). Unsetting the mask later inside a worker is unsafe because
    Torch/CANN may already have cached the logical-to-physical mapping.
    """
    native = engine_utils.set_device_control_env_var
    if getattr(native, "_cam82_visibility_policy", False):
        return
    if tuple(inspect.signature(native).parameters) != ("vllm_config", "local_dp_rank"):
        raise RuntimeError("Unsupported vLLM DP mask context signature; do not bypass")
    manager_source = inspect.getsource(engine_utils.CoreEngineProcManager.__init__)
    context_source = inspect.getsource(native)
    if ("set_device_control_env_var" not in manager_source
            or "get_device_indices" not in context_source
            or engine_utils.current_platform.device_control_env_var != "ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("Unsupported vLLM DP visibility implementation; do not bypass")

    @contextlib.contextmanager
    @functools.wraps(native)
    def keep_physical_visibility(vllm_config, local_dp_rank):
        p = vllm_config.parallel_config
        if getattr(p, "worker_cls", None) != WORKER_CLASS:
            with native(vllm_config, local_dp_rank):
                yield
            return
        role, mode = prepare_environment()
        validate_config(vllm_config)
        expected_rank = 0 if role == "d8-228" else 1
        if (role not in ("d8-228", "d8-238") or p.data_parallel_size != 2
                or p.data_parallel_size_local != 1 or local_dp_rank != 0
                or p.world_size != 4 or p.data_parallel_rank != expected_rank
                or p.data_parallel_backend != "mp"
                or p.distributed_executor_backend != "mp"):
            raise RuntimeError("CAM82 DP visibility policy requires the planned D8/TP4/localDP1 layout")
        print(f"CAM82_DP_VISIBILITY role={role} mode={mode} local_dp_rank=0 "
              "all8_visible=True (physical mapping remains in custom worker)", flush=True)
        yield

    keep_physical_visibility._cam82_visibility_policy = True
    engine_utils.set_device_control_env_var = keep_physical_visibility

