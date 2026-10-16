"""Use the native vLLM CLI after applying this experiment's parent-only policy."""
import runpy

from .policy import check_installation
from .startup import install_dp_visibility_policy, prepare_environment


def main():
    role, mode = prepare_environment()
    check_installation()
    from vllm.v1.engine import utils as engine_utils
    install_dp_visibility_policy(engine_utils)
    print(f"CAM82_STARTUP_READY role={role} mode={mode} HCCL_intra_overrides=unset", flush=True)
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
