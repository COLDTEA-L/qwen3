from vllm_ascend.worker.worker_v1 import NPUWorker
from .policy import ROLES, check_installation, physical_device, validate_config
from .startup import prepare_environment
from .native_kv import install as install_native_kv, validate_config as validate_kv_config
from .aligned_kv import install as install_aligned_kv


class QwenCamWorker(NPUWorker):
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method,
                 is_driver_worker=False, **kwargs):
        check_installation()
        validate_config(vllm_config)
        validate_kv_config(vllm_config)
        # Still reject a masked worker; the parent must prevent DP masking
        # before spawning it, not unset the variable after importing NPU code.
        role, mode = prepare_environment()
        self.cam_role, self.cam_mode = role, mode
        self.cam_physical_device = physical_device(role, local_rank)
        if vllm_config.parallel_config.data_parallel_size_local != 1:
            raise ValueError("One local DP instance per engine is required")
        # This old connector indexes PHYSICAL_DEVICES by LOCAL DP rank, not the
        # global DP rank. Both externally balanced D engines have local rank0.
        vllm_config.parallel_config.data_parallel_rank_local = 0
        # Leave all eight NPUs visible to CAM. Adjust worker local device without
        # altering global TP/DP ranks; no inference worker is started on relays.
        super().__init__(vllm_config, self.cam_physical_device, rank,
                         distributed_init_method, is_driver_worker, **kwargs)

    def init_device(self):
        import torch
        if torch.npu.device_count() != 8:
            raise RuntimeError("CAM82 requires all8 physical NPUs in each worker")
        super().init_device()
        expected = self.cam_physical_device
        actual = torch.npu.current_device()
        if actual != expected:
            raise RuntimeError(f"CAM82 wrong NPU mapping: expected physical={expected}, actual={actual}")
        print(f"CAM82_WORKER_DEVICE role={self.cam_role} mode={self.cam_mode} "
              f"physical={actual} all8_visible=True", flush=True)
        install_aligned_kv(self.model_runner)
        from vllm.distributed.kv_transfer import get_kv_transfer_group
        install_native_kv(get_kv_transfer_group())
        from vllm.distributed.parallel_state import get_tp_group, get_ep_group
        from .collectives import CamTP
        from .hooks import install
        expected_ep = 4 if self.cam_role == "p-228" else 8
        if get_ep_group().world_size != expected_ep:
            raise RuntimeError("EP membership does not match P4/D8 plan")
        self.cam_runtime = CamTP(get_tp_group(), ROLES[self.cam_role][1], self.cam_mode)
        install(self.cam_runtime, get_tp_group())
        from .moe_trace import install as install_moe_trace
        install_moe_trace()
