import os
from vllm_ascend.worker.worker_v1 import NPUWorker
from .policy import ROLES, MODES, check_installation, physical_device, validate_config


class QwenCamWorker(NPUWorker):
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method,
                 is_driver_worker=False, **kwargs):
        check_installation()
        validate_config(vllm_config)
        role = os.environ.get("QWEN_CAM82_ROLE", "")
        mode = os.environ.get("QWEN_CAM82_MODE", "off")
        if mode not in MODES or role not in ROLES or os.getenv("ASCEND_RT_VISIBLE_DEVICES"):
            raise ValueError("Explicit role/mode and full physical visibility required")
        self.cam_role, self.cam_mode = role, mode
        if vllm_config.parallel_config.data_parallel_size_local != 1:
            raise ValueError("One local DP instance per engine is required")
        # This old connector indexes PHYSICAL_DEVICES by LOCAL DP rank, not the
        # global DP rank. Both externally balanced D engines have local rank0.
        vllm_config.parallel_config.data_parallel_rank_local = 0
        # Leave all eight NPUs visible to CAM. Adjust worker local device without
        # altering global TP/DP ranks; no inference worker is started on relays.
        super().__init__(vllm_config, physical_device(role, local_rank), rank,
                         distributed_init_method, is_driver_worker, **kwargs)

    def init_device(self):
        super().init_device()
        from vllm.distributed.parallel_state import get_tp_group, get_ep_group
        from .collectives import CamTP
        from .hooks import install
        expected_ep = 4 if self.cam_role == "p-228" else 8
        if get_ep_group().world_size != expected_ep:
            raise RuntimeError("EP membership does not match P4/D8 plan")
        self.cam_runtime = CamTP(get_tp_group(), ROLES[self.cam_role][1], self.cam_mode)
        install(self.cam_runtime, get_tp_group())
