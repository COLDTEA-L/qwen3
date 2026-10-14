"""Explicit --worker-cls entry point; imported only for --moe-comm alltoall."""

import logging

from vllm_ascend.worker.worker import NPUWorker

from .patching import install
from .policy import validate_config

logger = logging.getLogger(__name__)


class QwenAllToAllWorker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        validate_config(vllm_config)
        install()
        super().__init__(vllm_config, *args, **kwargs)

    def init_device(self):
        super().init_device()
        from vllm.distributed.parallel_state import get_ep_group, get_tp_group
        import vllm_ascend.ascend_forward_context as context
        selected = context.select_moe_comm_method(1, self.vllm_config)
        if selected != context.MoECommType.ALLTOALL:
            raise RuntimeError("Requested MoE All-to-All is not selected")
        logger.warning("QWEN_MOE_POLICY_SELECTED=ALLTOALL rank=%s ep_ranks=%s tp_ranks=%s",
                       self.rank, get_ep_group().ranks, get_tp_group().ranks)
