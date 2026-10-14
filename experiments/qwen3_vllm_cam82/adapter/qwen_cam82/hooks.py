"""Exact TP group interception; native Qwen MoE transport is untouched."""
import contextvars
import functools
import logging

from .policy import selected_group

IN_MOE = contextvars.ContextVar("qwen_cam82_in_moe", default=False)
logger = logging.getLogger(__name__)


def install(runtime, tp_group):
    from vllm.distributed.parallel_state import GroupCoordinator, get_ep_group
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import FusedMoEState
    from vllm_ascend.models.qwen3_moe import CustomSparseMoeBlock
    if getattr(GroupCoordinator.all_reduce, "_cam82_hook", False):
        raise RuntimeError("CAM TP hooks already installed")
    for name in ("all_reduce", "reduce_scatter", "all_gather"):
        original = getattr(GroupCoordinator, name)
        def wrap(fn, operation):
            @functools.wraps(fn)
            def call(group, *args, **kwargs):
                if runtime.mode != "off" and selected_group(group, tp_group, IN_MOE.get()):
                    return getattr(runtime, operation)(*args, **kwargs)
                return fn(group, *args, **kwargs)
            call._cam82_hook = True
            return call
        setattr(GroupCoordinator, name, wrap(original, name))

    native = CustomSparseMoeBlock.forward
    seen = False
    @functools.wraps(native)
    def observed(self, *args, **kwargs):
        nonlocal seen
        context = get_forward_context()
        if context.fused_moe_state != FusedMoEState.All2All:
            raise RuntimeError("Native Qwen MoE is not All2All; refuse to label the run A2A")
        if type(context.token_dispatcher).__name__ != "TokenDispatcherWithAll2AllV":
            raise RuntimeError("Native Qwen token dispatcher is not All2AllV")
        if not seen:
            logger.warning("NATIVE_QWEN_MOE_ALL2ALL ep=%s (transport unchanged)", get_ep_group().world_size)
            seen = True
        token = IN_MOE.set(True)
        try:
            return native(self, *args, **kwargs)
        finally:
            IN_MOE.reset(token)
    CustomSparseMoeBlock.forward = observed
