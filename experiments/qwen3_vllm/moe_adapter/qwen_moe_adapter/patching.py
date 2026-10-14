"""Worker-local, opt-in adaptation of pinned upstream AlltoAllCommImpl.

No framework source files are edited. No fallback masks an A2A failure.
AllGather/MC2 classes and the KV transfer engine are left untouched.
"""

import functools
import logging
import sys

from .policy import check_installation, make_selector, padded_tokens

logger = logging.getLogger(__name__)


def make_prepare_class(native_class, torch):
    class PaddedAllToAll(native_class):
        def prepare(self, hidden_states, router_logits, enable_shared_expert_dp=False,
                    replace_allreduce=False, quant_type=None):
            if enable_shared_expert_dp or replace_allreduce:
                raise ValueError("Shared-expert/SP bypass is not supported by this experiment")
            original = hidden_states.shape[0]
            if hidden_states.ndim != 2 or router_logits.shape[0] != original:
                raise ValueError("Unexpected MoE token layout")
            padded = padded_tokens(original, self.tp_size)
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padded - original))
            router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, padded - original))
            prepared = super().prepare(hidden_states, router_logits, False, False, quant_type)
            # Native finalize trims using num_tokens. Preserve the real count.
            self.num_tokens = original
            self._qwen_padded_tokens = padded
            return prepared

        def pad_and_split_input_ids(self, input_ids):
            if input_ids.shape[0] != self.num_tokens:
                raise ValueError("Input-ID/token count mismatch")
            padded = torch.nn.functional.pad(input_ids, (0, self._qwen_padded_tokens - self.num_tokens))
            return torch.tensor_split(padded, self.tp_size, dim=0)[self.tp_rank]
    return PaddedAllToAll


def install():
    import vllm_ascend.ascend_forward_context as context

    if getattr(context.select_moe_comm_method, "_qwen_original", None) is not None:
        return
    root = check_installation()
    import torch
    import vllm_ascend.ops.fused_moe.moe_comm_method as methods
    from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if get_ascend_device_type() != AscendDeviceType.A2:
        raise RuntimeError("This opt-in worker experiment is only for A2")
    original = context.select_moe_comm_method
    selected = make_selector(original, context.MoECommType.ALLTOALL, context)
    context.select_moe_comm_method = selected
    # These modules may have captured the original function before the worker
    # constructor ran. Update only known aliases, not arbitrary loaded modules.
    for name in ("vllm_ascend.worker.model_runner_v1", "vllm_ascend.worker.v2.model_runner"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "select_moe_comm_method", None) is original:
            module.select_moe_comm_method = selected
    prepared_class = make_prepare_class(PrepareAndFinalizeWithAll2All, torch)
    methods.AlltoAllCommImpl._get_prepare_finalize = lambda self: prepared_class(self.moe_config)
    # These markers enclose the upstream All-to-All dispatcher, including its
    # metadata gathers. They are host ranges, not proof of device overlap.
    counters = {"dispatch": 0, "combine": 0}
    for name, phase in (("token_dispatch", "dispatch"), ("token_combine", "combine")):
        original_method = getattr(TokenDispatcherWithAll2AllV, name)
        def wrap(method, label):
            @functools.wraps(method)
            def traced(self, *args, **kwargs):
                with torch.profiler.record_function("QWEN_MOE_A2A::" + label):
                    result = method(self, *args, **kwargs)
                counters[label] += 1
                if counters[label] <= 2:
                    logger.warning("QWEN_MOE_A2A_%s ep=%s call=%s", label.upper(), self.ep_size, counters[label])
                return result
            return traced
        setattr(TokenDispatcherWithAll2AllV, name, wrap(original_method, phase))
    logger.warning("QWEN_MOE_A2A_INSTALLED source=%s; experimental BF16 TP4 EP4/8; no fallback", root)
