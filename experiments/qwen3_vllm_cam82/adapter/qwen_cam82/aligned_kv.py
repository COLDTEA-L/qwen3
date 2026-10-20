"""Explicit aligned Qwen GQA allocation; keep native bind/register/transport.

Install only on this experiment's pinned ModelRunner instance. Do not allocate
the original full cache first, change global Torch allocation, or round memory
registration pointers into memory owned by another tensor.
"""
import inspect
import math
from types import MethodType

from .native_kv import (ALIGNMENT, BLOCK_BYTES, BLOCK_SIZE, NUM_BLOCKS,
                        NUM_LAYERS, validate_config, validate_regions)


def allocate_aligned(torch, shape, dtype, device):
    """A BF16 view and its owner; worst-case reserve is 2MiB per region."""
    if str(dtype) not in ("torch.bfloat16", "bfloat16"):
        raise RuntimeError("CAM82 aligned KV requires BF16")
    count = math.prod(shape)
    if count <= 0:
        raise RuntimeError("CAM82 aligned KV requires a nonempty shape")
    owner = torch.zeros(count + ALIGNMENT // 2, dtype=dtype, device=device)
    element_bytes = owner.element_size()
    ptr = owner.data_ptr()
    if element_bytes != 2 or ptr <= 0 or ptr % element_bytes:
        raise RuntimeError("CAM82 aligned KV invalid backing tensor")
    offset = ((-ptr) % ALIGNMENT) // element_bytes
    view = owner[offset:offset + count].view(shape)
    if (view.data_ptr() % ALIGNMENT or view.numel() != count
            or view.data_ptr() < ptr
            or view.data_ptr() + count * element_bytes > ptr + owner.numel() * element_bytes
            or not view.is_contiguous()):
        raise RuntimeError("CAM82 aligned KV view is outside its backing or not aligned")
    return view, owner


def initialize(runner, config, torch, full_attention_spec, bind):
    """Narrow counterpart to the pinned ordinary-attention allocation method."""
    validate_config(runner.vllm_config)
    if (runner.model_config.is_deepseek_mla
            or getattr(runner.model_config, "enable_sleep_mode", False)
            or runner.vllm_config.kv_transfer_config is None
            or runner.vllm_config.additional_config.get("kv_cache_dtype") == "int8"
            or config.num_blocks != NUM_BLOCKS or runner.kv_caches
            or runner.runner_only_attn_layers or hasattr(runner, "_cam82_kv_backings")):
        raise RuntimeError("CAM82 aligned KV requires fresh Qwen GQA/BF16/PD; no sleep or runner-only layers")

    sizes = {}
    for tensor in config.kv_cache_tensors:
        if len(tensor.shared_by) != 1 or tensor.shared_by[0] in sizes:
            raise RuntimeError("CAM82 aligned KV does not support shared/duplicate layer storage")
        sizes[tensor.shared_by[0]] = tensor.size
    groups = list(runner._kv_cache_spec_attn_group_iterator())
    shapes = {}
    for spec, group in groups:
        if (not isinstance(spec, full_attention_spec) or spec.block_size != BLOCK_SIZE
                or spec.num_kv_heads != 1 or spec.head_size != 128
                or str(spec.dtype) not in ("torch.bfloat16", "bfloat16")
                or spec.page_size_bytes != 2 * BLOCK_BYTES
                or group.backend.get_supported_block_size() != [BLOCK_SIZE]):
            raise RuntimeError("CAM82 aligned KV unsupported attention spec/backend")
        shape = tuple(group.backend.get_kv_cache_shape(NUM_BLOCKS, BLOCK_SIZE, 1, 128))
        if shape != (2, NUM_BLOCKS, BLOCK_SIZE, 1, 128):
            raise RuntimeError("CAM82 aligned KV unexpected native backend shape")
        for layer in group.layer_names:
            if layer in shapes or sizes.get(layer) != 2 * NUM_BLOCKS * BLOCK_BYTES:
                raise RuntimeError("CAM82 aligned KV layer storage size/membership mismatch")
            shapes[layer] = (shape[1:], spec.dtype)
    if len(shapes) != NUM_LAYERS or set(shapes) != set(sizes):
        raise RuntimeError("CAM82 aligned KV expects exactly 48 unique Qwen attention layers")
    if not set(shapes).issubset(runner.compilation_config.static_forward_context):
        raise RuntimeError("CAM82 aligned KV attention binding target missing")

    caches, owners = {}, {}
    for layer, (shape, dtype) in shapes.items():
        key, key_owner = allocate_aligned(torch, shape, dtype, runner.device)
        value, value_owner = allocate_aligned(torch, shape, dtype, runner.device)
        caches[layer] = (key, value)
        owners[layer] = (key_owner, value_owner)
    count = validate_regions(caches)
    # No format cast, stack, clone or copy after alignment: any can change the
    # allocation. Plain BF16 is the pinned A2 ND representation, as in the
    # original aligned MLA branch; all consumers accept a (K,V) pair.
    bind(caches, runner.compilation_config.static_forward_context, runner.kv_caches)
    runner._cam82_kv_backings = owners  # Explicit lifetime, beyond local views.
    runner._cam82_aligned_kv_caches = caches
    print(f"CAM82_KV_ALLOCATION_OK layers={NUM_LAYERS} regions={count} "
          f"alignment={ALIGNMENT} extra_reserved_bytes={count * ALIGNMENT} "
          "allocator=explicit_aligned_views native_binding=True", flush=True)
    return caches


def _initialize(self, kv_cache_config):
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.utils import bind_kv_cache
    return initialize(self, kv_cache_config, torch, FullAttentionSpec, bind_kv_cache)


def install(runner):
    if getattr(runner, "_cam82_aligned_kv_installed", False):
        return
    if (type(runner).__module__ != "vllm_ascend.worker.model_runner_v1"
            or type(runner).__name__ != "NPUModelRunner"
            or tuple(inspect.signature(runner.initialize_kv_cache_tensors).parameters) != ("kv_cache_config",)):
        raise RuntimeError("CAM82 aligned KV requires the pinned Ascend NPUModelRunner")
    validate_config(runner.vllm_config)
    runner.initialize_kv_cache_tensors = MethodType(_initialize, runner)
    runner._cam82_aligned_kv_installed = True
