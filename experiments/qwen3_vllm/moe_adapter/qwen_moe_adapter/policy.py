"""Pure host-side policy/compatibility checks; no Torch or device imports."""

import hashlib
import importlib.metadata as md
import importlib.util
from pathlib import Path

SOURCE_HASHES = {
    "ascend_forward_context.py": "e688fb9529628f4987f6f796765cc18bfe8387b3e08af6d36f2e1ee900f45d9b",
    "ops/fused_moe/prepare_finalize.py": "bb14acb9eae45dc03cbaf35b74eb6a4ce9eba3eabb4de471506c97f87ae2c1e7",
    "ops/fused_moe/token_dispatcher.py": "1a3b98eb8ce9bf20e4f83c5dfc5b810df99710c505a30d54b8d8086f2b7cc772",
    "ops/fused_moe/comm_utils.py": "ab23fa062e11e4965028d4f0526660f152a41a8dc3d57a6536538df9a834680d",
    "ops/fused_moe/moe_comm_method.py": "dc700f50c7177f68581a8a3ac858df8f2050e4a29a7f731528bbb2957dc4177b",
}


def check_sources(root):
    root = Path(root)
    for filename, expected in SOURCE_HASHES.items():
        path = root / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unsupported upstream source: {path}; use pinned v0.22.1rc1, do not bypass the guard")


def check_installation():
    for name, expected in {"vllm": "0.22.1", "vllm-ascend": "0.22.1rc1",
                           "torch": "2.10.0", "torch-npu": "2.10.0"}.items():
        actual = md.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"Unsupported {name}: {actual}; expected {expected}")
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend is missing")
    root = Path(next(iter(spec.submodule_search_locations)))
    check_sources(root)
    return root


def padded_tokens(tokens, tp):
    if tokens < 0 or tp < 1:
        raise ValueError("Invalid token count/TP size")
    # Idle DP ranks still take part in EP; retain a zero-valued dummy slice.
    return max(tp, ((tokens + tp - 1) // tp) * tp)


def validate_config(config):
    if (config.additional_config or {}).get("qwen_moe_comm") != "alltoall":
        raise ValueError("Experimental worker requires --moe-comm alltoall")
    model, parallel = config.model_config, config.parallel_config
    hf = model.hf_text_config
    if hf.model_type != "qwen3_moe" or hf.num_experts != 128 or hf.num_experts_per_tok != 8:
        raise ValueError("Only original Qwen3-30B-A3B (128 experts/top8) is qualified by this adapter")
    if str(model.dtype) not in ("bfloat16", "torch.bfloat16") or model.quantization is not None:
        raise ValueError("Only unquantized BF16 is supported")
    if not model.enforce_eager or getattr(config.compilation_config.cudagraph_mode, "name", "NONE") != "NONE":
        raise ValueError("Use eager mode; graph capture is not supported")
    if getattr(config, "speculative_config", None) is not None:
        raise ValueError("Speculative decoding is not supported")
    if not parallel.enable_expert_parallel or parallel.tensor_parallel_size != 4 or parallel.data_parallel_size not in (1, 2):
        raise ValueError("Only TP4 with DP1/DP2, hence EP4/EP8, is supported")
    for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"{name} must be 1")
    if getattr(parallel, "enable_eplb", False) or getattr(parallel, "enable_dbo", False):
        raise ValueError("Disable EPLB/DBO for the first All-to-All experiment")
    if getattr(parallel, "expert_placement_strategy", "linear") != "linear":
        raise ValueError("Contiguous expert IDs/linear placement are required")
    extra = config.additional_config or {}
    for name in ("enable_flashcomm1", "enable_matmul_allreduce", "enable_fused_mc2",
                 "enable_flashcomm2_parallel_size", "enable_shared_expert_dp",
                 "multistream_overlap_gate", "multistream_overlap_shared_expert"):
        if extra.get(name, False):
            raise ValueError(f"Disable {name} for the first All-to-All experiment")
    if extra.get("eplb_config", {}).get("dynamic_eplb", False):
        raise ValueError("Disable dynamic EPLB")


def make_selector(original, alltoall, native_context):
    def select(num_tokens, vllm_config, is_draft_model=False):
        mode = (vllm_config.additional_config or {}).get("qwen_moe_comm", "native")
        if mode == "native":
            return original(num_tokens, vllm_config, is_draft_model)
        if mode != "alltoall" or is_draft_model:
            raise ValueError("Unknown MoE mode or unsupported draft model")
        validate_config(vllm_config)
        # Check real group membership, not merely a CLI EP-size claim.
        ep = native_context.get_ep_group().world_size
        expected = vllm_config.parallel_config.tensor_parallel_size * vllm_config.parallel_config.data_parallel_size
        if ep != expected:
            raise RuntimeError(f"EP group mismatch: actual={ep}, expected={expected}")
        return alltoall
    select._qwen_original = original
    return select
