"""Dependency-free, fail-closed experiment policy."""
import hashlib
import importlib.metadata as md
import importlib.util
import math
from pathlib import Path

ROLES = {"p-228": ("228", 0), "d8-228": ("228", 4), "d8-238": ("238", 0)}
MODES = ("off", "hccl-a2a", "cam-detour")
CHUNK_BYTES = 4 * 1024 * 1024
BUFFER_MB = 512
SOURCE_HASHES = {
    "models/qwen3_moe.py": "6ed8991573eb6ab0b98b7a85c55a1920f1cb6ef5aa50615718fd7cfe48ac2de5",
    "ascend_forward_context.py": "f0dbff6437887a27f7730138c43e6a10afaa07acbf0652b472e2deadaf078cd7",
    "worker/worker_v1.py": "5274bf9f2e6a8db44cba2e8c410b32f2d72d842b85b0ba66d87225f4f06c92d4",
    "ops/fused_moe.py": "cce13119538dacd8468f97c9cfb51d37bb91944b691e26553c29e05c62e7934d",
    "distributed/mooncake_connector.py": "2d620cde420544ce505c12c75263ef0ed12dd7b8ffdafe5b1539c86897d0373c",
}


def check_sources(root):
    for name, expected in SOURCE_HASHES.items():
        path = Path(root) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unexpected Ascend source {path}; require v0.10.2rc1, do not bypass")


def check_installation():
    for name, expected in {"vllm": "0.10.2", "vllm-ascend": "0.10.2rc1",
                           "torch": "2.7.1", "torch-npu": "2.7.1.dev20250724"}.items():
        actual = md.version(name)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {actual}")
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None:
        raise RuntimeError("vllm_ascend missing")
    root = Path(next(iter(spec.submodule_search_locations)))
    check_sources(root)
    vllm_spec = importlib.util.find_spec("vllm")
    vllm_root = Path(next(iter(vllm_spec.submodule_search_locations)))
    pg_source = vllm_root / "distributed/parallel_state.py"
    if hashlib.sha256(pg_source.read_bytes()).hexdigest() != "6e84acb9fd2117a10fdad1f7433c04f8b5434710dc936dbba82506731d53240b":
        raise RuntimeError("Unexpected vLLM GroupCoordinator source; require v0.10.2")
    infos = list(Path("/usr/local/Ascend").glob("**/ascend_toolkit_install.info"))
    if not any("version=8.2.RC1" in p.read_text(errors="replace").splitlines() for p in infos):
        raise RuntimeError("Require CANN 8.2.RC1 toolkit metadata; do not mix CANN9")
    return root


def physical_device(role, local_rank):
    if role not in ROLES or not 0 <= local_rank < 4:
        raise ValueError("Require known role and four local TP ranks")
    return ROLES[role][1] + local_rank


def capacity(element_bytes):
    if element_bytes not in (2, 4):
        raise ValueError("Only fp16/bf16/fp32")
    # Five paths: direct + four ranks outside the TP subgroup. Each slice is
    # a multiple of 128 elements; 640-element rows satisfy every path's alignment.
    cap = CHUNK_BYTES // (4 * element_bytes) // 640 * 640
    slot_payload = BUFFER_MB * 1024 * 1024 // 64 // 2 - 2 * 1024 * 1024
    if cap <= 0 or math.ceil(cap / 5) * element_bytes > slot_payload:
        raise ValueError("CAM slot overflow")
    return cap


def validate_config(cfg):
    if not cfg.model_config.enforce_eager or cfg.model_config.quantization is not None:
        raise ValueError("Require eager/unquantized BF16")
    if str(cfg.model_config.dtype) not in ("torch.bfloat16", "bfloat16"):
        raise ValueError("Require BF16")
    hf = cfg.model_config.hf_config
    if hf.model_type != "qwen3_moe" or hf.num_experts != 128 or hf.num_experts_per_tok != 8:
        raise ValueError("Require original Qwen3-30B-A3B")
    p = cfg.parallel_config
    if p.tensor_parallel_size != 4 or p.data_parallel_size not in (1, 2) or not p.enable_expert_parallel:
        raise ValueError("Require TP4, DP1/2 and native EP")
    if p.pipeline_parallel_size != 1 or getattr(p, "enable_eplb", False):
        raise ValueError("No PP/EPLB")
    if getattr(cfg, "speculative_config", None) is not None:
        raise ValueError("No speculative decoding")


def selected_group(group, tp_group, in_moe):
    # Identity matters: EP4 may have the same rank list as TP4, but must remain
    # native. Never select solely by size or rank-list equality.
    return not in_moe and group is tp_group
