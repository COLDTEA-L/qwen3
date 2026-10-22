"""Opt-in metadata-only capture of the pinned native EP8 dispatcher.

No tensor payloads are saved; capture runs are never performance runs.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path


def install():
    destination = os.getenv("QWEN_MOE_CAPTURE_DIR")
    if not destination or os.getenv("QWEN_CAM82_ROLE") == "p-228":
        return
    if os.getenv("QWEN_CAM82_MODE") != "off":
        raise RuntimeError("MoE capture requires native off mode")
    import torch.distributed as dist
    from vllm.forward_context import get_forward_context
    from vllm.distributed.parallel_state import get_ep_group
    from vllm_ascend.ops.moe import token_dispatcher as module
    ep = get_ep_group()
    if ep.world_size != 8:
        raise RuntimeError("MoE capture requires D EP8")
    cls = module.TokenDispatcherWithAll2AllV
    if getattr(cls.token_dispatch, "_capture", False):
        raise RuntimeError("MoE capture installed twice")
    skip = int(os.getenv("QWEN_MOE_CAPTURE_SKIP_PAIRS", "192"))
    count = int(os.getenv("QWEN_MOE_CAPTURE_PAIRS", "48"))
    if skip < 0 or not 1 <= count <= 192:
        raise ValueError("Require skip>=0 and 1<=capture pairs<=192")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"ep{ep.rank_in_group}.json"
    if path.exists():
        raise FileExistsError(f"Use a fresh capture directory: {path}")
    state = {"seen": 0, "phase": None, "ops": [], "pairs": [], "active": False}
    sources = {}
    from vllm_ascend.ops.moe import comm_utils
    for mod in (module, comm_utils):
        sources[mod.__name__] = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()

    def add(kind, tensor, group, ins=None, outs=None):
        if not state["active"]:
            return
        if group is not ep.device_group:
            raise RuntimeError("Captured collective not in the native EP group")
        state["ops"].append({"kind": kind, "phase": state["phase"],
            "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch."),
            "input_splits": None if ins is None else [int(x) for x in ins],
            "output_splits": None if outs is None else [int(x) for x in outs]})

    native_a2a = module.async_all_to_all
    @functools.wraps(native_a2a)
    def a2a(input_, output_split_sizes, input_split_sizes, group, event=None):
        if event is not None and state["active"]:
            raise RuntimeError("Unsupported multi-stream capture")
        add("all_to_all", input_, group, input_split_sizes, output_split_sizes)
        return native_a2a(input_, output_split_sizes, input_split_sizes, group, event)
    module.async_all_to_all = a2a
    native_gather = module.gather_from_sequence_parallel_region
    @functools.wraps(native_gather)
    def gather(input_, group, output_split_sizes=None):
        if output_split_sizes is not None and state["active"]:
            raise RuntimeError("Unsupported variable metadata gather")
        add("all_gather", input_, group)
        return native_gather(input_, group, output_split_sizes)
    module.gather_from_sequence_parallel_region = gather

    native_dispatch = cls.token_dispatch
    dispatch_signature = inspect.signature(native_dispatch)
    @functools.wraps(native_dispatch)
    def dispatch(self, *args, **kwargs):
        context = get_forward_context()
        eligible = not context.with_prefill and not context.in_profile_run
        state["active"] = eligible and skip <= state["seen"] < skip + count
        if eligible:
            state["seen"] += 1
        if state["active"]:
            bound = dispatch_signature.bind(self, *args, **kwargs)
            if bound.arguments.get("with_quant", False):
                raise RuntimeError("Only unquantized BF16 replay is supported")
            state["ops"] = []
        state["phase"] = "dispatch"
        return native_dispatch(self, *args, **kwargs)
    dispatch._capture = True
    cls.token_dispatch = dispatch
    native_combine = cls.token_combine
    @functools.wraps(native_combine)
    def combine(self, *args, **kwargs):
        state["phase"] = "combine"
        result = native_combine(self, *args, **kwargs)
        if state["active"]:
            state["pairs"].append(state["ops"])
            if len(state["pairs"]) == count:
                document = {"schema": 1, "origin": "native_decode_capture",
                    "ep_rank": ep.rank_in_group, "ep_size": 8,
                    "physical_device": __import__("torch").npu.current_device(),
                    "host": os.environ["QWEN_CAM82_HOST"], "skip_pairs": skip,
                    "source_hashes": sources,
                    "global_ranks": dist.get_process_group_ranks(ep.device_group),
                    "pairs": state["pairs"]}
                with path.open("x") as out:
                    json.dump(document, out)
                print(f"MOE_CAPTURE_READY rank={ep.rank_in_group} pairs={count} file={path}", flush=True)
        state["active"] = False
        return result
    cls.token_combine = combine
