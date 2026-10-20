"""Sparse-memory CPU tests: model NPU pointers without allocating gigabytes."""
import copy
import gc
import math
import sys
import threading
import unittest
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "adapter"))
from qwen_cam82 import aligned_kv as alloc
from qwen_cam82 import native_kv as kv


class Storage:
    def __init__(self, ptr, count):
        self.ptr, self.count, self.values = ptr, count, {}


class Tensor:
    dtype = "torch.bfloat16"

    def __init__(self, storage, shape, offset=0):
        self.storage, self.shape, self.offset = storage, tuple(shape), offset

    def data_ptr(self): return self.storage.ptr + self.offset * 2
    def element_size(self): return 2
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return True
    def size(self, dim): return self.shape[dim]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.numel())
            assert step == 1
            return Tensor(self.storage, (stop - start,), self.offset + start)
        return self.storage.values.get(self.offset + key, 0)

    def view(self, shape):
        assert math.prod(shape) == self.numel()
        return Tensor(self.storage, shape, self.offset)

    def write(self, index, value):
        self.storage.values[self.offset + index] = value


class Torch:
    def __init__(self, base=0x12c5ded00600):
        self.base, self.allocations = base, []

    def zeros(self, count, dtype, device):
        assert dtype == "torch.bfloat16"
        # Address spacing, not actual memory. Each mock models a <=34MiB region.
        owner = Tensor(Storage(self.base + len(self.allocations) * 0x4000000, count), (count,))
        self.allocations.append(owner)
        return owner


class Spec(NS):
    pass


class Backend:
    @staticmethod
    def get_supported_block_size(): return [64]

    @staticmethod
    def get_kv_cache_shape(n, block, heads, dim): return (2, n, block, heads, dim)


def fixture():
    names = [f"model.layers.{i}.self_attn.attn" for i in range(48)]
    spec = Spec(block_size=64, num_kv_heads=1, head_size=128,
                dtype="torch.bfloat16", page_size_bytes=32768)
    group = NS(backend=Backend, layer_names=names)
    cfg = NS(num_blocks=2048, kv_cache_tensors=[
        NS(size=2048 * 32768, shared_by=[name]) for name in names])
    runner = NS(vllm_config=NS(cache_config=NS(block_size=64, num_gpu_blocks_override=2048),
                kv_transfer_config=NS(), additional_config={}),
                model_config=NS(is_deepseek_mla=False, enable_sleep_mode=False),
                kv_caches=[], runner_only_attn_layers=set(), device="npu:0",
                compilation_config=NS(static_forward_context={name: NS() for name in names}),
                _kv_cache_spec_attn_group_iterator=lambda: iter([(spec, group)]))
    return runner, cfg, spec, group


def bind(caches, context, runner_caches):
    assert not runner_caches
    for layer, pair in caches.items():
        context[layer].kv_cache = [pair]
        runner_caches.append(pair)


class AlignedKVTests(unittest.TestCase):
    def test_actual_reported_unaligned_K_pointer_gets_a_safe_forward_slice(self):
        for base in (0x12c5ded00600, 0x100000000, 0x100000002, 0x1001ffffe):
            torch = Torch(base)
            view, owner = alloc.allocate_aligned(torch, (3, 4), "torch.bfloat16", "npu:0")
            self.assertEqual(view.data_ptr(), (base + kv.ALIGNMENT - 1) // kv.ALIGNMENT * kv.ALIGNMENT)
            self.assertEqual(view.data_ptr() % kv.ALIGNMENT, 0)
            self.assertGreaterEqual(view.data_ptr(), owner.data_ptr())
            self.assertLessEqual(view.data_ptr() + 24, owner.data_ptr() + owner.numel() * 2)
            self.assertEqual([view[i] for i in range(12)], [0] * 12)
            view.write(3, 42)
            self.assertEqual(owner[view.offset + 3], 42)
            self.assertIs(view.storage, owner.storage)

    def test_full_cache_binding_registration_views_and_lifetime_match(self):
        runner, cfg, _, _ = fixture()
        torch = Torch()
        caches = alloc.initialize(runner, cfg, torch, Spec, bind)
        self.assertEqual(kv.validate_regions(caches), 96)
        self.assertEqual(len(torch.allocations), 96)
        self.assertEqual(len(runner._cam82_kv_backings), 48)
        self.assertIs(caches, runner._cam82_aligned_kv_caches)
        for index, (layer, pair) in enumerate(caches.items()):
            self.assertIs(runner.kv_caches[index], pair)
            self.assertIs(runner.compilation_config.static_forward_context[layer].kv_cache[0], pair)
            for part, owner in zip(pair, runner._cam82_kv_backings[layer]):
                self.assertIs(part.storage, owner.storage)
                self.assertEqual(owner.numel() * 2 - part.numel() * 2, kv.ALIGNMENT)
        owner = runner._cam82_kv_backings[next(iter(caches))][0]
        ref = weakref.ref(owner)
        del owner
        torch.allocations.clear()
        gc.collect()
        self.assertIsNotNone(ref())
        # The same dictionary can be handed directly to the native guard;
        # neither a stacked replacement tensor nor an adjusted pointer is used.
        self.assertEqual(kv.validate_regions(runner._cam82_aligned_kv_caches), 96)

    def test_unsupported_layouts_fail_before_allocating_or_binding(self):
        for change in ("capacity", "shared", "size", "dtype", "block", "binding", "mla", "sleep", "initialized"):
            runner, cfg, spec, group = fixture()
            if change == "capacity": cfg.num_blocks = 2047
            if change == "shared": cfg.kv_cache_tensors[0].shared_by.append("other")
            if change == "size": cfg.kv_cache_tensors[0].size -= 2
            if change == "dtype": spec.dtype = "torch.float16"
            if change == "block": spec.block_size = 128
            if change == "binding": runner.compilation_config.static_forward_context.clear()
            if change == "mla": runner.model_config.is_deepseek_mla = True
            if change == "sleep": runner.model_config.enable_sleep_mode = True
            if change == "initialized": runner.kv_caches.append("old")
            torch = Torch()
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                alloc.initialize(runner, cfg, torch, Spec, bind)
            self.assertEqual(torch.allocations, [])

    def test_native_initialization_binds_and_registers_the_same_tuple_views(self):
        import ast
        source = HERE.parents[2] / "vllm-ascend-082/vllm_ascend/worker/model_runner_v1.py"
        if not source.is_file():
            self.skipTest("Pinned Ascend source snapshot absent")
        cls = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "initialize_kv_cache")
        events = []
        namespace = {"KVCacheConfig": object, "deepcopy": copy.deepcopy,
                     "has_kv_transfer_group": lambda: True,
                     "get_kv_transfer_group": lambda: NS(register_kv_caches=lambda caches: events.append(caches))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        runner, cfg, _, _ = fixture()
        runner.may_reinitialize_input_batch = lambda config: None
        runner.initialize_attn_backend = lambda config: None
        runner.initialize_kv_cache_tensors = lambda config: alloc.initialize(runner, config, Torch(), Spec, bind)
        namespace["initialize_kv_cache"](runner, cfg)
        self.assertEqual(len(events), 1)
        self.assertIs(events[0], runner._cam82_aligned_kv_caches)
        self.assertIs(events[0][next(iter(events[0]))], runner.kv_caches[0])

    def test_installer_changes_only_the_selected_runner_instance(self):
        runner, _, _, _ = fixture()

        def native(self, kv_cache_config):
            raise AssertionError("native allocation should not be called")

        cls = type("NPUModelRunner", (), {"__module__": "vllm_ascend.worker.model_runner_v1",
                                         "initialize_kv_cache_tensors": native})
        selected, untouched = cls(), cls()
        selected.vllm_config = runner.vllm_config
        alloc.install(selected)
        installed = selected.initialize_kv_cache_tensors
        alloc.install(selected)
        self.assertIs(selected.initialize_kv_cache_tensors, installed)
        self.assertIs(untouched.initialize_kv_cache_tensors.__func__, native)
        self.assertIs(selected.initialize_kv_cache_tensors.__func__, alloc._initialize)
        with self.assertRaises(RuntimeError):
            alloc.install(NS())

    def test_pinned_native_registration_accepts_same_aligned_K_V_pairs(self):
        import ast
        source = HERE.parents[2] / "vllm-ascend-082/vllm_ascend/distributed/mooncake_connector.py"
        if not source.is_file():
            self.skipTest("Pinned Ascend source snapshot absent")
        cls = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "MooncakeConnectorWorker")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "register_kv_caches")

        class ReadyThread:
            def __init__(self, *args): self.ready = args[-1]
            def start(self): self.ready.set()

        namespace = {"torch": NS(Tensor=Tensor), "math": math, "threading": threading,
                     "logger": NS(info=lambda *args: None), "MooncakeAgentMetadata": NS,
                     "KVCacheSendingThread": ReadyThread, "KVCacheRecvingThread": ReadyThread}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        runner, cfg, _, _ = fixture()
        caches = alloc.initialize(runner, cfg, Torch(), Spec, bind)
        expected = [(tensor.data_ptr(), tensor.numel() * 2)
                    for pair in caches.values() for tensor in pair]
        for role in ("kv_producer", "kv_consumer"):
            registered = []
            worker = NS(kv_role=role, tp_rank=0, tp_size=4, _decode_tp_size=4,
                        engine_id="test", side_channel_host="host", side_channel_port=20010,
                        handshake_port=20010, te_rpc_port=12345, engine=NS(),
                        _register=lambda ptr, length: registered.append((ptr, length)))
            namespace["register_kv_caches"](worker, caches)
            self.assertIs(worker.kv_caches, caches)
            self.assertEqual(registered, expected)
            self.assertEqual(worker.block_len, [16384])
            self.assertEqual(worker.num_blocks, 2048)

    def test_current_version_and_docs_describe_explicit_allocation(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("### 5.5", text)
        self.assertIn("0.1.3", text)
        self.assertIn("CAM82_KV_ALLOCATION_OK", text)
        self.assertIn("192MiB", text)
        self.assertIn('version = "0.1.3"', (HERE / "adapter/pyproject.toml").read_text())


if __name__ == "__main__":
    unittest.main()
