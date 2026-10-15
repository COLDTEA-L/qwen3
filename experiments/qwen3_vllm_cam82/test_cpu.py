"""CPU/mock regressions. Numeric tests use NumPy, never Torch or an NPU."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "adapter"))
from qwen_cam82 import policy
from qwen_cam82.collectives import CamTP
from qwen_cam82 import hooks
import launch

try:
    import numpy as np
except ImportError:
    np = None


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.device = NS(type="npu")
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim

    def numel(self): return self.data.size
    def element_size(self): return self.data.itemsize
    def contiguous(self): return Tensor(np.ascontiguousarray(self.data))
    def view(self, *shape): return Tensor(self.data.reshape(*shape))
    def reshape(self, *shape): return self.view(*shape)
    def view_as(self, x): return Tensor(self.data.reshape(x.shape))
    def clone(self): return Tensor(self.data.copy())
    def movedim(self, a, b): return Tensor(np.moveaxis(self.data, a, b))
    def float(self): return Tensor(self.data.astype(np.float32))
    def to(self, dtype): return Tensor(self.data.astype(dtype))
    def sum(self, dim): return Tensor(self.data.sum(axis=dim))
    def expand(self, *shape): return Tensor(np.broadcast_to(self.data, shape))
    def __getitem__(self, key): return Tensor(self.data[key])
    def copy_(self, other): self.data[...] = other.data; return self


def fake_torch():
    return NS(float16=np.dtype("float16"), bfloat16="BF16-placeholder", float32=np.dtype("float32"),
        zeros=lambda shape, dtype, device: Tensor(np.zeros(shape, dtype=dtype)),
        empty=lambda shape, dtype, device: Tensor(np.empty(shape, dtype=dtype)),
        compiler=NS(is_compiling=lambda: False), npu=NS(),
        profiler=NS(record_function=lambda name: contextlib.nullcontext()))


def mock_runtime(exchange):
    obj = CamTP.__new__(CamTP)
    obj.mode, obj.calls, obj.lock = "cam-detour", 0, threading.Lock()
    obj.exchange = exchange
    return obj


class PolicyTests(unittest.TestCase):
    def test_physical_roles_no_reindex(self):
        self.assertEqual([policy.physical_device("p-228", i) for i in range(4)], [0, 1, 2, 3])
        self.assertEqual([policy.physical_device("d8-228", i) for i in range(4)], [4, 5, 6, 7])
        self.assertEqual([policy.physical_device("d8-238", i) for i in range(4)], [0, 1, 2, 3])
        for role, rank in (("unknown", 0), ("p-228", 4)):
            with self.assertRaises(ValueError): policy.physical_device(role, rank)

    def test_ep_group_not_selected_even_if_same_ranks(self):
        tp = NS(ranks=[0, 1, 2, 3])
        ep = NS(ranks=[0, 1, 2, 3])
        self.assertTrue(policy.selected_group(tp, tp, False))
        self.assertFalse(policy.selected_group(ep, tp, False))
        self.assertFalse(policy.selected_group(tp, tp, True))

    def test_hooks_exclude_moe_and_preserve_native_calls(self):
        class Group:
            def all_reduce(self, x): return "native-ar"
            def all_gather(self, x, dim=-1): return "native-ag"
            def reduce_scatter(self, x, dim=-1): return "native-rs"
        tp, ep = Group(), Group()
        class MoE:
            def forward(self, x):
                return tp.all_reduce(x), ep.all_reduce(x), tp.all_gather(x), tp.reduce_scatter(x)
        dispatcher = type("TokenDispatcherWithAll2AllV", (), {})()
        context = NS(fused_moe_state="a2a", token_dispatcher=dispatcher)
        names = ("vllm", "vllm.distributed", "vllm.distributed.parallel_state", "vllm.forward_context",
                 "vllm_ascend", "vllm_ascend.ascend_forward_context", "vllm_ascend.models", "vllm_ascend.models.qwen3_moe")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        pg = modules["vllm.distributed.parallel_state"]
        pg.GroupCoordinator, pg.get_ep_group = Group, lambda: NS(world_size=8)
        modules["vllm.forward_context"].get_forward_context = lambda: context
        modules["vllm_ascend.ascend_forward_context"].FusedMoEState = NS(All2All="a2a")
        modules["vllm_ascend.models.qwen3_moe"].CustomSparseMoeBlock = MoE
        runtime = NS(mode="cam-detour", all_reduce=lambda x: "cam-ar",
                     all_gather=lambda x: "cam-ag", reduce_scatter=lambda x: "cam-rs")
        with patch.dict(sys.modules, modules):
            hooks.install(runtime, tp)
            self.assertEqual(tp.all_reduce(1), "cam-ar")
            self.assertEqual(ep.all_reduce(1), "native-ar")
            self.assertEqual(MoE().forward(1), ("native-ar", "native-ar", "native-ag", "native-rs"))
            self.assertFalse(hooks.IN_MOE.get())
            context.fused_moe_state = "allgather"
            with self.assertRaises(RuntimeError): MoE().forward(1)
            self.assertFalse(hooks.IN_MOE.get())

    def test_capacity_matches_five_path_padding(self):
        for size in (2, 4):
            cap = policy.capacity(size)
            self.assertEqual(cap % 640, 0)
            self.assertLessEqual(cap * 4 * size, policy.CHUNK_BYTES)
            self.assertLessEqual(cap // 5 * size, 2 * 1024 * 1024)

    def test_moe_original_switch_and_pd_layout(self):
        for role, rank in (("d8-228", "0"), ("d8-238", "1")):
            cmd, env = launch.command(role, "cam-detour", True)
            self.assertEqual(cmd[cmd.index("--data-parallel-size") + 1], "2")
            self.assertEqual(cmd[cmd.index("--data-parallel-rank") + 1], rank)
            self.assertNotIn("--data-parallel-rank-local", cmd)
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", env)
            self.assertEqual(env["VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP"], "0")
            self.assertNotIn("qwen_moe_adapter", " ".join(cmd))
            kv = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
            self.assertFalse(kv["kv_connector_extra_config"]["use_ascend_direct"])
            self.assertIn("vllm_ascend.distributed.mooncake_connector", kv["kv_connector_module_path"])

    def test_config_native_only(self):
        cfg = NS(model_config=NS(enforce_eager=True, quantization=None, dtype="bfloat16",
            hf_config=NS(model_type="qwen3_moe", num_experts=128, num_experts_per_tok=8)),
            parallel_config=NS(tensor_parallel_size=4, data_parallel_size=2,
                               enable_expert_parallel=True, pipeline_parallel_size=1))
        policy.validate_config(cfg)
        cfg.model_config.enforce_eager = False
        with self.assertRaises(ValueError): policy.validate_config(cfg)

    def test_conservative_budgets_can_be_lowered(self):
        cmd, _ = launch.command("p-228", max_len=1024, max_seqs=2, memory=0.7)
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "1024")
        for kw in ({"max_len": 0}, {"max_seqs": 8}, {"memory": 1}):
            with self.assertRaises(ValueError): launch.command("p-228", **kw)

    def test_pinned_ascend_source_if_available(self):
        root = HERE.parents[2] / "vllm-ascend-082/vllm_ascend"
        if not root.is_dir(): self.skipTest("Pinned upstream source snapshot absent")
        policy.check_sources(root)
        text = (root / "ascend_forward_context.py").read_text()
        self.assertIn("ep_size < 16 or with_prefill", text)
        self.assertIn("return FusedMoEState.All2All", text)

    def test_source_guard_refuses_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError): policy.check_sources(directory)

    def test_no_hccl_payload_gather_in_cam_path(self):
        text = (HERE / "adapter/qwen_cam82/collectives.py").read_text()
        self.assertNotIn("dist.all_gather", text)
        self.assertNotIn("dist.all_reduce", text)
        self.assertIn("self.op.all2_all_detour", text)

    def test_docs_and_image_are_isolated(self):
        docker = (HERE / "Dockerfile").read_text()
        self.assertIn("v0.10.2rc1", docker)
        self.assertIn("USE_ASCEND_DIRECT=OFF", docker)
        self.assertIn("WITH_STORE=OFF", docker)
        self.assertNotIn("pip install --upgrade", docker)
        self.assertNotRegex((HERE / "README.md").read_text(), r"git\s+push")

    def test_crane_import_uses_local_base_and_has_resume_step(self):
        text = (HERE / "README.md").read_text()
        section = text.split("## 4 238宿主机", 1)[1].split("## 5 ", 1)[0]
        self.assertIn("### 4.5 检查vendor源码并构建实验镜像（现在从这里继续）", section)
        self.assertIn("--build-arg BASE_IMAGE=qwen3-vllm:cam82-base", section)
        self.assertIn("docker load -i /ssd_2/yuanwenliu/vllm-ascend.tar", section)
        self.assertIn("--entrypoint python qwen3-vllm:cam82-base -m pip show", section)
        self.assertNotRegex(section, r"(?m)^docker pull ")
        self.assertNotRegex(section, r"(?m)^docker (?:image inspect|run).*quay\.io/")


@unittest.skipIf(np is None, "Numeric mock tests require NumPy (already in inference image)")
class NumericTests(unittest.TestCase):
    def test_all_reduce_multiple_chunks_and_nonmutation(self):
        for n in (7, 641, 4 * policy.capacity(4) + 17):
            calls = [0]
            phases = (n + 3) // 4
            rs_calls = (phases + policy.capacity(4) - 1) // policy.capacity(4)
            def exchange(packed):
                calls[0] += 1
                if calls[0] <= rs_calls:
                    # Rank1 receives one shard from each source whose input
                    # is constant 1,2,3,4. Ignore dummy padding values.
                    return Tensor(np.stack([packed.data[1] + i for i in range(4)]))
                return Tensor(np.broadcast_to(packed.data[0], packed.shape).copy())
            runtime = mock_runtime(exchange)
            x = Tensor(np.ones(n, dtype=np.float32))
            with patch.dict(sys.modules, {"torch": fake_torch()}):
                actual = runtime.all_reduce(x)
            np.testing.assert_array_equal(actual.data, np.full(n, 10, dtype=np.float32))
            np.testing.assert_array_equal(x.data, np.ones(n, dtype=np.float32))

    def test_scalar_all_reduce(self):
        # First exchange gives reduced scalar10; second broadcasts same scalar.
        calls = [0]
        def exchange(p):
            calls[0] += 1
            return Tensor(np.stack([p.data[0] + i for i in range(4)])) if calls[0] == 1 else Tensor(np.tile(p.data[0], (4, 1)))
        runtime = mock_runtime(exchange)
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            actual = runtime.all_reduce(Tensor(np.array(1, dtype=np.float32)))
        self.assertEqual(actual.data.item(), 10)

    def test_gather_dimensions_and_noncontiguous(self):
        x = Tensor(np.arange(15, dtype=np.float32).reshape(3, 5).T)
        runtime = mock_runtime(lambda p: Tensor(np.stack([p.data[0] + i for i in range(4)])))
        for dim in (0, 1, -1):
            with patch.dict(sys.modules, {"torch": fake_torch()}):
                actual = runtime.all_gather(x, dim)
            np.testing.assert_array_equal(actual.data, np.concatenate([x.data + i for i in range(4)], axis=dim))

    def test_scatter_dimensions(self):
        x = Tensor(np.arange(32, dtype=np.float32).reshape(8, 4))
        runtime = mock_runtime(lambda p: Tensor(np.stack([p.data[1] + i for i in range(4)])))
        for dim in (0, 1, -1):
            with patch.dict(sys.modules, {"torch": fake_torch()}):
                actual = runtime.reduce_scatter(x, dim)
            np.testing.assert_array_equal(actual.data, np.split(4 * x.data + 6, 4, axis=dim)[1])

    def test_empty_shapes_and_bad_dimensions(self):
        runtime = mock_runtime(lambda p: self.fail("Empty input must not send payload"))
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            x = Tensor(np.empty((0, 4), dtype=np.float32))
            self.assertEqual(runtime.all_reduce(x).shape, (0, 4))
            self.assertEqual(runtime.all_gather(x, 0).shape, (0, 4))
            self.assertEqual(runtime.reduce_scatter(x, 0).shape, (0, 4))
            with self.assertRaises(ValueError): runtime.all_gather(x, 2)
            with self.assertRaises(ValueError): runtime.reduce_scatter(Tensor(np.ones((3, 4), dtype=np.float32)), 0)


if __name__ == "__main__":
    unittest.main()
