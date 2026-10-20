"""CPU/mock regressions. Numeric tests use NumPy, never Torch or an NPU."""
import contextlib
import ast
import ctypes
import importlib.util
import io
import json
import os
import shutil
import subprocess
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
from qwen_cam82 import startup
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
    @staticmethod
    def startup_config(role="d8-228"):
        return NS(model_config=NS(enforce_eager=True, quantization=None, dtype="bfloat16",
                  hf_config=NS(model_type="qwen3_moe", num_experts=128, num_experts_per_tok=8)),
                  parallel_config=NS(tensor_parallel_size=4, pipeline_parallel_size=1,
                    enable_expert_parallel=True, world_size=4, data_parallel_size=2,
                    data_parallel_size_local=1, data_parallel_rank=0 if role == "d8-228" else 1,
                    data_parallel_backend="mp", distributed_executor_backend="mp",
                    worker_cls=startup.WORKER_CLASS))

    @staticmethod
    def startup_environment(role="d8-228"):
        base = policy.ROLES[role][1]
        return {"QWEN_CAM82_ROLE": role, "QWEN_CAM82_MODE": "off",
                "PHYSICAL_DEVICES": ",".join(str(base + i) for i in range(4))}

    @staticmethod
    def fake_engine_utils():
        def get_device_indices():
            return "0,1,2,3"

        @contextlib.contextmanager
        def native(vllm_config, local_dp_rank):
            with patch.dict(os.environ, {"ASCEND_RT_VISIBLE_DEVICES": get_device_indices()}):
                yield

        class Manager:
            def __init__(self):
                # Same native helper lookup at the process-creation boundary.
                self.helper_name = "set_device_control_env_var"

        return NS(set_device_control_env_var=native, CoreEngineProcManager=Manager,
                  current_platform=NS(device_control_env_var="ASCEND_RT_VISIBLE_DEVICES"))

    def test_startup_environment_defaults_and_physical_mapping(self):
        for role in policy.ROLES:
            env = {**self.startup_environment(role), "HCCL_INTRA_ROCE_ENABLE": "0",
                   "HCCL_INTRA_PCIE_ENABLE": "1"}
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(startup.prepare_environment(), (role, "off"))
                self.assertNotIn("HCCL_INTRA_ROCE_ENABLE", os.environ)
                self.assertNotIn("HCCL_INTRA_PCIE_ENABLE", os.environ)
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
                with self.assertRaisesRegex(ValueError, "before process creation"):
                    startup.prepare_environment()
                del os.environ["ASCEND_RT_VISIBLE_DEVICES"]
                os.environ["PHYSICAL_DEVICES"] = "4,5,6,7" if role != "d8-228" else "0,1,2,3"
                with self.assertRaisesRegex(ValueError, "PHYSICAL_DEVICES"):
                    startup.prepare_environment()

    def test_dp_mask_is_prevented_before_child_creation_but_other_workers_unchanged(self):
        for role in ("d8-228", "d8-238"):
            helper = self.fake_engine_utils()
            startup.install_dp_visibility_policy(helper)
            installed = helper.set_device_control_env_var
            startup.install_dp_visibility_policy(helper)
            self.assertIs(installed, helper.set_device_control_env_var)
            cfg = self.startup_config(role)
            with patch.dict(os.environ, self.startup_environment(role), clear=True):
                with helper.set_device_control_env_var(cfg, 0):
                    # A newly created child sees the unmasked environment.
                    output = subprocess.check_output([sys.executable, "-c",
                        'import os; print(os.getenv("ASCEND_RT_VISIBLE_DEVICES", "UNSET")); '
                        'print(os.environ["PHYSICAL_DEVICES"])'], text=True)
                    self.assertEqual(output.splitlines(), ["UNSET", os.environ["PHYSICAL_DEVICES"]])
                cfg.parallel_config.worker_cls = "another.worker.Class"
                with helper.set_device_control_env_var(cfg, 0):
                    self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "0,1,2,3")
                self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", os.environ)

    def test_dp_policy_refuses_unsupported_layouts_and_masked_parent(self):
        helper = self.fake_engine_utils()
        startup.install_dp_visibility_policy(helper)
        with patch.dict(os.environ, self.startup_environment(), clear=True):
            for field, value in (("data_parallel_size_local", 2), ("world_size", 8),
                                 ("data_parallel_rank", 1), ("data_parallel_backend", "ray")):
                cfg = self.startup_config()
                setattr(cfg.parallel_config, field, value)
                with self.assertRaisesRegex(RuntimeError, "planned D8"):
                    with helper.set_device_control_env_var(cfg, 0):
                        self.fail("Unsupported layout entered")
            with self.assertRaisesRegex(RuntimeError, "planned D8"):
                with helper.set_device_control_env_var(self.startup_config(), 1):
                    self.fail("Unexpected local DP rank entered")
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
            with self.assertRaisesRegex(ValueError, "before process creation"):
                with helper.set_device_control_env_var(self.startup_config(), 0):
                    self.fail("Masked parent entered")
        helper = self.fake_engine_utils()
        helper.current_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            startup.install_dp_visibility_policy(helper)

    def test_launch_uses_startup_entrypoint_in_all_modes(self):
        for role in policy.ROLES:
            for mode in policy.MODES:
                cmd, env = launch.command(role, mode)
                self.assertEqual(cmd[1:4], ["-m", "qwen_cam82.entrypoint", "serve"])
                self.assertEqual(env["QWEN_CAM82_ROLE"], role)
                self.assertEqual(env["QWEN_CAM82_MODE"], mode)

    def test_hotfix_docs_require_both_installed_packages_and_scoped_git_trust(self):
        text = (HERE / "README.md").read_text()
        section = text.split("### 5.1 ", 1)[1].split("## 6 ", 1)[0]
        self.assertIn("228、238都执行（相同命令，容器内）", section)
        self.assertIn("--no-deps --no-build-isolation --no-index --force-reinstall", section)
        self.assertIn("git -c safe.directory=/ssd_2/yuanwenliu/sglang", section)
        self.assertIn('assert v=="0.1.1"', section)
        self.assertIn("CAM82_DP_VISIBILITY", section)
        self.assertIn("CAM82_WORKER_DEVICE", section)
        self.assertNotIn("--global", section)

    def test_entrypoint_installs_policy_before_running_native_cli(self):
        from qwen_cam82 import entrypoint
        events = []
        names = ("vllm", "vllm.v1", "vllm.v1.engine")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        modules["vllm.v1.engine"].utils = NS()
        with patch.dict(sys.modules, modules), \
             patch.dict(os.environ, self.startup_environment(), clear=True), \
             patch.object(entrypoint, "check_installation", side_effect=lambda: events.append("check")), \
             patch.object(entrypoint, "install_dp_visibility_policy", side_effect=lambda _: events.append("policy")), \
             patch.object(entrypoint.runpy, "run_module", side_effect=lambda *a, **kw: events.append((a, kw))):
            entrypoint.main()
        self.assertEqual(events[:2], ["check", "policy"])
        self.assertEqual(events[2], (("vllm.entrypoints.cli.main",), {"run_name": "__main__", "alter_sys": True}))

    def test_actual_worker_keeps_global_rank_and_checks_physical_card(self):
        class BaseWorker:
            def __init__(self, vllm_config, local_rank, rank, *args, **kwargs):
                self.local_rank, self.rank = local_rank, rank
                self.cfg = vllm_config
            def init_device(self):
                current[0] = self.local_rank

        names = ("vllm_ascend", "vllm_ascend.worker", "vllm_ascend.worker.worker_v1",
                 "vllm", "vllm.distributed", "vllm.distributed.parallel_state")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        modules["vllm_ascend.worker.worker_v1"].NPUWorker = BaseWorker
        current, ep, count = [None], [8], [8]
        torch = ModuleType("torch")
        torch.npu = NS(device_count=lambda: count[0], current_device=lambda: current[0])
        modules["torch"] = torch
        pg = modules["vllm.distributed.parallel_state"]
        pg.get_tp_group = lambda: "tp-group"
        pg.get_ep_group = lambda: NS(world_size=ep[0])
        spec = importlib.util.spec_from_file_location("qwen_cam82._worker_test",
                                                     HERE / "adapter/qwen_cam82/worker.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules), patch.object(policy, "check_installation"), \
             patch("qwen_cam82.collectives.CamTP", side_effect=lambda *args: NS(args=args)), \
             patch.object(hooks, "install"):
            spec.loader.exec_module(module)
            for role in policy.ROLES:
                ep[0] = 4 if role == "p-228" else 8
                with patch.dict(os.environ, self.startup_environment(role), clear=True):
                    for local_rank in range(4):
                        cfg = self.startup_config(role)
                        cfg.parallel_config.data_parallel_size = 1 if role == "p-228" else 2
                        cfg.parallel_config.data_parallel_rank_local = 1
                        worker = module.QwenCamWorker(cfg, local_rank, 10 + local_rank, "test")
                        self.assertEqual(worker.local_rank, policy.physical_device(role, local_rank))
                        self.assertEqual(worker.rank, 10 + local_rank)
                        self.assertEqual(cfg.parallel_config.data_parallel_rank_local, 0)
                        worker.init_device()
                        self.assertEqual(current[0], policy.physical_device(role, local_rank))
                    count[0] = 4
                    with self.assertRaisesRegex(RuntimeError, "all8 physical"):
                        worker.init_device()
                    count[0] = 8
                    with patch.object(BaseWorker, "init_device", side_effect=lambda: current.__setitem__(0, 99)):
                        with self.assertRaisesRegex(RuntimeError, "wrong NPU mapping"):
                            worker.init_device()
                    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
                    with self.assertRaisesRegex(ValueError, "before process creation"):
                        module.QwenCamWorker(cfg, 0, 0, "test")

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
            self.assertEqual(kv["kv_connector"], "MooncakeConnector")
            self.assertFalse(kv["kv_connector_extra_config"]["use_ascend_direct"])
            self.assertIn("vllm_ascend.distributed.mooncake_connector", kv["kv_connector_module_path"])

    def test_all_roles_and_modes_use_actual_ascend_connector_name(self):
        for role in policy.ROLES:
            for mode in policy.MODES:
                cmd, _ = launch.command(role, mode)
                kv = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
                self.assertEqual(kv["kv_connector"], "MooncakeConnector")
                self.assertEqual(kv["kv_connector_module_path"], "vllm_ascend.distributed.mooncake_connector")

    def test_all_roles_and_modes_use_64_token_kv_blocks(self):
        self.assertEqual(launch.KV_BLOCK_SIZE, 64)
        for role in policy.ROLES:
            for mode in policy.MODES:
                with self.subTest(role=role, mode=mode):
                    cmd, _ = launch.command(role, mode)
                    self.assertEqual(cmd.count("--block-size"), 1)
                    self.assertEqual(cmd[cmd.index("--block-size") + 1], "64")

    def test_docs_explain_kv_block_units_and_correctness_gate(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("### 5.3", text)
        self.assertIn("--block-size 64", text)
        self.assertIn("逻辑block", text)
        self.assertIn("物理block", text)
        self.assertIn("Compute 17 + 25. Give only the answer.", text)
        self.assertIn("不能以HTTP200或不同TP模式输出相同作为正确性通过", text)

    def test_connector_factory_resolution_is_v1_and_never_constructs_transport(self):
        class Base:
            pass
        class Connector(Base):
            def __init__(self, *args, **kwargs):
                raise AssertionError("Preflight must not construct a connector/TE")
        names = ("vllm", "vllm.config", "vllm.distributed", "vllm.distributed.kv_transfer",
                 "vllm.distributed.kv_transfer.kv_connector", "vllm.distributed.kv_transfer.kv_connector.factory",
                 "vllm.distributed.kv_transfer.kv_connector.v1", "vllm.distributed.kv_transfer.kv_connector.v1.base",
                 "vllm_ascend", "vllm_ascend.distributed", "vllm_ascend.distributed.mooncake_connector")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        connector_module = modules["vllm_ascend.distributed.mooncake_connector"]
        connector_module.MooncakeConnector = Connector
        modules["vllm.config"].KVTransferConfig = NS
        modules["vllm.distributed.kv_transfer.kv_connector.v1.base"].KVConnectorBase_V1 = Base
        factory = NS(get_connector_class=lambda cfg: getattr(connector_module, cfg.kv_connector))
        modules["vllm.distributed.kv_transfer.kv_connector.factory"].KVConnectorFactory = factory
        cmd, _ = launch.command("p-228")
        kv = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
        with patch.dict(sys.modules, modules):
            self.assertIs(launch.check_connector(kv), Connector)
            with self.assertRaisesRegex(RuntimeError, "class name"):
                launch.check_connector({**kv, "kv_connector": "MooncakeConnectorV1"})
            factory.get_connector_class = lambda cfg: type("WrongTransport", (Base,), {})
            with self.assertRaisesRegex(RuntimeError, "different connector"):
                launch.check_connector(kv)
            connector_module.MooncakeConnector = object
            with self.assertRaisesRegex(RuntimeError, "Expected Ascend V1"):
                launch.check_connector(kv)

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
        connector_tree = ast.parse((root / "distributed/mooncake_connector.py").read_text())
        classes = {node.name: node for node in connector_tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("MooncakeConnector", classes)
        self.assertNotIn("MooncakeConnectorV1", classes)
        self.assertIn("KVConnectorBase_V1", [base.id for base in classes["MooncakeConnector"].bases
                                            if isinstance(base, ast.Name)])

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

    def test_toolkit_libraries_precede_devlib_and_local_copies(self):
        docker = (HERE / "Dockerfile").read_text()
        cam_stage = docker.split('RUN test "$(git -C /opt/umdk-cam82', 1)[1].split("COPY check.py", 1)[0]
        self.assertNotIn('LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/aarch64-linux/devlib:', cam_stage)
        ld_line = next(line.strip().removesuffix(" && \\") for line in cam_stage.splitlines()
                       if line.strip().startswith("export LD_LIBRARY_PATH="))
        env_line = next(line for line in (HERE / "env.sh").read_text().splitlines()
                        if line.startswith("export LD_LIBRARY_PATH="))
        for assignment in (ld_line, env_line):
            script = 'ASCEND_HOME_PATH=/mock/cann\nLD_LIBRARY_PATH=/mock/cann/aarch64-linux/devlib:/mock/atb\n'
            script += assignment + '\nprintf "%s" "$LD_LIBRARY_PATH"'
            paths = subprocess.check_output(["bash", "-c", script], text=True).split(":")
            self.assertEqual(paths[:2], ["/mock/cann/lib64", "/mock/cann/aarch64-linux/lib64"])
        self.assertLess(cam_stage.index("CAM82_BUILD_TORCH_IMPORT_OK"), cam_stage.index("BUILD_JOBS=4"))
        self.assertIn("CAM82_NATIVE_IMPORT_OK", docker)

    def test_runtime_environment_for_both_hosts_and_missing_cam_api(self):
        # Source the actual script with only machine-specific SDK/workdir
        # operations mocked; no Torch/NPU imports or real SDK writes.
        text = (HERE / "env.sh").read_text()
        text = text.replace("source /usr/local/Ascend/ascend-toolkit/set_env.sh || return", ":")
        text = text.replace("source /usr/local/Ascend/nnal/atb/set_env.sh || return", ":")
        text = text.replace("cd /ssd_2/yuanwenliu/sglang || return", 'cd "$CAM82_TEST_WORKDIR" || return')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "env.sh"
            script.write_text(text)
            library = root / "opp/vendors/CAM/op_api/lib/libcust_opapi.so"
            library.parent.mkdir(parents=True)
            library.touch()
            env = {**os.environ, "CAM82_TEST_WORKDIR": directory,
                   "ASCEND_HOME_PATH": "/mock/cann", "ASCEND_OPP_PATH": str(root / "opp"),
                   "ASCEND_CUSTOM_OPP_PATH": "/mock/existing/vendor",
                   "LD_LIBRARY_PATH": "/mock/cann/aarch64-linux/devlib",
                   "HCCL_INTRA_PCIE_ENABLE": "1", "HCCL_INTRA_ROCE_ENABLE": "0"}
            for host in ("228", "238"):
                command = ('source "$1" "$2" || exit $?; '
                           'test -z "${HCCL_INTRA_PCIE_ENABLE+x}" || exit 20; '
                           'test -z "${HCCL_INTRA_ROCE_ENABLE+x}" || exit 21; '
                           'printf "STATE=%s|%s|%s|%s\\n" "$QWEN_CAM82_HOST" '
                           '"$VLLM_HOST_IP" "$ASCEND_CUSTOM_OPP_PATH" "$LD_LIBRARY_PATH"')
                output = subprocess.check_output(["bash", "-c", command, "test", str(script), host],
                                                 env=env, text=True)
                state = next(line for line in output.splitlines() if line.startswith("STATE="))
                actual_host, ip, custom, ld = state.removeprefix("STATE=").split("|")
                self.assertEqual((actual_host, ip), (host, "10.155.205." + host))
                self.assertEqual(custom.split(":"), [str(root / "opp/vendors/CAM"), "/mock/existing/vendor"])
                self.assertEqual(ld.split(":")[:3], ["/mock/cann/lib64", "/mock/cann/aarch64-linux/lib64",
                                                       str(library.parent)])
            library.unlink()
            result = subprocess.run(["bash", "-c", 'source "$1" 238', "test", str(script)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Missing CAM operator API:", result.stderr)
            self.assertNotIn("CAM82_ENV_READY", result.stdout)

    def test_238_docs_gate_each_probe_and_do_not_require_rebuild(self):
        text = (HERE / "README.md").read_text()
        section = text.split("## 6 ", 1)[1].split("## 7 ", 1)[0]
        self.assertIn("env.sh 238", section)
        self.assertIn('ctypes.CDLL("libcust_opapi.so")', section)
        self.assertIn("CAM_TP_SMOKE_PASS mode=hccl-a2a", section)
        self.assertIn("前一项失败不要继续", section)
        self.assertIn("不需要重建、重新传输镜像", text)

    def test_docs_windows_are_self_contained_and_hosts_explicit(self):
        text = (HERE / "README.md").read_text()
        self.assertNotIn("source参数改为", text)
        self.assertIn("228、238都执行（相同命令", text)
        gates = text.split("## 6 ", 1)[1].split("## 7 ", 1)[0]
        for host in ("228", "238"):
            self.assertIn("env.sh " + host, gates)
        for section_id, next_id, modes in (("7", "8", ("off",)),
                                            ("8", "9", ("cam-detour", "hccl-a2a"))):
            section = text.split("## " + section_id + " ", 1)[1].split("## " + next_id + " ", 1)[0]
            blocks = [chunk.split("```", 1)[0] for chunk in section.split("```bash\n")[1:]]
            for mode in modes:
                for role, host in (("p-228", "228"), ("d8-228", "228"), ("d8-238", "238")):
                    command = "launch.py serve --role " + role + " --tp-mode " + mode
                    matches = [block for block in blocks if command in block]
                    self.assertEqual(len(matches), 1, command)
                    self.assertTrue(matches[0].startswith("docker exec -it qwen3-vllm-cam82 bash\n"))
                    self.assertIn("env.sh " + host + "\n", matches[0])
                    self.assertIn("set -o pipefail", matches[0])
            for block in blocks:
                if "launch.py proxy" in block or "compare_outputs.py collect" in block:
                    self.assertTrue(block.startswith("docker exec -it qwen3-vllm-cam82 bash\n"))
                    self.assertIn("env.sh 228\n", block)
                if "relay.py" in block:
                    self.assertTrue(block.startswith("docker exec -it qwen3-vllm-cam82 bash\n"))
                    self.assertIn("env.sh 238\n", block)

    def test_pregen_patch_maps_only_all2all_and_refuses_reapplication(self):
        patch_file = HERE / "patches/umdk-all2all-pregen-prefix.patch"
        original = ('\n# aclnn autogen file prefix: all_gather_detour -> aclnn_all_gather_detour\n'
                    'ACLN_PREFIX="aclnn_${OP_NAME}"\n\nCopySingleOp() {\n    local dst_dir="$1"\n}\n')
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "build/cam/comm_operator/compile_single_op.sh"
            script.parent.mkdir(parents=True)
            script.write_text(original)
            # Outside a repository, git apply still performs normal hunk checks.
            cmd = ["git", "apply"]
            subprocess.run(cmd + ["--check", str(patch_file)], cwd=directory, check=True,
                           capture_output=True, text=True)
            subprocess.run(cmd + [str(patch_file)], cwd=directory, check=True,
                           capture_output=True, text=True)
            text = script.read_text()
            subprocess.run(["bash", "-n", str(script)], check=True)
            mapping = text.split("CopySingleOp()", 1)[0]
            for operator, expected in (("all_to_all_detour", "aclnn_all2_all_detour"),
                                       ("all_gather_detour", "aclnn_all_gather_detour"),
                                       ("reduce_scatter_detour", "aclnn_reduce_scatter_detour")):
                output = subprocess.check_output(["bash", "-c", 'OP_NAME="$1"\n' + mapping +
                                                   '\nprintf "%s" "$ACLN_PREFIX"', "test", operator], text=True)
                self.assertEqual(output, expected)
            result = subprocess.run(cmd + ["--check", str(patch_file)], cwd=directory,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_pregen_patch_is_guarded_and_applied_before_cam_build(self):
        docker = (HERE / "Dockerfile").read_text()
        cam_stage = docker.split('RUN test "$(git -C /opt/umdk-cam82', 1)[1].split("COPY check.py", 1)[0]
        self.assertIn("ce9d1a7821b2956c2b56f6d4f4e8518abf99238676494c6d323dc9f2fca9c889", cam_stage)
        self.assertIn("git -C /opt/umdk-cam82 apply --check", cam_stage)
        self.assertIn("aclnn_all2_all_detour.cpp", cam_stage)
        self.assertIn("aclnn_all2_all_detour.h", cam_stage)
        self.assertLess(cam_stage.index("CAM82_PREGEN_PREFIX_PATCH_OK"), cam_stage.index("BUILD_JOBS=4"))
        self.assertIn("4.5.2 已定位", (HERE / "README.md").read_text())

    def test_cann_env_helper_is_installed_and_guarded_before_cam_build(self):
        docker = (HERE / "Dockerfile").read_text()
        self.assertIn("COPY build_helpers/source_cann_env_compat.sh /opt/umdk-cam82/build/cam/comm_operator/", docker)
        self.assertIn("apply --check /opt/cam82/umdk-cann-env-guard.patch", docker)
        self.assertLess(docker.index("CAM82_CANN_ENV_PATCH_OK"), docker.index("BUILD_JOBS=4"))
        text = (HERE / "patches/umdk-cann-env-guard.patch").read_text()
        self.assertIn("+set -Eeo pipefail", text)
        self.assertIn('+    source "${MODULE_SCRIPTS_PATH}/source_cann_env_compat.sh"', text)
        self.assertIn("4.5.3 已定位", (HERE / "README.md").read_text())


class CannEnvTests(unittest.TestCase):
    """The SDK fixture reproduces a grep miss without requiring CANN or Torch."""

    def run_sdk(self, body, missing=None, tail='echo BUILD_NEXT_OK'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("bin", "include/acl", "opp", "tools"):
                (root / name).mkdir(parents=True)
            (root / "include/acl/acl.h").write_text("// fixture header\n")
            env_script = root / "bin/setenv.bash"
            env_script.write_text(body)
            for name in ("msopgen", "cmake"):
                tool = root / "tools" / name
                tool.write_text("#!/bin/bash\nexit 0\n")
                tool.chmod(0o755)
            if missing:
                target = root / missing
                if target.is_dir():
                    target.rmdir()
                else:
                    target.unlink()
            script = '''set -Ee -o pipefail
trap 'rc=$?; printf "STRICT_FAIL rc=%s source=%s line=%s\\n" "$rc" "${BASH_SOURCE[0]:-shell}" "$LINENO" >&2; exit "$rc"' ERR
export ASCEND_HOME_PATH="$1"
export ASCEND_OPP_PATH="$1/opp"
export PATH="$1/tools"
export LD_LIBRARY_PATH="$1/aarch64-linux/devlib:/mock/atb"
source "$2"
printf "SDK_LD=%s\\n" "$LD_LIBRARY_PATH"
'''
            return subprocess.run(["/bin/bash", "-c", script + tail, "test", str(root),
                                   str(HERE / "build_helpers/source_cann_env_compat.sh")],
                                  capture_output=True, text=True)

    def test_grep_miss_can_finish_env_and_preserve_exports(self):
        body = 'printf "available\\n" | /usr/bin/grep "absent" >/dev/null\nexport SDK_FINISH_SEEN=yes\nreturn 0\n'
        result = self.run_sdk(body, tail='test "$SDK_FINISH_SEEN" = yes\necho BUILD_NEXT_OK')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CAM82_CANN_ENV_READY", result.stdout)
        self.assertIn("BUILD_NEXT_OK", result.stdout)
        paths = next(line for line in result.stdout.splitlines() if line.startswith("SDK_LD=")).split("=", 1)[1].split(":")
        self.assertTrue(paths[0].endswith("/lib64"))
        self.assertTrue(paths[1].endswith("/aarch64-linux/lib64"))
        self.assertTrue(paths[-1].endswith("/aarch64-linux/devlib"))

    def test_nonzero_final_sdk_status_is_not_ignored(self):
        result = self.run_sdk("return 7\n")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("CANN environment returned 7; stopping", result.stderr)
        self.assertNotIn("CAM82_CANN_ENV_READY", result.stdout)
        self.assertNotIn("BUILD_NEXT_OK", result.stdout)

    def test_missing_sdk_resources_and_tools_are_fatal(self):
        for missing in ("bin/setenv.bash", "include/acl/acl.h", "opp", "tools/msopgen", "tools/cmake"):
            with self.subTest(missing=missing):
                result = self.run_sdk("return 0\n", missing=missing)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("[cam82_env] missing", result.stderr)
                self.assertNotIn("CAM82_CANN_ENV_READY", result.stdout)
                self.assertNotIn("BUILD_NEXT_OK", result.stdout)

    def test_build_failure_after_env_still_stops_immediately(self):
        result = self.run_sdk("return 0\n", tail="/bin/false\necho BUILD_NEXT_OK")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("CAM82_CANN_ENV_READY", result.stdout)
        self.assertIn("STRICT_FAIL rc=1", result.stderr)
        self.assertNotIn("BUILD_NEXT_OK", result.stdout)


class OpPackageInstallTests(unittest.TestCase):
    def installation_stage(self):
        lines = (HERE / "patches/umdk-op-package-install.patch").read_text().splitlines()
        hunk = lines[next(i for i, line in enumerate(lines) if line.startswith("@@")) + 1:]
        revised = "\n".join(line[1:] for line in hunk if line.startswith(("+", " ")))
        return revised[revised.index("# msopgen"):].split("export CAM_COMM_PATH", 1)[0]

    def run_installation(self, mode="ok", custom_prefix=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "custom op.run"
            package.write_text('''#!/bin/bash
set -eu
printf '%s\\n' "$@" > "$CAM82_TEST_ARGS"
dest=
for arg in "$@"; do
    case "$arg" in
        --quiet|--nox11) ;;
        --install-path=*) dest="${arg#--install-path=}" ;;
        *) echo "unsupported installer option: $arg" >&2; exit 99 ;;
    esac
done
test "$dest" = "$CAM82_TEST_EXPECTED_OPP"
case "$CAM82_TEST_MODE" in fail) exit 7 ;; empty) exit 0 ;; esac
mkdir -p "$dest/vendors/CAM/op_api/lib" "$dest/vendors/CAM/op_api/include"
if [ "$CAM82_TEST_MODE" != no-library ]; then
    printf library > "$dest/vendors/CAM/op_api/lib/libcust_opapi.so"
fi
if [ "$CAM82_TEST_MODE" != no-header ]; then
    printf header > "$dest/vendors/CAM/op_api/include/aclnn_all2_all_detour.h"
fi
''')
            opp = root / "mock opp"
            destination = root / "isolated opp" if custom_prefix else opp
            args_file = root / "installer-args.txt"
            env = dict(os.environ, CAM82_TEST_ARGS=str(args_file), CAM82_TEST_MODE=mode,
                       CAM82_TEST_EXPECTED_OPP=str(destination))
            env.pop("CAM_OPP_INSTALL_PATH", None)
            if custom_prefix: env["CAM_OPP_INSTALL_PATH"] = str(destination)
            script = ('set -euo pipefail\npackages=("$1")\nexport ASCEND_OPP_PATH="$2"\n' +
                      self.installation_stage() + '\necho WHEEL_STAGE_REACHED\n')
            result = subprocess.run(["/bin/bash", "-c", script, "test", str(package), str(opp)],
                                    env=env, capture_output=True, text=True, timeout=15)
            if custom_prefix: self.assertFalse(opp.exists(), "Do not overwrite the SDK OPP")
            return result, args_file.read_text().splitlines(), str(destination)

    def test_supported_arguments_install_into_current_opp(self):
        result, args, opp = self.run_installation()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args, ["--quiet", "--nox11", "--install-path=" + opp])
        self.assertIn("CAM82_OP_PACKAGE_INSTALLED", result.stdout)
        self.assertIn("WHEEL_STAGE_REACHED", result.stdout)

    def test_installer_failure_or_missing_files_never_reaches_wheel_stage(self):
        for mode in ("fail", "empty", "no-library", "no-header"):
            with self.subTest(mode=mode):
                result, _, _ = self.run_installation(mode)
                self.assertEqual(result.returncode, 7 if mode == "fail" else 1, result.stderr)
                self.assertNotIn("CAM82_OP_PACKAGE_INSTALLED", result.stdout)
                self.assertNotIn("WHEEL_STAGE_REACHED", result.stdout)
                if mode != "fail": self.assertIn("required API files are missing", result.stderr)

    def test_installation_can_use_an_isolated_prefix(self):
        result, args, destination = self.run_installation(custom_prefix=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[-1], "--install-path=" + destination)

    def test_install_patch_is_guarded_and_applied_before_build(self):
        docker = (HERE / "Dockerfile").read_text()
        self.assertIn("867b56c0f655ee0ce0e92164725d64a356eec9e2b80a0b366ae7c01e31c88a65", docker)
        self.assertIn("apply --check /opt/cam82/umdk-op-package-install.patch", docker)
        self.assertLess(docker.index("CAM82_OP_INSTALL_PATCH_OK"), docker.index("BUILD_JOBS=4"))
        self.assertNotIn('" --install\n', self.installation_stage())
        self.assertIn("4.5.5 已定位", (HERE / "README.md").read_text())


class PlatformIncludeTests(unittest.TestCase):
    def patcher(self):
        # Reconstruct the changed embedded Python from the actual shipped hunk,
        # not a second implementation of its CMake modification.
        lines = (HERE / "patches/umdk-platform-include.patch").read_text().splitlines()
        hunk = lines[next(i for i, line in enumerate(lines) if line.startswith("@@")) + 1:]
        revised = "\n".join(line[1:] for line in hunk if line.startswith(("+", " ")))
        prefix = '''import sys
from pathlib import Path
cmake_file = Path(sys.argv[1])
text = cmake_file.read_text()
insert = """
if(EXISTS "${CMAKE_SOURCE_DIR}/pregen/build_out/autogen")
    file(COPY "${CMAKE_SOURCE_DIR}/pregen/build_out/autogen/"
         DESTINATION "${ASCEND_AUTOGEN_PATH}"
         FILES_MATCHING
         PATTERN "aclnn_*.cpp"
         PATTERN "aclnn_*.h")
'''
        return prefix + revised.split("\nPY\n", 1)[0]

    def apply_to(self, path):
        subprocess.run([sys.executable, "-c", self.patcher(), str(path)],
                       check=True, capture_output=True, text=True)

    def test_platform_patch_is_target_scoped_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            cmake_file = Path(directory) / "CMakeLists.txt"
            cmake_file.write_text('file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)\n')
            self.apply_to(cmake_file)
            first = cmake_file.read_text()
            self.apply_to(cmake_file)
            self.assertEqual(cmake_file.read_text(), first)
            self.assertEqual(first.count("CAM82_PLATFORM_INCLUDE="), 1)
            self.assertIn('target_include_directories(cust_opapi PRIVATE "${_cam82_platform_include}")', first)
            self.assertIn('$ENV{ASCEND_HOME_PATH}/include/experiment/platform', first)
            self.assertNotIn("CPLUS_INCLUDE_PATH", first)
        docker = (HERE / "Dockerfile").read_text()
        self.assertIn("apply --check /opt/cam82/umdk-platform-include.patch", docker)
        self.assertLess(docker.index("CAM82_PLATFORM_INCLUDE_PATCH_OK"), docker.index("BUILD_JOBS=4"))
        self.assertIn("4.5.4 已定位", (HERE / "README.md").read_text())

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cmake") and shutil.which("c++"),
                         "Linux CMake/compiler regression (Ascend deployment targets Linux)")
    def test_host_compile_reproduces_failure_then_uses_sdk_header_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "mock sdk"
            platform = sdk / "include/experiment/platform/platform"
            platform.mkdir(parents=True)
            (platform / "platform_infos_def.h").write_text('inline int platform_fixture() { return 7; }\n')
            (platform / "platform_info.h").write_text('#include "platform/platform_infos_def.h"\n')
            opdev = sdk / "include/aclnn/opdev"
            opdev.mkdir(parents=True)
            (opdev / "platform.h").write_text('#include "platform/platform_info.h"\n')
            project = root / "project"
            project.mkdir()
            autogen = project / "autogen"
            autogen.mkdir()
            (autogen / "aclnn_fake.cpp").write_text('extern "C" int fixture_inner();\nextern "C" int fixture_wrapper() { return fixture_inner(); }\n')
            (autogen / "aclnnInner_fake.cpp").write_text('extern "C" int fixture_inner() { return 42; }\n')
            (project / "api.cpp").write_text('#include "aclnn/opdev/platform.h"\nint api() { return platform_fixture(); }\n')
            (project / "other.cpp").write_text('int other() { return 1; }\n')
            cmake_file = project / "CMakeLists.txt"
            cmake_file.write_text('''cmake_minimum_required(VERSION 3.12)
project(cam82_header_fixture LANGUAGES CXX)
set(ASCEND_AUTOGEN_PATH "${CMAKE_SOURCE_DIR}/autogen")
file(MAKE_DIRECTORY "${ASCEND_AUTOGEN_PATH}")
file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
add_library(cust_opapi SHARED api.cpp ${aclnn_src})
target_include_directories(cust_opapi PRIVATE "$ENV{ASCEND_HOME_PATH}/include")
add_library(unrelated OBJECT other.cpp)
''')
            env = dict(os.environ, ASCEND_HOME_PATH=str(sdk))
            configure = [shutil.which("cmake"), "-S", str(project), "-B", str(root / "build"),
                         "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
            build = [shutil.which("cmake"), "--build", str(root / "build"), "--parallel", "2"]
            subprocess.run(configure, env=env, check=True, capture_output=True, text=True, timeout=45)
            before = subprocess.run(build, env=env, capture_output=True, text=True, timeout=45)
            self.assertNotEqual(before.returncode, 0)
            self.assertIn("platform/platform_info.h", before.stderr)
            self.apply_to(cmake_file)
            after_config = subprocess.run(configure, env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(after_config.returncode, 0, after_config.stderr)
            self.assertIn("CAM82_PLATFORM_INCLUDE=", after_config.stdout)
            after = subprocess.run(build, env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(after.returncode, 0, after.stdout + after.stderr)
            commands = json.loads((root / "build/compile_commands.json").read_text())
            self.assertIn("aclnnInner_fake.cpp", [Path(item["file"]).name for item in commands])
            for item in commands:
                self.assertEqual("include/experiment/platform" in item["command"],
                                 Path(item["file"]).name != "other.cpp")
            native = ctypes.CDLL(str(root / "build/libcust_opapi.so"))
            self.assertEqual(native.fixture_wrapper(), 42)
            # CANN8.5 places the same platform headers directly below include/.
            platform85 = sdk / "include/platform"
            platform85.mkdir()
            for name in ("platform_info.h", "platform_infos_def.h"):
                shutil.move(str(platform / name), str(platform85 / name))
            layout85 = subprocess.run(configure, env=env, capture_output=True, text=True, timeout=45)
            self.assertEqual(layout85.returncode, 0, layout85.stderr)
            self.assertIn("CAM82_PLATFORM_INCLUDE=" + str(sdk / "include"), layout85.stdout)
            subprocess.run(build, env=env, check=True, capture_output=True, text=True, timeout=45)
            (platform85 / "platform_info.h").unlink()
            missing = subprocess.run(configure, env=env, capture_output=True, text=True, timeout=45)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("CAM82 missing SDK header", missing.stderr)


class Ipv4ValidationTests(unittest.TestCase):
    def function(self):
        lines = (HERE / "patches/umdk-ipv4-no-static-regex.patch").read_text().splitlines()
        return "\n".join(line[1:] for line in lines if line.startswith("+") and not line.startswith("+++"))

    def test_no_static_regex_and_build_wiring(self):
        text = (HERE / "patches/umdk-ipv4-no-static-regex.patch").read_text()
        self.assertIn("-static const std::regex ipv4_regex(", text)
        self.assertNotIn("std::regex", self.function().split("bool Ipv4IsValid", 1)[1])
        docker = (HERE / "Dockerfile").read_text()
        self.assertIn("apply --check /opt/cam82/umdk-ipv4-no-static-regex.patch", docker)
        self.assertLess(docker.index("CAM82_IPV4_NO_STATIC_REGEX_PATCH_OK"), docker.index("BUILD_JOBS=4"))

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("c++"), "Linux C++ regression")
    def test_cpp_parser_matches_previous_regex(self):
        source = r'''#include <regex>
#include <string>
#include <vector>
#include <iostream>
#include <random>
''' + self.function() + r'''
int main() {
    const std::regex original(R"(^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$)", std::regex::optimize);
    size_t cases = 0;
    auto check = [&](const std::string& ip) {
        ++cases;
        if (Ipv4IsValid(ip) != std::regex_match(ip, original)) {
            std::cerr << "IPv4 mismatch: " << ip << std::endl;
            return false;
        }
        return true;
    };
    for (int field = 0; field != 4; ++field) {
        for (int n = 0; n < 10000; ++n) {
            std::vector<std::string> v{"001", "2", "03", "255"};
            v[field] = std::to_string(n);
            if (!check(v[0]+"."+v[1]+"."+v[2]+"."+v[3])) return 1;
        }
        for (int n = 0; n < 256; ++n) {
            for (int width = 1; width != 5; ++width) {
                std::vector<std::string> v{"0", "1", "254", "255"};
                v[field] = std::to_string(n);
                while (v[field].size() < static_cast<size_t>(width)) v[field] = "0" + v[field];
                if (!check(v[0]+"."+v[1]+"."+v[2]+"."+v[3])) return 1;
            }
        }
    }
    const std::vector<std::string> invalid{"", ".", "...", "1.2.3", "1.2.3.4.5", "1..2.3", "1.2.3.", "1.2.3.4\n", " 1.2.3.4", "+1.2.3.4", std::string("1.2.3.4\0", 8)};
    for (const auto& ip : invalid) if (!check(ip)) return 1;
    std::mt19937 gen(20261016);
    const std::string chars = "0123456789.a:+- \t\n";
    for (int n = 0; n < 10000; ++n) {
        std::string ip;
        for (unsigned i = 0, len = gen() % 40; i < len; ++i) ip += chars[gen() % chars.size()];
        if (!check(ip)) return 1;
    }
    std::cout << "IPV4_REGEX_EQUIVALENCE_PASS cases=" << cases << std::endl;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.cpp"
            binary = Path(directory) / "check"
            path.write_text(source)
            subprocess.run([shutil.which("c++"), "-std=c++14", "-O2", "-D_GLIBCXX_USE_CXX11_ABI=1",
                            str(path), "-o", str(binary)], check=True, capture_output=True, text=True, timeout=45)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
            self.assertIn("IPV4_REGEX_EQUIVALENCE_PASS cases=", result.stdout)


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
