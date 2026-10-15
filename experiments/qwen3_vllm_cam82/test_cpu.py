"""CPU/mock regressions. Numeric tests use NumPy, never Torch or an NPU."""
import contextlib
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
            (project / "api.cpp").write_text('#include "aclnn/opdev/platform.h"\nint api() { return platform_fixture(); }\n')
            (project / "other.cpp").write_text('int other() { return 1; }\n')
            cmake_file = project / "CMakeLists.txt"
            cmake_file.write_text('''cmake_minimum_required(VERSION 3.12)
project(cam82_header_fixture LANGUAGES CXX)
set(ASCEND_AUTOGEN_PATH "${CMAKE_SOURCE_DIR}/autogen")
file(MAKE_DIRECTORY "${ASCEND_AUTOGEN_PATH}")
file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
add_library(cust_opapi SHARED api.cpp)
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
            for item in commands:
                self.assertEqual("include/experiment/platform" in item["command"],
                                 Path(item["file"]).name == "api.cpp")
            (platform / "platform_info.h").unlink()
            missing = subprocess.run(configure, env=env, capture_output=True, text=True, timeout=45)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("CAM82 missing SDK header", missing.stderr)


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
