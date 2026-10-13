"""Dependency-free host tests. These do NOT validate Ascend device kernels."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cam_adapter_test", ROOT / "python/sglang/srt/distributed/device_communicators/cam_detour.py")
cam = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cam
spec.loader.exec_module(cam)
spec2 = importlib.util.spec_from_file_location("replay_analysis", Path(__file__).with_name("analyze_replay.py"))
analysis = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(analysis)
spec3 = importlib.util.spec_from_file_location("serving_workload", Path(__file__).with_name("serving_workload.py"))
workload = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(workload)
spec4 = importlib.util.spec_from_file_location("runtime_config", Path(__file__).with_name("configure_runtime.py"))
runtime_config = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(runtime_config)


class CamTests(unittest.TestCase):
    def test_inference_env_uses_role_ip_and_removes_download_proxy(self):
        source = Path(__file__).with_name("env.sh").read_text()
        # Execute the actual environment block, omitting CANN sourcing and
        # directory creation. No real proxy credentials enter test output.
        block = source[source.index("unset RANK"):source.index("mkdir -p")]
        render = block + '\nprintf "%s\\0" "$SGLANG_HOST_IP" "$HCCL_SOCKET_IFNAME" "${HTTP_PROXY-unset}" "${all_proxy-unset}" "$NO_PROXY" "$ASCEND_MF_TRANSFER_PROTOCOL"'
        environment = {**os.environ, "HOST_228_IP": "192.0.2.228", "HOST_238_IP": "192.0.2.238",
            "HOST_IF_228": "nic228", "HOST_IF_238": "nic238", "_qwen_exp_dir": "/mock/experiments/qwen3_cam",
            "HTTP_PROXY": "mock-proxy", "all_proxy": "mock-proxy"}
        for role in ("228", "238"):
            result = subprocess.run(["bash", "-c", render], env={**environment, "_qwen_role": role}, check=True, capture_output=True)
            values = result.stdout.decode().rstrip("\0").split("\0")
            self.assertEqual(values[:4], ["192.0.2." + role, "nic" + role, "unset", "unset"])
            self.assertIn("192.0.2.228", values[4])
            self.assertIn("192.0.2.238", values[4])
            self.assertEqual(values[5], "device_rdma")

    def device_buffer_expression(self):
        # Execute the production initializer's generator without importing the
        # full Torch/NPU stack. A real device run remains required on A2.
        source = ROOT / "python/sglang/srt/mem_cache/pool_host/mha.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MHATokenToKVPoolHost")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        call = next(n for n in ast.walk(init) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "_init_device_row_stride")
        return compile(ast.Expression(call.args[0]), str(source), "eval")

    def test_npu_tensor_buffers_do_not_use_truthiness(self):
        class TensorLike:
            def __init__(self, layers):
                self.layers = layers
            def __bool__(self):
                raise RuntimeError("Boolean value of Tensor is ambiguous")
            def __iter__(self):
                return iter(self.layers)
        k, v = [object(), object()], [object(), object()]
        main = SimpleNamespace(k_buffer=TensorLike(k), v_buffer=TensorLike(v))
        draft = SimpleNamespace(k_buffer=(k[0],), v_buffer=[v[0]])
        owner = SimpleNamespace(device_pool=main, mtp_draft_device_pools=(None, draft))
        result = list(eval(self.device_buffer_expression(), {"self": owner}))
        self.assertEqual(result, k + v + [k[0], v[0]])

    def test_device_buffers_none_empty_list_and_tuple(self):
        pools = (SimpleNamespace(k_buffer=None, v_buffer=[]), SimpleNamespace(k_buffer=(), v_buffer=None))
        owner = SimpleNamespace(device_pool=pools[0], mtp_draft_device_pools=pools[1:])
        self.assertEqual(list(eval(self.device_buffer_expression(), {"self": owner})), [])

    def test_transfer_engine_failure_preserves_native_context(self):
        source = ROOT / "python/sglang/srt/disaggregation/ascend/transfer_engine.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendTransferEngine")
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "initialize")
        globals_ = {"os": os, "logger": MagicMock(),
                    "memfabric_hybrid": SimpleNamespace(get_last_err_msg=lambda: "mock RDMA error")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), globals_)
        engine = SimpleNamespace(initialize=lambda *args: 17)
        owner = SimpleNamespace(engine=engine, store_url="tcp://192.0.2.228:7100",
            session_id="192.0.2.228:1234", role="Prefill", npu_id=1,
            _get_transfer_protocol=lambda: "sdma", _resolve_trans_op_type=lambda p: p)
        context = SimpleNamespace(get_parallel=lambda: None)
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}), patch.dict(os.environ, {"SGLANG_ASCEND_MF_LOG_LEVEL": "1"}):
            with self.assertRaisesRegex(RuntimeError, "ret=17.*native_error=mock RDMA error"):
                globals_["initialize"](owner)
            engine.initialize = lambda *args: 0
            globals_["initialize"](owner)

    def test_pd_store_wrapper_success_failure_and_url(self):
        spec = importlib.util.spec_from_file_location("pd_store_test", Path(__file__).with_name("start_pd_store.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.store_address("tcp://192.0.2.228:7100"), ("192.0.2.228", 7100))
        for url in ("https://192.0.2.228:7100", "tcp://user:pass@192.0.2.228:7100", "tcp://192.0.2.228:7100/path"):
            with self.assertRaises(ValueError):
                module.store_address(url)
        mf = SimpleNamespace(create_config_store=MagicMock(return_value=0), set_log_level=MagicMock(), get_last_err_msg=lambda: "mock store error")
        with patch.object(module.socket, "socket", return_value=MagicMock()), patch.object(module.socket, "create_connection", return_value=MagicMock()):
            module.start_store("tcp://192.0.2.228:7100", mf)
            mf.create_config_store.assert_called_once()
            mf.create_config_store.return_value = 9
            with self.assertRaisesRegex(RuntimeError, "ret=9.*mock store error"):
                module.start_store("tcp://192.0.2.228:7100", mf)

    def container_arguments(self, firmware):
        helper = Path(__file__).parent / "docker/create_container.sh"
        program = 'source "$1"; build_container_command image:test test-container /usr/local/bin/npu-smi "$2"; printf "%s\\0" "${container_command[@]}"'
        result = subprocess.run(["bash", "-c", program, "test", str(helper), firmware],
                                check=True, capture_output=True)
        return result.stdout.decode().rstrip("\0").split("\0")

    def test_container_profile(self):
        args = self.container_arguments("/usr/local/Ascend/firmware")
        self.assertIn("--privileged", args)
        self.assertEqual(args[args.index("--runtime")+1], "ascend")
        self.assertEqual(args[args.index("--ipc")+1], "host")
        self.assertEqual(args[args.index("--workdir")+1], "/ssd_2/yuanwenliu/sglang")
        self.assertIn("ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", args)
        devices = [args[i+1] for i,x in enumerate(args) if x == "--device"]
        self.assertEqual(len(devices), 11)
        self.assertEqual(devices[:8], [f"/dev/davinci{i}" for i in range(8)])
        self.assertIn("/usr/local/dcmi:/usr/local/dcmi:ro", args)
        self.assertIn("/var/log/npu:/usr/slog:rw", args)
        self.assertIn("/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro", args)

    def test_container_mount_scope(self):
        args = self.container_arguments("")
        mounts = [args[i+1] for i,x in enumerate(args) if x == "-v"]
        self.assertNotIn("/ssd_2:/ssd_2", mounts)
        self.assertFalse(any(m.split(":",1)[0] in ("/home", "/data", "/var") for m in mounts))
        self.assertFalse(any("opensora" in m.lower() or "open-sora" in m.lower() for m in mounts))
        self.assertFalse(any("/firmware" in m for m in mounts))
        self.assertIn("/ssd_2/yuanwenliu/sglang:/ssd_2/yuanwenliu/sglang:rw", mounts)

    def test_runtime_config(self):
        result = runtime_config.render("192.0.2.228", "192.0.2.238", "testuser")
        pool, client = (json.loads(result[name]) for name in ("pool.json", "client.json"))
        self.assertEqual(pool["hbm_size"], "8GB")
        self.assertEqual(client["hbm_size"], 0)
        self.assertEqual(client["dram_size"], 0)
        self.assertEqual(client["meta_service_url"], "tcp://192.0.2.238:5000")
        self.assertIn("export HOST_228_IP=192.0.2.228", result["deployment.env"])
        self.assertIn("export HOST_238_IP=192.0.2.238", result["deployment.env"])
        self.assertNotIn("8.146.225.22", "".join(result.values()))
        with self.assertRaises(ValueError):
            runtime_config.render("192.0.2.228", "192.0.2.228", "testuser")

    def test_runtime_write_guard(self):
        configs = runtime_config.render("192.0.2.228", "192.0.2.238", "testuser")
        with tempfile.TemporaryDirectory() as directory:
            runtime_config.write_configs(directory, configs)
            path = Path(directory) / "client.json"
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            original = path.read_text()
            with self.assertRaises(FileExistsError):
                runtime_config.write_configs(directory, configs)
            self.assertEqual(path.read_text(), original)

    def test_runtime_cli(self):
        program = Path(__file__).with_name("configure_runtime.py")
        result = subprocess.run([sys.executable, str(program), "--help"],
                                check=True, capture_output=True, text=True)
        self.assertIn("--ip-228", result.stdout)
        self.assertIn("--ip-238", result.stdout)
        self.assertNotIn("--a-ip", result.stdout)

    def test_personal_paths(self):
        configs = runtime_config.render("192.0.2.228", "192.0.2.238", "testuser")
        text = configs["deployment.env"]
        self.assertIn("/ssd_2/yuanwenliu/models/Qwen3-30B-A3B", text)
        self.assertIn("/ssd_2/yuanwenliu/umdk-qwen3-cam", text)
        args = self.container_arguments("")
        sources = [args[i+1].split(":",1)[0] for i,x in enumerate(args) if x == "-v"]
        work = [p for p in sources if p.startswith("/ssd_2/")]
        self.assertEqual(len(work), 4)
        self.assertTrue(all(p.startswith("/ssd_2/yuanwenliu/") for p in work))

    def test_launch_roles(self):
        launch = Path(__file__).with_name("launch.sh")
        # Render the final argument arrays instead of executing Python/model code.
        source = launch.read_text()
        invocation = 'exec python -m sglang.launch_server "${common[@]}" "${args[@]}" "$@"'
        self.assertEqual(source.count(invocation), 1)
        source = source.replace(invocation, 'printf "%s\\0" "${common[@]}" "${args[@]}" "$@"')
        environment = {**os.environ, "MODEL_PATH":"/ssd_2/yuanwenliu/models/Qwen3-30B-A3B",
                       "HOST_228_IP":"192.0.2.228", "ASCEND_MF_STORE_URL":"tcp://192.0.2.228:7100"}
        for role, tp, ep, base, node in (("p-228", "4", "4", "0", "0"),
                                       ("d-228", "8", "8", "4", "0"),
                                       ("d-238", "8", "8", "0", "1")):
            result = subprocess.run(["bash", "-c", source, str(launch), role],
                                    env=environment, check=True, capture_output=True)
            args = result.stdout.decode().rstrip("\0").split("\0")
            for flag, value in (("--tp-size",tp),("--ep-size",ep),
                                ("--base-gpu-id",base),("--node-rank",node)):
                self.assertEqual(args[args.index(flag)+1], value)
            if role.startswith("d-"):
                self.assertEqual(args[args.index("--dist-init-addr")+1], "192.0.2.228:29500")
                self.assertEqual(args[args.index("--attn-dp-size")+1], "2")
                self.assertEqual(args[args.index("--disaggregation-decode-retraction-backup")+1], "cpu_tensor")

    def test_role_mappings(self):
        for filename, ranks, expected in (
            ("p_228", range(4), (0, 1, 2, 3)),
            ("d_228", range(4), (4, 5, 6, 7)),
            ("d_238", range(4, 8), (0, 1, 2, 3)),
        ):
            cfg = cam.CamConfig.from_dict(json.loads((Path(__file__).parent / f"configs/{filename}.json").read_text()))
            self.assertEqual(cfg.cam_group(ranks), expected)
            self.assertIsNone(cfg.cam_group(range(8)))
            self.assertIsNone(cfg.cam_group(reversed(tuple(ranks))))

    def test_memory_guard(self):
        data = {"process_rank_to_cam": {str(r): r for r in range(4)}, "groups": [list(range(4))]}
        for override in ({"buffer_mb": 128}, {"buffer_mb": 2001}, {"chunk_bytes": 2 ** 31},
                         {"physical_size": 16}, {"comm_id": 64}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                cam.CamConfig.from_dict({**data, **override})

    def test_invalid_groups(self):
        for mapping, groups in (
            ({"0": 0, "1": 0, "2": 2, "3": 3}, [[0, 1, 2, 3]]),
            ({"0": 0}, [[0, 1, 2, 3]]),
            ({str(r): r for r in range(4)}, [[0, 1, 2, 3], [0, 1, 2, 3]]),
        ):
            with self.assertRaises(ValueError):
                cam.CamConfig.from_dict({"process_rank_to_cam": mapping, "groups": groups})

    def test_chunk_alignment(self):
        for element_size in (2, 4):
            count, capacity = cam.padded_shard_elements(2 ** 24, element_size, 8 * 1024 * 1024)
            self.assertEqual(capacity % 640, 0)
            self.assertLessEqual(capacity * 4 * element_size, 8 * 1024 * 1024)

    def test_reduce_scatter_all_gather_algebra(self):
        for n in (1, 3, 4, 7, 259, 2560, 5128):
            inputs = [[100 * r + i for i in range(n)] for r in range(4)]
            padded_n = (n + 3) // 4 * 4
            padded = [x + [0] * (padded_n - n) for x in inputs]
            shard = padded_n // 4
            result = []
            for destination in range(4):
                # AllToAll receive source-rank-ordered slices, then local sum.
                received = [x[destination * shard:(destination + 1) * shard] for x in padded]
                result.extend(sum(values) for values in zip(*received))
            self.assertEqual(result[:n], [sum(values) for values in zip(*inputs)])

    def test_analysis(self):
        base = dict(mode="cam-detour", kv_target="228", tp_mib=8, moe_mib=8, kv_mib=64,
                    rank=4, flow="moe")
        result = analysis.summarize([{**base, "case": "moe", "samples_ms": [1, 1, 1]},
                                     {**base, "case": "moe+kv", "samples_ms": [2, 2, 2]}])
        self.assertEqual(result[1]["p95_slowdown"], 2.0)
        with self.assertRaises(ValueError):
            analysis.summarize([{**base, "case": "moe+kv", "samples_ms": [2]}])

    def test_fixed_arrivals(self):
        plan = workload.arrival_plan(30, 0.5, 10, 8)
        self.assertEqual(sum(kind == "background" for _, kind, _ in plan), 15)
        self.assertEqual(sum(kind == "restore" for _, kind, _ in plan), 16)
        self.assertEqual(sorted({when for when, kind, _ in plan if kind == "restore"}), [10, 20])
        self.assertEqual(plan, sorted(plan))
        with self.assertRaises(ValueError):
            workload.arrival_plan(30, 0, 10, 8)


if __name__ == "__main__":
    unittest.main()
