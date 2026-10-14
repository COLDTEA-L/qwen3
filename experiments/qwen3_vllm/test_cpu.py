"""Dependency-free regressions. These do not validate NPU/Mooncake execution."""

import json
import contextlib
import io
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import runtime
import te_probe
import workload

HERE = Path(__file__).resolve().parent


class MigrationTests(unittest.TestCase):
    def test_solo_is_local_bf16_eager_without_kv(self):
        cmd, env = runtime.build_command("solo-228")
        self.assertNotIn("--kv-transfer-config", cmd)
        self.assertIn("--enforce-eager", cmd)
        self.assertEqual(cmd[cmd.index("--dtype") + 1], "bfloat16")
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertNotIn("sglang", " ".join(cmd))
        self.assertNotIn("--quantization", cmd)

    def test_pd4_roles_and_unique_engines(self):
        p = runtime.connector("p-228", "pd4")
        d = runtime.connector("d-238", "pd4")
        self.assertEqual(p["kv_role"], "kv_producer")
        self.assertEqual(d["kv_role"], "kv_consumer")
        self.assertNotEqual(p["engine_id"], d["engine_id"])
        self.assertIsInstance(p["kv_port"], int)
        self.assertEqual(p["kv_connector_extra_config"]["decode"], {"tp_size": 4, "dp_size": 1})

    def test_pd8_attention_tp_and_global_dp(self):
        for role, rank, devices in (("d8-228", "0", "4,5,6,7"), ("d8-238", "1", "0,1,2,3")):
            cmd, env = runtime.build_command(role, "pd8")
            self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "4")
            self.assertEqual(cmd[cmd.index("--data-parallel-size") + 1], "2")
            self.assertEqual(cmd[cmd.index("--data-parallel-rank") + 1], rank)
            self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], devices)
            self.assertNotIn("--headless", cmd)
        self.assertEqual(runtime.connector("p-228", "pd8")["kv_connector_extra_config"]["decode"]["dp_size"], 2)

    def test_incompatible_layouts_refused(self):
        for role, layout in (("d-238", "pd8"), ("d8-228", "pd4"), ("solo-228", "pd8")):
            with self.assertRaises(ValueError):
                runtime.build_command(role, layout)

    def test_bad_budgets_refused(self):
        for kwargs in ({"max_len": 0}, {"max_seqs": 0}, {"memory": 1}, {"model": "remote/model"}):
            with self.assertRaises(ValueError):
                runtime.build_command("p-228", **kwargs)

    def test_pool_is_explicit_multiconnector(self):
        cfg = runtime.connector("p-228", "pd4", pool=True)
        self.assertEqual(cfg["kv_connector"], "MultiConnector")
        self.assertEqual(cfg["kv_load_failure_policy"], "fail")
        children = cfg["kv_connector_extra_config"]["connectors"]
        self.assertEqual(children[0]["kv_connector"], "MooncakeConnectorV1")
        self.assertEqual(children[1]["kv_connector_extra_config"]["backend"], "mooncake")
        with self.assertRaises(ValueError):
            runtime.build_command("solo-228", pool=True)

    def test_config_private_small_dram_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime.write_configs(folder)
            cfg = json.loads((Path(folder) / "mooncake-dram-qualification.json").read_text())
            self.assertEqual(cfg["protocol"], "ascend")
            self.assertEqual(cfg["global_segment_size"], "64MB")
            self.assertEqual((Path(folder) / "deployment.env").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                runtime.write_configs(folder)

    def test_dry_run_never_imports_vllm_or_runs_device(self):
        result = subprocess.run([os.sys.executable, str(HERE / "runtime.py"), "serve", "--role", "p-228", "--dry-run"],
                                check=True, capture_output=True, text=True)
        self.assertIn("MooncakeConnectorV1", result.stdout)
        self.assertIn("COMMAND", result.stdout)

    def test_proxy_pd8_uses_two_decode_endpoints(self):
        result = subprocess.run([os.sys.executable, str(HERE / "runtime.py"), "proxy", "--layout", "pd8", "--dry-run"],
                                check=True, capture_output=True, text=True)
        self.assertIn("--decoder-hosts 10.155.205.228 10.155.205.238", result.stdout)
        self.assertIn("--decoder-ports 30011 30011", result.stdout)

    def test_environment_drops_sglang_and_proxy(self):
        text = (HERE / "env.sh").read_text()
        self.assertIn("unset PYTHONPATH", text)
        self.assertIn("unset ASCEND_MF_STORE_URL", text)
        self.assertIn("unset HTTP_PROXY", text)
        self.assertNotIn("../../python", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)

    def test_container_mounts_hccn_and_readonly_weights(self):
        result = subprocess.run(["bash", "-c", f'source "{HERE}/create_container.sh"; build_command test-image test-container /mock/npu-smi; printf "%s\\n" "${{container_command[@]}}"'],
                                check=True, capture_output=True, text=True)
        tokens = result.stdout.splitlines()
        self.assertIn("/etc/hccn.conf:/etc/hccn.conf:ro", tokens)
        self.assertIn("/ssd_2/yuanwenliu/models/Qwen3-30B-A3B:/ssd_2/yuanwenliu/models/Qwen3-30B-A3B:ro", tokens)
        self.assertNotIn("/var:/var", tokens)
        self.assertNotIn("/home:/home", tokens)

    def test_sse_usage_and_done(self):
        self.assertIsNone(workload.sse_json(b"data: [DONE]\n"))
        self.assertIsNone(workload.sse_json(b": heartbeat\n"))
        self.assertEqual(workload.sse_json(b'data: {"usage":{"completion_tokens":32}}\n')["usage"]["completion_tokens"], 32)

    def test_summary_counts_failures_not_just_success_latency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.jsonl"
            path.write_text(json.dumps({"success": True, "ttft_ms": 10, "e2e_ms": 20}) + "\n" +
                            json.dumps({"success": False, "error": "timeout"}) + "\n")
            report = workload.summary(path)
            self.assertEqual(report["requests"], 2)
            self.assertEqual(report["success"], 1)
            self.assertIn("warning", report)

    def test_te_descriptor_is_bounded_and_cluster_only(self):
        good = {"endpoint": "10.155.205.228:12345", "pointer": 1024, "bytes": te_probe.BYTES,
                "pattern": te_probe.PATTERN, "created_at": time.time()}
        self.assertEqual(te_probe.checked_descriptor(good), good)
        for update in ({"endpoint": "8.146.225.22:12345"}, {"bytes": 1 << 40}, {"pointer": 0}, {"created_at": 0}):
            with self.assertRaises(ValueError):
                te_probe.checked_descriptor({**good, **update})

    def run_mock_read(self, transfer_result):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "peer.json"
            path.write_text(json.dumps({"endpoint": "10.155.205.228:12345", "pointer": 1024,
                "bytes": te_probe.BYTES, "pattern": te_probe.PATTERN, "created_at": time.time()}))
            tensor = MagicMock()
            tensor.data_ptr.return_value = 2048
            tensor.cpu.return_value.__eq__.return_value.all.return_value = True
            engine = MagicMock()
            engine.initialize.return_value = engine.register_memory.return_value = 0
            engine.transfer_sync_read.return_value = transfer_result
            torch = SimpleNamespace(npu=MagicMock(), uint8="uint8", full=lambda *a, **k: tensor)
            mods = {"torch": torch, "torch_npu": SimpleNamespace(), "mooncake": SimpleNamespace(),
                    "mooncake.engine": SimpleNamespace(TransferEngine=lambda: engine)}
            with patch.dict(os.sys.modules, mods), patch.dict(os.environ, {"QWEN_VLLM_HOST": "238"}), \
                 patch.object(os.sys, "argv", ["te_probe.py", "read", "--descriptor", str(path)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                if transfer_result:
                    with self.assertRaisesRegex(RuntimeError, "READ failed"):
                        te_probe.main()
                else:
                    te_probe.main()
            engine.initialize.assert_called_once_with("10.155.205.238", "P2PHANDSHAKE", "ascend", "")
            engine.transfer_sync_read.assert_called_once_with("10.155.205.228:12345", 2048, 1024, te_probe.BYTES)
            engine.unregister_memory.assert_called_once_with(2048)

    def test_mock_npu_read_success_and_own_buffer_cleanup(self):
        self.run_mock_read(0)

    def test_mock_npu_read_failure_propagates(self):
        self.run_mock_read(-1)

    def test_cam_mode_refuses_to_execute_vllm(self):
        with patch.dict(os.environ, {"QWEN_VLLM_HOST": "228", "QWEN_CAM_MODE": "cam-detour"}), \
             patch.object(os.sys, "argv", ["runtime.py", "serve", "--role", "p-228"]), \
             patch.object(os, "execvpe") as execute, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runtime.main()
            execute.assert_not_called()

    def test_wrong_host_refuses_to_execute_vllm(self):
        with patch.dict(os.environ, {"QWEN_VLLM_HOST": "238"}), \
             patch.object(os.sys, "argv", ["runtime.py", "serve", "--role", "p-228"]), \
             patch.object(os, "execvpe") as execute, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runtime.main()
            execute.assert_not_called()

    def test_readme_does_not_publish_from_experiment_hosts(self):
        text = (HERE.parent / "qwen3_cam/README.md").read_text()
        self.assertNotRegex(text, r"git\s+push")
        self.assertIn("Host DRAM", text)
        self.assertIn("AllGather", text)

    def test_pinned_source_contracts_when_available(self):
        # Optional local source audit; no vLLM imports and no NPU initialization.
        ascend = HERE.parents[2] / "vllm-ascend"
        vllm = HERE.parents[2] / "vllm"
        mooncake = HERE.parents[2] / "Mooncake"
        if not all(x.is_dir() for x in (ascend, vllm, mooncake)):
            self.skipTest("Fixed upstream source snapshots are not present")
        args_text = (vllm / "vllm/engine/arg_utils.py").read_text()
        for flag in ("data_parallel_rank", "data_parallel_address", "data_parallel_backend", "max_num_batched_tokens"):
            self.assertIn(flag, args_text)
        registry = (ascend / "vllm_ascend/distributed/kv_transfer/__init__.py").read_text()
        self.assertIn('"MooncakeConnectorV1"', registry)
        self.assertIn('"AscendStoreConnector"', registry)
        context = (ascend / "vllm_ascend/ascend_forward_context.py").read_text()
        self.assertIn("ep_world_size >= 16", context)
        allocator = (mooncake / "mooncake-store/src/utils.cpp").read_text()
        self.assertIn("aclrtMallocHost", allocator)


if __name__ == "__main__":
    unittest.main()
