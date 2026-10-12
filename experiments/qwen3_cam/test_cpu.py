"""Dependency-free host tests. These do NOT validate Ascend device kernels."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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
        self.assertIn("/ssd_2/sglang:/ssd_2/sglang:rw", mounts)

    def test_runtime_config(self):
        result = runtime_config.render("192.0.2.1", "192.0.2.2", "testuser")
        pool, client = (json.loads(result[name]) for name in ("pool.json", "client.json"))
        self.assertEqual(pool["hbm_size"], "8GB")
        self.assertEqual(client["hbm_size"], 0)
        self.assertEqual(client["dram_size"], 0)
        self.assertEqual(client["meta_service_url"], "tcp://192.0.2.2:5000")
        with self.assertRaises(ValueError):
            runtime_config.render("192.0.2.1", "192.0.2.1", "testuser")

    def test_runtime_write_guard(self):
        configs = runtime_config.render("192.0.2.1", "192.0.2.2", "testuser")
        with tempfile.TemporaryDirectory() as directory:
            runtime_config.write_configs(directory, configs)
            path = Path(directory) / "client.json"
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            original = path.read_text()
            with self.assertRaises(FileExistsError):
                runtime_config.write_configs(directory, configs)
            self.assertEqual(path.read_text(), original)

    def test_role_mappings(self):
        for filename, ranks, expected in (
            ("p_a", range(4), (0, 1, 2, 3)),
            ("d_a", range(4), (4, 5, 6, 7)),
            ("d_b", range(4, 8), (0, 1, 2, 3)),
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
        base = dict(mode="cam-detour", kv_target="a", tp_mib=8, moe_mib=8, kv_mib=64,
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
