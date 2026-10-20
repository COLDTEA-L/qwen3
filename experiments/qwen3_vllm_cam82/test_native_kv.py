"""CPU-only native KV qualification checks; no Torch/NPU imports or allocations."""
import contextlib
import ast
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "adapter"))
from qwen_cam82 import native_kv as kv


def regions():
    def tensor(ptr):
        return NS(shape=(kv.NUM_BLOCKS, 64, 1, 128), dtype="torch.bfloat16",
                  data_ptr=lambda: ptr, element_size=lambda: 2,
                  is_contiguous=lambda: True, numel=lambda: kv.NUM_BLOCKS * kv.BLOCK_BYTES // 2)
    size = kv.NUM_BLOCKS * kv.BLOCK_BYTES
    return {f"model.layers.{i}.self_attn": [tensor(0x100000000 + 2 * size * i),
                                          tensor(0x100000000 + 2 * size * i + size)]
            for i in range(kv.NUM_LAYERS)}


def connector(fail=False):
    events = []
    worker = NS(vllm_config=NS(cache_config=NS(block_size=64, num_gpu_blocks_override=2048)),
                engine_id="test-native-worker", kv_recv_thread=None)

    def transfer(meta):
        events.append(("native_read", meta))
        if fail:
            raise RuntimeError("Mooncake transfer failed, ret: -1")

    def register(caches):
        events.append(("native_register", caches))
        worker.kv_recv_thread = NS(_transfer_kv_cache=transfer)

    worker.register_kv_caches = register
    worker.get_finished = lambda: (set(), {"request-1"})
    cls = type("MooncakeConnector", (), {"__module__": "vllm_ascend.distributed.mooncake_connector"})
    conn = cls()
    conn.connector_worker = worker
    return conn, events


class NativeKVTests(unittest.TestCase):
    def test_capacity_makes_both_native_packed_offsets_aligned(self):
        self.assertEqual(kv.NUM_BLOCKS * kv.BLOCK_BYTES, 32 * 1024 * 1024)
        self.assertEqual(kv.NUM_BLOCKS * kv.BLOCK_BYTES % kv.ALIGNMENT, 0)
        self.assertNotEqual(8212 * kv.BLOCK_BYTES % kv.ALIGNMENT, 0)
        self.assertEqual(kv.validate_regions(regions()), 96)

    def test_layout_config_and_alignment_refuse_wrong_inputs(self):
        for cache in (NS(block_size=128, num_gpu_blocks_override=2048),
                      NS(block_size=64, num_gpu_blocks_override=None)):
            with self.assertRaises(RuntimeError):
                kv.validate_config(NS(cache_config=cache))
        for attr, value in (("data_ptr", lambda: 0x100001000),
                            ("shape", (2048, 128, 1, 128)),
                            ("dtype", "torch.float16"),
                            ("is_contiguous", lambda: False)):
            caches = regions()
            setattr(caches["model.layers.0.self_attn"][1], attr, value)
            with self.assertRaises(RuntimeError):
                kv.validate_regions(caches)
        caches = regions()
        del caches["model.layers.0.self_attn"]
        with self.assertRaises(RuntimeError):
            kv.validate_regions(caches)
        caches = regions()
        caches["model.layers.1.self_attn"] = caches["model.layers.0.self_attn"]
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            kv.validate_regions(caches)

    def test_bad_registration_is_stopped_before_native_side_effects(self):
        conn, events = connector()
        kv.install(conn)
        caches = regions()
        caches["model.layers.0.self_attn"][1].data_ptr = lambda: 0x100001000
        with self.assertRaisesRegex(RuntimeError, "ALIGNMENT_FAILED"):
            conn.connector_worker.register_kv_caches(caches)
        self.assertEqual(events, [])

    def test_valid_cache_uses_same_native_views_and_read(self):
        conn, events = connector()
        guard = kv.install(conn)
        self.assertIs(kv.install(conn), guard)
        caches = regions()
        conn.connector_worker.register_kv_caches(caches)
        self.assertIs(events[0][1], caches)
        meta = dict(request_id="request-1", local_block_ids=[7], remote_block_ids=[10])
        conn.connector_worker.kv_recv_thread._transfer_kv_cache(meta)
        self.assertEqual(events[1], ("native_read", meta))
        self.assertEqual(conn.connector_worker.get_finished(), (set(), {"request-1"}))

    def test_transfer_failure_cannot_be_reported_as_finished(self):
        conn, _ = connector(fail=True)
        kv.install(conn)
        worker = conn.connector_worker
        worker.register_kv_caches(regions())
        with self.assertRaisesRegex(RuntimeError, "ret: -1"):
            worker.kv_recv_thread._transfer_kv_cache(
                dict(request_id="request-1", local_block_ids=[1], remote_block_ids=[1]))
        # Even if the pinned native handler has queued a false "done", it must
        # never reach the scheduler. The native cleanup/ack remains untouched.
        with self.assertRaisesRegex(RuntimeError, "KV_FAIL_STOP"):
            worker.get_finished()

    def test_pinned_native_handler_ack_stays_but_false_done_is_blocked(self):
        source = HERE.parents[2] / "vllm-ascend-082/vllm_ascend/distributed/mooncake_connector.py"
        if not source.is_file():
            self.skipTest("Pinned Ascend source snapshot absent")
        # Exercise the exact pinned exception/finally path, without importing
        # the native module or constructing a transport/device.
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "KVCacheRecvingThread")
        handler = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                       and n.name == "_handle_request")
        noop = lambda *args, **kwargs: None
        namespace = {"Any": object, "logger": NS(debug=noop, error=noop)}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), namespace)
        conn, _ = connector(fail=True)
        kv.install(conn)
        worker = conn.connector_worker
        worker.register_kv_caches(regions())
        receiver = worker.kv_recv_thread
        actions = []
        receiver.task_tracker = NS(update_done_task_count=lambda req: actions.append("native_done"))
        receiver._send_done_recv_signal = lambda *args: actions.append("native_ack")
        receiver.request_queue = NS(task_done=lambda: actions.append("queue_cleanup"))
        namespace["_handle_request"](receiver, dict(request_id="request-1", remote_host="host",
            remote_handshake_port=20010, local_block_ids=[1], remote_block_ids=[1]))
        self.assertEqual(actions, ["native_done", "native_ack", "queue_cleanup"])
        with self.assertRaisesRegex(RuntimeError, "KV_FAIL_STOP"):
            worker.get_finished()

    def test_failure_during_finished_collection_is_also_blocked(self):
        conn, _ = connector()
        worker = conn.connector_worker
        guard_holder = []

        def native_finished():
            guard_holder[0].fail({"request_id": "request-1"}, RuntimeError("read failed"))
            return set(), {"request-1"}

        worker.get_finished = native_finished
        guard_holder.append(kv.install(conn))
        with self.assertRaisesRegex(RuntimeError, "KV_FAIL_STOP"):
            worker.get_finished()

    def test_invalid_block_ids_do_not_reach_native_read(self):
        for local, remote in (([2048], [0]), ([0], [-1]), ([0, 1], [0])):
            conn, events = connector()
            kv.install(conn)
            worker = conn.connector_worker
            worker.register_kv_caches(regions())
            with self.assertRaisesRegex(RuntimeError, "block count/index"):
                worker.kv_recv_thread._transfer_kv_cache(
                    dict(request_id="request-1", local_block_ids=local, remote_block_ids=remote))
            self.assertEqual([e[0] for e in events], ["native_register"])

    def test_install_rejects_scheduler_or_other_connector(self):
        with self.assertRaises(RuntimeError):
            kv.install(NS(connector_worker=None))
        conn, _ = connector()
        conn.connector_worker = None
        with self.assertRaisesRegex(RuntimeError, "WORKER"):
            kv.install(conn)

    def test_old_adapter_preflight_and_current_docs(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("### 5.5", text)
        self.assertIn("--num-gpu-blocks-override 2048", text)
        self.assertIn("CAM82_KV_REGIONS_OK", text)
        self.assertIn("CAM82_KV_FAIL_STOP", text)
        self.assertNotIn("git push", text)
        spec = importlib.util.spec_from_file_location("launch_kv_test", HERE / "launch.py")
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        from unittest.mock import patch
        # No native imports/checks should occur with an old installed adapter.
        env = {"QWEN_CAM82_HOST": "228"}
        with patch.dict(launch.os.environ, env, clear=True), \
             patch.object(launch.md, "version", return_value="0.1.1"), \
             patch.object(sys, "argv", ["launch.py", "serve", "--role", "p-228"]), \
             contextlib.redirect_stderr(io.StringIO()) as output, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                launch.main()
        self.assertIn("README 5.5", output.getvalue())

    def test_short_report_distinguishes_registration_from_real_transfer(self):
        spec = importlib.util.spec_from_file_location("report_kv_test", HERE / "report_kv.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        log = ("COMMAND --block-size 64 --num-gpu-blocks-override 2048\n"
               "Registering KV cache: ptr=0x200000, length=33554432\n"
               "Registering KV cache: ptr=0x401000, length=33554432\n"
               "CAM82_KV_REGIONS_OK engine=test\n"
               "KV cache transfer for request good took 1.0 ms\n"
               "Mooncake transfer failed for request bad\n"
               "Mooncake transfer failed for request bad\n"
               "CAM82_KV_TRANSFER_FAILED request=bad RuntimeError\n"
               "CAM82_KV_FAIL_STOP\n")
        lines = report.summarize(log)
        self.assertIn("unaligned_records=1", lines[1])
        self.assertIn("request_ids_with_transfer_success=1", lines[2])
        self.assertIn("request_ids_with_transfer_failure=1", lines[2])
        self.assertIn("request_ids_with_transfer_success=0", report.summarize(log, "bad")[2])
        self.assertIn("registration_records=0", report.summarize("")[1])

    def test_report_uses_newest_launch_not_old_log_with_new_mtime(self):
        spec = importlib.util.spec_from_file_location("report_kv_order_test", HERE / "report_kv.py")
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "p-off-20261020-072806.log"
            new = Path(directory) / "p-off-20261020-095217.log"
            old.write_text("COMMAND --block-size 128\n")
            new.write_text("COMMAND --block-size 64 --num-gpu-blocks-override 2048\n")
            os.utime(old, (200, 200))
            os.utime(new, (100, 100))
            with patch.object(sys, "argv", ["report_kv.py", "--host", "228", "--results", directory]), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                report.main()
            self.assertIn(new.name, output.getvalue())
            self.assertNotIn(old.name, output.getvalue())


if __name__ == "__main__":
    unittest.main()
