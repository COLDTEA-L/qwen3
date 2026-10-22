"""CPU-only tests: rank consistency, pairing, corrupted/incomplete results.

Fixtures below are synthetic TEST DATA, never hardware performance evidence.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from spec import CONDITIONS, describe, digest, kv_layout, validate
from analyze import analyze, markdown
import capture_load


def traces():
    result = []
    # Deliberately nonuniform routing; recv splits must be transposed.
    matrix = [[(r + c) % 3 for c in range(8)] for r in range(8)]
    for rank in range(8):
        send, recv = matrix[rank], [matrix[r][rank] for r in range(8)]
        def op(phase, ins, outs):
            return {"kind": "all_to_all", "phase": phase, "shape": [sum(ins), 2048],
                    "dtype": "bfloat16", "input_splits": ins, "output_splits": outs}
        result.append({"schema": 1, "origin": "native_decode_capture", "ep_size": 8,
            "ep_rank": rank, "host": "228" if rank < 4 else "238",
            "physical_device": rank + 4 if rank < 4 else rank - 4,
            "skip_pairs": 192, "source_hashes": {"vllm_ascend.ops.moe.token_dispatcher": "a" * 64,
                "vllm_ascend.ops.moe.comm_utils": "b" * 64}, "global_ranks": list(range(8)),
            "pairs": [[{"kind": "all_gather", "phase": "dispatch", "shape": [128], "dtype": "float32",
                        "input_splits": None, "output_splits": None},
                       op("dispatch", send, recv), op("combine", recv, send)]]})
    return result


def fixture(root):
    config = {"iterations": 1, "repeats": 5, "kv_transfers": 1, "profile": False}
    info = validate(traces())
    signature = digest({"config": config, "trace": info})
    for rank in range(12):
        rows = [{"type": "header", "schema": 1, "rank": rank, "host": "228" if rank < 8 else "238",
                 "config": config, "trace": info, "signature": signature},
                {"type": "preflight", "moe_payload_ok": rank >= 4, "kv_payload_ok": rank >= 8}]
        for repeat in range(5):
            for ci, condition in enumerate(CONDITIONS):
                # Host228 deliberately has a different clock epoch.
                base = (repeat * 10 + ci + 1) * 10**9 + (10**14 if rank < 8 else 0)
                if rank >= 4 and condition != "K":
                    for op in range(3):
                        duration = 1.2 if condition == "MK" else 1.0
                        begin = base + (op + 1) * 10**6
                        rows.append({"type": "moe", "repeat": repeat, "condition": condition,
                            "iteration": 0, "pair": 0, "op": op, "phase": ("metadata", "dispatch", "combine")[op],
                            "start_ns": begin, "end_ns": begin + int(duration * 1e6),
                            "wall_ms": duration, "stream_ms": duration / 2})
                if rank >= 8 and condition != "M":
                    begin = base + (20_000_000 if condition == "SERIAL" else 0)
                    duration = 15 if condition == "MK" else 10
                    rows.append({"type": "kv", "repeat": repeat, "condition": condition,
                        "transfer": 0, "start_ns": begin, "end_ns": begin + duration * 10**6,
                        "wall_ms": duration, "bytes": 24 * 2**20})
                rows.append({"type": "phase", "repeat": repeat, "condition": condition,
                    "start_ns": base, "work_end_ns": base + 40_000_000, "end_ns": base + 50_000_000,
                    "phase_ms": 50, "payload_ok": True})
        rows.append({"type": "complete", "ok": True})
        (root / f"rank{rank:02d}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


class ReplayTests(unittest.TestCase):
    def test_dp128_document_matches_capture_and_replay_parameters(self):
        doc = (Path(__file__).parent / "LOAD_SWEEP.md").read_text()
        capture = doc.split("## 4 ", 1)[1].split("## 5 ", 1)[0]
        replay = doc.split("## 6 ", 1)[1].split("## 7 ", 1)[0]
        self.assertEqual(capture.count("--max-num-seqs 128 --max-model-len 1024"), 2)
        self.assertIn("--requests-per-dp 128 --tokens 128", capture)
        self.assertIn('"requests": 256, "requests_per_dp": 128', capture)
        for host in (228, 238):
            line = next(x for x in replay.splitlines() if f"run.py --host {host}" in x)
            self.assertIn("capture-dp128-01", line)
            self.assertIn("--run-id dp128-kv2048-n500-01", line)
            self.assertIn("--kv-transfers 500 --repeats 5 --prefix-tokens 2048", line)
        self.assertNotIn("capture-dp32-01", replay)
        self.assertNotRegex(doc, r"git\s+push")

    def test_capture_concurrency_targets_both_dp_without_serial_submissions(self):
        def fake_request(url, tokens, index, gate):
            # If fewer than all requested workers are active this times out.
            gate.wait(timeout=5)
            return (url, tokens, index)
        with patch.object(capture_load, "request", side_effect=fake_request):
            result = capture_load.collect(128, 128)
        self.assertEqual(len(result), 256)
        for host in (228, 238):
            self.assertEqual(sorted(i for url, tokens, i in result
                                    if url == f"http://10.155.205.{host}:30011"), list(range(128)))
        for tokens, requests in ((15, 8), (513, 8), (128, 0), (128, 129)):
            with self.assertRaises(ValueError):
                capture_load.collect(tokens, requests)

    def test_measured_payload_description_does_not_change_trace_signature(self):
        data = traces()
        before = validate(data)
        detail = describe(data)
        for key in before:
            self.assertEqual(detail[key], before[key])
        self.assertEqual(detail["cross_host_kib_per_dispatch_combine_pair"],
                         before["cross_host_payload_bytes_per_window"] / before["pairs"] / 1024)
        self.assertLessEqual(detail["dispatch_input_rows_per_rank_min"],
                             detail["dispatch_input_rows_per_rank_max"])

    def test_report_exposes_overlap_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(Path(tmp))
            report = markdown(analyze(tmp))
        self.assertIn("配对 Dispatch", report)
        self.assertIn("100.00%", report)
        self.assertIn("KiB", report)

    def test_nonuniform_native_shape_and_cross_host_bytes(self):
        info = validate(traces())
        self.assertGreater(info["cross_host_payload_bytes_per_window"], 0)

    def test_reject_wrong_receiver_splits(self):
        data = traces()
        data[7]["pairs"][0][1]["output_splits"][0] += 1
        with self.assertRaises(ValueError):
            validate(data)

    def test_reject_shifted_pair_sequence_even_with_inverse_layouts(self):
        data = traces()
        for op in data[0]["pairs"][0][1:]:
            op["input_splits"] = list(reversed(op["input_splits"]))
            op["output_splits"] = list(reversed(op["output_splits"]))
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            validate(data)

    def test_reject_incomplete_capture_or_wrong_physical_rank(self):
        with self.assertRaises(ValueError):
            validate(traces()[:-1])
        data = traces()
        data[4]["physical_device"] = 4
        with self.assertRaises(ValueError):
            validate(data)

    def test_recorded_integer_histogram_dtype_is_preserved(self):
        data = traces()
        for trace in data:
            trace["pairs"][0][0]["dtype"] = "int64"
        validate(data)
        data[0]["pairs"][0][0]["dtype"] = "int32"
        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            validate(data)

    def test_kv_padding_and_volume(self):
        self.assertEqual(kv_layout(1024)["bytes_per_request"], 96 * 2**20)
        self.assertEqual(kv_layout(65), kv_layout(128))
        with self.assertRaises(ValueError):
            kv_layout(4096)

    def test_known_paired_slowdown_and_separate_host_clocks(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fixture(root)
            result = analyze(root)
            self.assertAlmostEqual(result["summary"]["dispatch_slowdown_pct"]["p50"], 20)
            self.assertAlmostEqual(result["summary"]["kv_slowdown_pct"]["p50"], 50)
            self.assertEqual(result["summary"]["host_submission_overlap_fraction"]["p50"], 1)
            self.assertIn("+20.00%", markdown(result))

    def test_cli_dry_run_and_analysis_without_torch(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            captured = root / "trace"
            captured.mkdir()
            for record in traces():
                (captured / f'ep{record["ep_rank"]}.json').write_text(json.dumps(record))
            dry = subprocess.run([sys.executable, str(here / "run.py"), "--host", "228",
                "--trace-dir", str(captured), "--output-dir", str(root / "unused"),
                "--run-id", "cpu-test", "--dry-run"], capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse((root / "unused").exists())
            measured = root / "fixture-not-real-measurements"
            measured.mkdir()
            fixture(measured)
            output = root / "analysis"
            result = subprocess.run([sys.executable, str(here / "analyze.py"), "--run-dir", str(measured),
                                     "--output-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "paired_rounds.csv").is_file())
            self.assertIn("+20.00%", (output / "report.md").read_text())
            self.assertEqual(json.loads((output / "analysis.json").read_text())["summary"]["kv_slowdown_pct"]["p50"], 50)

    def test_refuse_failed_missing_or_duplicate_samples(self):
        for mode in ("footer", "duplicate", "mixed_signature", "rank"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                fixture(root)
                file = root / "rank08.jsonl"
                rows = [json.loads(x) for x in file.read_text().splitlines()]
                if mode == "footer":
                    rows.pop()
                elif mode == "duplicate":
                    rows.insert(3, copy.deepcopy(rows[2]))
                elif mode == "mixed_signature":
                    rows[0]["signature"] = "wrong"
                else:
                    rows[0]["rank"] = 9
                file.write_text("".join(json.dumps(x) + "\n" for x in rows))
                with self.assertRaises(ValueError):
                    analyze(root)

    def test_opt_in_capture_skips_prefill_profile_and_preserves_native_return(self):
        module_path = Path(__file__).resolve().parents[1] / "adapter/qwen_cam82/moe_trace.py"
        spec = importlib.util.spec_from_file_location("trace_under_test", module_path)
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        with patch.dict(os.environ, {}, clear=True):
            capture.install()  # Disabled capture must not import Torch/vLLM.
        names = ["torch", "torch.distributed", "vllm", "vllm.forward_context",
                 "vllm.distributed", "vllm.distributed.parallel_state", "vllm_ascend",
                 "vllm_ascend.ops", "vllm_ascend.ops.moe", "vllm_ascend.ops.moe.token_dispatcher",
                 "vllm_ascend.ops.moe.comm_utils"]
        modules = {name: ModuleType(name) for name in names}
        for name, mod in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, mod)
        context = NS(with_prefill=True, in_profile_run=False)
        ep = NS(world_size=8, rank_in_group=0, device_group=object())
        modules["torch"].npu = NS(current_device=lambda: 4)
        modules["torch.distributed"].get_process_group_ranks = lambda group: list(range(8))
        modules["vllm.forward_context"].get_forward_context = lambda: context
        modules["vllm.distributed.parallel_state"].get_ep_group = lambda: ep
        dispatch_mod = modules["vllm_ascend.ops.moe.token_dispatcher"]
        native_calls = []
        def native_a2a(tensor, out, ins, group, event=None):
            native_calls.append((tensor, out, ins, group, event))
            return "native-a2a-result"
        dispatch_mod.async_all_to_all = native_a2a
        dispatch_mod.gather_from_sequence_parallel_region = lambda *args: "native-gather-result"
        counts = NS(shape=(128,), dtype="torch.int64")
        hidden = NS(shape=(8, 2048), dtype="torch.bfloat16")
        class Dispatcher:
            def token_dispatch(self, with_quant=False):
                dispatch_mod.gather_from_sequence_parallel_region(counts, ep.device_group)
                return dispatch_mod.async_all_to_all(hidden, [1] * 8, [1] * 8, ep.device_group)
            def token_combine(self):
                return dispatch_mod.async_all_to_all(hidden, [1] * 8, [1] * 8, ep.device_group)
        dispatch_mod.TokenDispatcherWithAll2AllV = Dispatcher
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for key in ("vllm_ascend.ops.moe.token_dispatcher", "vllm_ascend.ops.moe.comm_utils"):
                file = root / (key + ".py")
                file.write_text("# mock source for CPU capture test\n")
                modules[key].__file__ = str(file)
            env = {"QWEN_MOE_CAPTURE_DIR": str(root), "QWEN_CAM82_ROLE": "d8-228",
                   "QWEN_CAM82_MODE": "off", "QWEN_CAM82_HOST": "228",
                   "QWEN_MOE_CAPTURE_SKIP_PAIRS": "1", "QWEN_MOE_CAPTURE_PAIRS": "1"}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear=True):
                capture.install()
                instance = Dispatcher()
                for prefill, profile in ((True, False), (False, True), (False, False)):
                    context.with_prefill, context.in_profile_run = prefill, profile
                    self.assertEqual(instance.token_dispatch(), "native-a2a-result")
                    self.assertEqual(instance.token_combine(), "native-a2a-result")
                self.assertFalse((root / "ep0.json").exists())
                instance.token_dispatch()
                instance.token_combine()
                data = json.loads((root / "ep0.json").read_text())
                self.assertEqual(len(data["pairs"]), 1)
                self.assertEqual(data["pairs"][0][0]["dtype"], "int64")
                self.assertEqual([x["kind"] for x in data["pairs"][0]], ["all_gather", "all_to_all", "all_to_all"])
                self.assertEqual(len(native_calls), 8)
                self.assertTrue(all(call[0] is hidden and call[3] is ep.device_group for call in native_calls))
                with self.assertRaisesRegex(RuntimeError, "twice"):
                    capture.install()


if __name__ == "__main__":
    unittest.main()
