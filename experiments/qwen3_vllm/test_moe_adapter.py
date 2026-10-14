"""CPU-only adapter regressions. Never initialize Torch or NPU."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "moe_adapter"))
from qwen_moe_adapter import policy
from qwen_moe_adapter.patching import make_prepare_class
from qwen_moe_adapter import patching
import moe_probe
import runtime
import compare_outputs


def config(dp=1):
    return NS(additional_config={"qwen_moe_comm": "alltoall"},
        model_config=NS(hf_text_config=NS(model_type="qwen3_moe", num_experts=128,
            num_experts_per_tok=8), dtype="torch.bfloat16", quantization=None, enforce_eager=True),
        compilation_config=NS(cudagraph_mode=NS(name="NONE")), speculative_config=None,
        parallel_config=NS(enable_expert_parallel=True, tensor_parallel_size=4,
            data_parallel_size=dp, enable_eplb=False, enable_dbo=False))


class Tensor:
    def __init__(self, rows, width=None):
        self.rows = rows
        self.width = width
        self.ndim = 1 if width is None else 2
        self.shape = (len(rows),) if width is None else (len(rows), width)


def pad(tensor, padding):
    size = padding[-1]
    return Tensor(tensor.rows + ([0] if tensor.width is None else [[0] * tensor.width]) * size, tensor.width)


def split(tensor, chunks, dim=0):
    assert dim == 0 and tensor.shape[0] % chunks == 0
    size = tensor.shape[0] // chunks
    return [Tensor(tensor.rows[i * size:(i + 1) * size], tensor.width) for i in range(chunks)]


class NativePrepare:
    def prepare(self, hidden, router, shared, replaced, quant):
        self.num_tokens = hidden.shape[0]
        self.enable_shared_expert_dp, self.replace_allreduce = shared, replaced
        return NS(hidden_states=split(hidden, self.tp_size)[self.tp_rank],
                  router_logits=split(router, self.tp_size)[self.tp_rank],
                  padded_hidden_states_shape=hidden.shape)


class AdapterTests(unittest.TestCase):
    def test_support_ep4_ep8(self):
        for dp in (1, 2):
            policy.validate_config(config(dp))

    def test_reject_unsupported_model_quant_graph_parallel(self):
        changes = [("model_config", "dtype", "float16"), ("model_config", "quantization", "int8"),
            ("model_config", "enforce_eager", False), ("parallel_config", "tensor_parallel_size", 2),
            ("parallel_config", "data_parallel_size", 4), ("parallel_config", "enable_eplb", True),
            ("parallel_config", "enable_dbo", True), ("parallel_config", "expert_placement_strategy", "round_robin"),
            ("parallel_config", "pipeline_parallel_size", 2)]
        for obj, key, value in changes:
            with self.subTest(key=key):
                cfg = config()
                setattr(getattr(cfg, obj), key, value)
                with self.assertRaises(ValueError):
                    policy.validate_config(cfg)
        cfg = config()
        cfg.model_config.hf_text_config.num_experts = 256
        with self.assertRaises(ValueError):
            policy.validate_config(cfg)
        cfg = config()
        cfg.compilation_config.cudagraph_mode.name = "FULL"
        with self.assertRaises(ValueError):
            policy.validate_config(cfg)
        cfg = config()
        cfg.speculative_config = NS()
        with self.assertRaises(ValueError):
            policy.validate_config(cfg)

    def test_reject_overlap_and_flashcomm(self):
        for name in ("enable_flashcomm1", "enable_matmul_allreduce", "enable_fused_mc2",
                     "enable_flashcomm2_parallel_size", "multistream_overlap_gate",
                     "multistream_overlap_shared_expert", "enable_shared_expert_dp"):
            cfg = config()
            cfg.additional_config[name] = True
            with self.assertRaises(ValueError):
                policy.validate_config(cfg)

    def test_native_selector_preserved(self):
        calls = []
        native = lambda *args: calls.append(args) or "native-result"
        selected = policy.make_selector(native, "a2a", NS())
        cfg = config()
        cfg.additional_config = {}
        self.assertEqual(selected(9, cfg), "native-result")
        self.assertEqual(calls, [(9, cfg, False)])
        self.assertIs(selected._qwen_original, native)

    def test_real_ep_group_validation_and_draft_refusal(self):
        ctx = NS(get_ep_group=lambda: NS(world_size=8))
        selected = policy.make_selector(lambda *a: None, "a2a", ctx)
        self.assertEqual(selected(1, config(2)), "a2a")
        with self.assertRaises(RuntimeError):
            selected(1, config(1))
        with self.assertRaises(ValueError):
            selected(1, config(2), True)
        cfg = config()
        cfg.additional_config["qwen_moe_comm"] = "typo"
        with self.assertRaises(ValueError):
            selected(1, cfg)

    def test_padding_splits_and_restores_real_count(self):
        padded = make_prepare_class(NativePrepare, NS(nn=NS(functional=NS(pad=pad)), tensor_split=split))
        for n in (0, 1, 3, 4, 5, 7, 8, 9, 1023):
            hidden = Tensor([[i, i + 1] for i in range(n)], 2)
            router = Tensor([[i] for i in range(n)], 1)
            pieces, id_pieces = [], []
            for rank in range(4):
                obj = padded()
                obj.tp_size, obj.tp_rank = 4, rank
                out = obj.prepare(hidden, router)
                self.assertEqual(obj.num_tokens, n)
                self.assertEqual(out.padded_hidden_states_shape[0], policy.padded_tokens(n, 4))
                self.assertEqual(out.hidden_states.shape[0], policy.padded_tokens(n, 4) // 4)
                pieces.extend(out.hidden_states.rows)
                id_pieces.extend(obj.pad_and_split_input_ids(Tensor(list(range(n)))).rows)
            self.assertEqual(pieces[:n], hidden.rows)
            self.assertEqual(id_pieces[:n], list(range(n)))

    def test_padding_refuses_bypass_and_mismatched_ids(self):
        padded = make_prepare_class(NativePrepare, NS(nn=NS(functional=NS(pad=pad)), tensor_split=split))
        obj = padded()
        obj.tp_size, obj.tp_rank = 4, 0
        with self.assertRaises(ValueError):
            obj.prepare(Tensor([[1]], 1), Tensor([[1]], 1), replace_allreduce=True)
        obj.prepare(Tensor([[1]], 1), Tensor([[1]], 1))
        with self.assertRaises(ValueError):
            obj.pad_and_split_input_ids(Tensor([1, 2]))
        with self.assertRaises(ValueError):
            policy.padded_tokens(-1, 4)

    def test_alltoall_command_opt_in_and_native_default(self):
        for role, layout in (("solo-228", "pd4"), ("p-228", "pd8"),
                             ("d8-228", "pd8"), ("d8-238", "pd8")):
            native, _ = runtime.build_command(role, layout)
            self.assertNotIn("--worker-cls", native)
            cmd, env = runtime.build_command(role, layout, moe_comm="alltoall")
            self.assertIn("qwen_moe_adapter.worker.QwenAllToAllWorker", cmd)
            self.assertEqual(json.loads(cmd[cmd.index("--additional-config") + 1])["qwen_moe_comm"], "alltoall")
            self.assertEqual(env["VLLM_USE_V2_MODEL_RUNNER"], "0")
        with self.assertRaises(ValueError):
            runtime.build_command("solo-228", moe_comm="wrong")

    def test_unqualified_adapter_does_not_launch_workers(self):
        with patch.dict(os.environ, {"QWEN_VLLM_HOST": "228", "QWEN_CAM_MODE": "off"}), \
             patch.object(sys, "argv", ["runtime.py", "serve", "--role", "solo-228", "--moe-comm", "alltoall"]), \
             patch.object(policy, "check_installation", side_effect=RuntimeError("changed source")), \
             patch.object(os, "execvpe") as execute, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runtime.main()
            execute.assert_not_called()

    def test_missing_or_changed_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                policy.check_sources(directory)

    def test_install_repairs_captured_alias_and_is_idempotent(self):
        names = ("vllm_ascend", "vllm_ascend.ascend_forward_context", "vllm_ascend.ops",
                 "vllm_ascend.ops.fused_moe", "vllm_ascend.ops.fused_moe.moe_comm_method",
                 "vllm_ascend.ops.fused_moe.prepare_finalize",
                 "vllm_ascend.ops.fused_moe.token_dispatcher", "vllm_ascend.utils",
                 "vllm_ascend.worker", "vllm_ascend.worker.model_runner_v1", "torch")
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            if "." in name:
                parent, attr = name.rsplit(".", 1)
                setattr(modules[parent], attr, module)
        context = modules["vllm_ascend.ascend_forward_context"]
        original = lambda *args: "native"
        context.select_moe_comm_method = original
        context.MoECommType = NS(ALLTOALL="a2a")
        runner = modules["vllm_ascend.worker.model_runner_v1"]
        runner.select_moe_comm_method = original
        impl = type("Impl", (), {})
        modules["vllm_ascend.ops.fused_moe.moe_comm_method"].AlltoAllCommImpl = impl
        modules["vllm_ascend.ops.fused_moe.prepare_finalize"].PrepareAndFinalizeWithAll2All = NativePrepare
        dispatcher = type("Dispatcher", (), {"ep_size": 8, "token_dispatch": lambda *a, **k: "dispatch-result",
                                            "token_combine": lambda *a, **k: "combine-result"})
        modules["vllm_ascend.ops.fused_moe.token_dispatcher"].TokenDispatcherWithAll2AllV = dispatcher
        utils = modules["vllm_ascend.utils"]
        utils.AscendDeviceType, utils.get_ascend_device_type = NS(A2="A2"), lambda: "A2"
        scopes = []
        modules["torch"].profiler = NS(record_function=lambda name: scopes.append(name) or contextlib.nullcontext())
        with patch.dict(sys.modules, modules), patch.object(patching, "check_installation", return_value="pinned") as check:
            patching.install()
            selected = context.select_moe_comm_method
            self.assertIs(runner.select_moe_comm_method, selected)
            self.assertIs(selected._qwen_original, original)
            self.assertEqual(dispatcher().token_dispatch(object()), "dispatch-result")
            self.assertEqual(dispatcher().token_combine(object(), object()), "combine-result")
            patching.install()
            check.assert_called_once()
            self.assertIs(context.select_moe_comm_method, selected)
        self.assertEqual(scopes, ["QWEN_MOE_A2A::dispatch", "QWEN_MOE_A2A::combine"])

    def test_pinned_hashes_when_upstream_present(self):
        root = HERE.parents[2] / "vllm-ascend/vllm_ascend"
        if not root.is_dir():
            self.skipTest("Pinned source snapshot absent")
        policy.check_sources(root)

    def test_probe_ep_counts_and_idle_case(self):
        self.assertEqual(moe_probe.cases(8)[-1], ("idle-dp", (0, 3)))
        self.assertTrue(all(len(counts) == 1 for _, counts in moe_probe.cases(4)))
        with self.assertRaises(ValueError):
            moe_probe.cases(16)

    def test_output_smoke_difference_is_not_hidden(self):
        rows = [{"prompt": p, "text": "answer", "finish_reason": "length"}
                for p in compare_outputs.PROMPTS]
        with tempfile.TemporaryDirectory() as directory:
            native, changed = Path(directory) / "native.json", Path(directory) / "a2a.json"
            native.write_text(json.dumps(rows))
            changed.write_text(json.dumps(rows))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(compare_outputs.compare(native, changed))
                rows[0]["text"] = "different"
                changed.write_text(json.dumps(rows))
                self.assertFalse(compare_outputs.compare(native, changed))


if __name__ == "__main__":
    unittest.main()
