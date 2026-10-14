"""Weight-free A2 EP4/EP8 routing gate. Run only on reserved experiment cards.

Exercises upstream Dispatch/Combine + adapter TP padding/finalize with toy
per-expert scaling. Does NOT certify the real expert FFN, model quality or KV.
All ranks must run the same cases. A failed rank may leave peers until timeout.
"""

import argparse
import os
from datetime import timedelta
from types import SimpleNamespace


CASES = (("balanced", (12, 8)), ("uneven", (5, 7)),
         ("hotspot", (1, 5)), ("idle-dp", (0, 3)))


def cases(world):
    if world not in (4, 8):
        raise ValueError("This probe requires EP4/EP8 with local TP4")
    return [(name, counts[:world // 4]) for name, counts in CASES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "0"))
    experiments = cases(world)
    if args.timeout < 10:
        parser.error("Timeout must be at least 10 seconds")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in ("0,1,2,3", "4,5,6,7"):
        parser.error("Explicitly reserve four local cards with ASCEND_RT_VISIBLE_DEVICES")

    from qwen_moe_adapter.policy import check_installation
    check_installation()
    import torch
    import torch_npu  # noqa: F401: installs NPU/HCCL registrations
    import torch.distributed as dist
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    from qwen_moe_adapter.patching import install, make_prepare_class
    install()
    from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All
    from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV
    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        MoETokenDispatchInput, MoERoutingParams, MoEQuantParams,
    )

    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout))
    try:
        rank = dist.get_rank()
        tp_group = None
        # All world ranks create all subgroups in identical order.
        for start in range(0, world, 4):
            group = dist.new_group(list(range(start, start + 4)),
                                   timeout=timedelta(seconds=args.timeout))
            if start <= rank < start + 4:
                tp_group = group

        class ProbeDispatcher(TokenDispatcherWithAll2AllV):
            # Use a real HCCL process group without loading an engine/model.
            @property
            def ep_group(self):
                return dist.group.WORLD

            @property
            def ep_rank(self):
                return rank

            @property
            def ep_size(self):
                return world

        dispatcher = ProbeDispatcher(num_experts=128, num_local_experts=128 // world, top_k=8)
        prepare_type = make_prepare_class(PrepareAndFinalizeWithAll2All, torch)
        # The probe does not initialize global vLLM config/group coordinators.
        # Set exactly the fields consumed by native prepare/finalize instead.
        prepare = prepare_type.__new__(prepare_type)
        prepare.tp_size, prepare.tp_rank = 4, rank % 4
        prepare.moe_config = SimpleNamespace(tp_group=SimpleNamespace(device_group=tp_group))
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        for label, counts in experiments:
            n = counts[rank // 4]
            # Identical original tokens across each TP group, distinct across DP.
            hidden = (torch.arange(n * 128, device=device).reshape(n, 128) % 16 + 1).to(torch.bfloat16) / 16
            hidden = hidden + rank // 4
            ids = torch.arange(n, device=device, dtype=torch.int64) + (rank // 4) * 32
            router = torch.zeros((n, 128), device=device, dtype=torch.float32)
            prepared = prepare.prepare(hidden, router)
            shard_ids = prepare.pad_and_split_input_ids(ids)

            def experts(token_ids):
                if label == "hotspot":
                    return torch.arange(8, device=device).expand(token_ids.shape[0], 8).to(torch.int32)
                return ((token_ids[:, None] * 13 + torch.arange(8, device=device)[None, :] * 17) % 128).to(torch.int32)

            topk = experts(shard_ids)
            routed = dispatcher.token_dispatch(MoETokenDispatchInput(
                hidden_states=prepared.hidden_states,
                topk_ids=topk,
                topk_weights=torch.full(topk.shape, 1 / 8, device=device, dtype=torch.float32),
                routing=MoERoutingParams(), quant=MoEQuantParams(),
            ))
            if routed.group_list_type != 1:
                raise RuntimeError("Expected per-expert counts")
            local_ids = torch.arange(rank * (128 // world), (rank + 1) * (128 // world), device=device)
            scales = torch.repeat_interleave((local_ids % 11 + 1), routed.group_list.to(torch.int64))
            if scales.shape[0] != routed.hidden_states.shape[0]:
                raise RuntimeError("Routing group-list mismatch")
            expert_output = routed.hidden_states * scales[:, None].to(torch.bfloat16)
            combined = dispatcher.token_combine(expert_output, routed.combine_metadata)
            restored = prepare.finalize(combined, True, prepared.padded_hidden_states_shape)
            expected = hidden.float() * (experts(ids).float() % 11 + 1).mean(dim=1)[:, None]
            ok = restored.shape == hidden.shape and torch.allclose(restored.float(), expected, rtol=0.025, atol=0.02)
            verdict = torch.tensor([int(ok)], device=device, dtype=torch.int32)
            dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
            if verdict.item() != 1:
                raise RuntimeError(f"Routing correctness failed: {label}, rank={rank}, tokens={n}")
            if rank == 0:
                print(f"MOE_A2A_CASE_PASS ep={world} case={label} tokens_per_dp={counts}", flush=True)
        dist.barrier()
        if rank == 0:
            print(f"MOE_A2A_ROUTING_PASS ep={world}; toy experts only, model qualification still required", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
