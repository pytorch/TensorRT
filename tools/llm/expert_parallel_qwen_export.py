"""
Pure-EP (single WORLD group, NO 2D mesh) export of a REAL Qwen3 MoE block.

Same driver as moe_ep_a2a_export.py, but the block being sharded is an actual
``Qwen3MoeSparseMoeBlock`` from transformers instead of the toy RefMoE. This is
the "plug in the real block" step: shard_moe_block reads .gate/.top_k/
.num_experts/.norm_topk_prob and this rank's expert weights from block.experts.

NOTE: HF migrated Qwen3MoeSparseMoeBlock to the FUSED ``Qwen3MoeExperts`` (packed
weight tensors gate_up_proj [E,2*inter,H] / down_proj [E,H,inter]), not a
ModuleList. shard_moe_block handles this by slicing the packed tensors for this
rank and running a batched bmm (_PackedExperts); the old ModuleList layout still
works too. So a wrapper change WAS needed -- it lives in moe_ep_a2a_wrapper.py.

PHASE 1 ONLY: one WORLD group, experts sharded across ranks, all_to_all routing.
No tensor parallel, no EP x TP mesh (that's Phase 2 + the subgroup MRs).

The DENSE REFERENCE here is the block's own unmodified forward() (all experts on
one device, dropless). With capacity_factor=4 the sharded EP path takes no drops,
so TRT / eager-wrapper should match the block's native output bit-for-bit.

RUN (from tools/llm/, on the GPU box, via the torch-tensorrt launcher):
  # eager parity + export + build + run, 2 ranks:
  torchtrtrun --nproc_per_node=2 qwen3_moe_ep_export.py

  # real pretrained weights (the full R5 datapoint):
  torchtrtrun --nproc_per_node=2 qwen3_moe_ep_export.py --hf-model Qwen/Qwen3-30B-A3B

  # smaller/larger block:
  torchtrtrun --nproc_per_node=2 qwen3_moe_ep_export.py --experts 8 --hidden 64 --inter 128

  # export + build only:
  torchtrtrun --nproc_per_node=2 qwen3_moe_ep_export.py --build-only
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch_tensorrt
import torch_tensorrt.distributed.md_conversion as md
from moe_ep_a2a_wrapper import shard_moe_block
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt


def _log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def build_qwen3_block(hidden, inter, experts, topk, device, dtype):
    """Construct a REAL (randomly-initialised) Qwen3MoeSparseMoeBlock at a small
    config so the whole thing fits and runs fast. Real Qwen3 code path, toy size."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    cfg = Qwen3MoeConfig(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=topk,
        norm_topk_prob=True,
        # keep the rest tiny -- we only instantiate the MoE block, not the model
        intermediate_size=inter,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    block = Qwen3MoeSparseMoeBlock(cfg).eval().to(device, dtype)
    # Qwen3MoeExperts uses torch.empty (uninitialized) — init explicitly
    torch.nn.init.normal_(block.experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(block.experts.down_proj, std=0.02)
    return block, cfg.hidden_size


def load_hf_qwen3_block(model_id, layer, device, dtype):
    """Pull ONE real MoE block (with pretrained weights) out of an HF checkpoint.
    We only need a single Qwen3MoeSparseMoeBlock, so load the model, take its
    block, and free the rest. Returns (block, hidden_size). Real HF-model datapoint."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    layers = model.model.layers
    # find the requested layer if it's an MoE block, else the first MoE layer
    cand = (
        layer
        if isinstance(getattr(layers[layer], "mlp", None), Qwen3MoeSparseMoeBlock)
        else None
    )
    if cand is None:
        for i, lyr in enumerate(layers):
            if isinstance(getattr(lyr, "mlp", None), Qwen3MoeSparseMoeBlock):
                cand = i
                break
    assert cand is not None, "no Qwen3MoeSparseMoeBlock found in the model"
    block = layers[cand].mlp.to(device, dtype).eval()
    hidden = model.config.hidden_size
    del model  # free the rest of the 30B weights; keep only the one block
    torch.cuda.empty_cache()
    _log(
        0,
        f"loaded REAL HF block from {model_id} layer {cand} "
        f"(E={block.experts.num_experts}, k={block.experts.config.num_experts_per_tok}, H={hidden})",
    )
    return block, hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--build-only",
        action="store_true",
        help="export + compile only; skip the TRT forward pass",
    )
    p.add_argument("--python-runtime", action="store_true")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--inter", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="load a REAL pretrained MoE block, e.g. Qwen/Qwen3-30B-A3B "
        "(overrides --hidden/--inter/--experts/--topk from the checkpoint)",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=0,
        help="which decoder layer's MoE block to pull (with --hf-model)",
    )
    p.add_argument(
        "--capacity-factor",
        type=float,
        default=4.0,
        help="dropless factor is E/k (toy 8/2=4, Qwen3 128/8=16). "
        "Use E/k for bit-exact correctness; ~1.25 for a realistic benchmark.",
    )
    p.add_argument("--verify-iters", type=int, default=5)
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="relative tolerance for the [4] allclose pass/fail (bf16 noise ~1e-2)",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="absolute tolerance for the [4] allclose pass/fail",
    )
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="total token count T (default 8*ws). Use a larger value "
        "(e.g. 512, 2048) for a realistic benchmark load; must be divisible by ws",
    )
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(
        os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1))
    )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29557")
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(ws))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl")
    setup_nccl_for_torch_tensorrt()
    md.set_config(ws, rank, dist.group.WORLD.group_name)

    torch.manual_seed(0)
    dtype = torch.bfloat16

    # REAL Qwen3 block = the dense reference; shard_moe_block keeps this rank's
    # experts (shares weights, so the comparison is apples-to-apples).
    #   --hf-model  -> pretrained weights (real HF-model datapoint)
    #   otherwise   -> randomly-initialised real Qwen3 block at a small config
    if args.hf_model:
        block, H = load_hf_qwen3_block(
            args.hf_model, args.layer, device, dtype
        )  # loading pretrained model weights
    else:
        block, H = build_qwen3_block(
            args.hidden, args.inter, args.experts, args.topk, device, dtype
        )  # building qwen3 block
    E = block.experts.num_experts
    assert E % ws == 0, f"experts ({E}) must be divisible by world_size ({ws})"
    wrapped = (
        shard_moe_block(block, ws, rank, capacity_factor=args.capacity_factor)
        .eval()
        .to(device, dtype)
    )  # custom model

    if args.tokens:
        assert (
            args.tokens % ws == 0
        ), f"--tokens ({args.tokens}) must be divisible by ws ({ws})"
        B, S = 1, args.tokens
    else:
        B, S = 1, 8 * ws
    hs = torch.randn(B, S, H, device=device, dtype=dtype)

    # -- 1. eager parity: real Qwen3 block vs all_to_all EP wrapper --
    with torch.no_grad():
        ref_out = block(hs)  # Qwen3MoeSparseMoeBlock returns a single tensor
        eager_out, _ = wrapped(hs)
    eager_err = (ref_out - eager_out).abs().max().item()
    _log(
        rank,
        f"[1] eager parity (Qwen3 block vs EP wrapper): max_abs_err = {eager_err:.3e} "
        f"(should be ~0 when capacity >= actual load)",
    )

    dist.barrier()

    # -- 2. torch.export --
    with torch.no_grad():
        ep = torch.export.export(wrapped, (hs,))
    _log(rank, "[2] torch.export: OK")

    # -- 3. torch_tensorrt.dynamo.compile (native all_to_all) --
    trt_mod = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[hs],
        min_block_size=1,
        use_python_runtime=args.python_runtime,
        use_distributed_mode_trace=True,
        enabled_precisions={dtype},
    )
    _log(rank, "[3] torch_tensorrt.dynamo.compile: engine built")

    # Debug-build torch-tensorrt has engine-level execution profiling on by default,
    # which writes 6 trace files to /tmp on every forward call and adds ~30ms.
    # Explicitly disable it on every TRT engine submodule before benchmarking.
    disabled = 0
    for sub in trt_mod.modules():
        if hasattr(sub, "disable_profiling"):
            try:
                sub.disable_profiling()
                disabled += 1
            except Exception as ex:
                _log(
                    rank, f"    disable_profiling on {type(sub).__name__} failed: {ex}"
                )
    _log(rank, f"    disabled profiling on {disabled} TRT engine submodule(s)")

    if args.build_only:
        _log(rank, "[build-only] done.")
        dist.destroy_process_group()
        return

    def _fwd(module, x):
        out = module(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def _same_input():
        x = torch.randn(B, S, H, device=device, dtype=dtype)
        dist.broadcast(x, src=0)
        return x

    # -- 4. correctness: TRT vs eager wrapper AND TRT vs the real Qwen3 block --
    try:
        eager_ok = ref_ok = 0
        max_e = max_r = 0.0
        with torch.no_grad():
            for _ in range(args.verify_iters):
                x = _same_input()
                r = _fwd(block, x)  # real Qwen3 block (dropless dense)
                e = _fwd(wrapped, x)  # eager all_to_all EP wrapper
                t = _fwd(trt_mod, x)  # TRT engine
                de = (t - e).abs().max().item()
                dr = (t - r).abs().max().item()
                max_e = max(max_e, de)
                max_r = max(max_r, dr)
                eager_ok += torch.allclose(
                    t.float(), e.float(), rtol=args.rtol, atol=args.atol
                )
                ref_ok += torch.allclose(
                    t.float(), r.float(), rtol=args.rtol, atol=args.atol
                )
        n = args.verify_iters
        engine_ok = eager_ok == n  # TRT == our eager EP  -> engine + collective correct
        exact = ref_ok == n  # TRT == dropless reference -> no capacity drops
        if engine_ok and exact:
            verdict = "OK (engine correct AND dropless)"
        elif engine_ok and not exact:
            verdict = (
                f"CAPACITY DROPS -- engine is CORRECT (TRT==eager {eager_ok}/{n}), but "
                f"TRT!=reference (max {max_r:.3e} > atol {args.atol:.0e}). "
                f"Raise --capacity-factor toward E/k for bit-exact."
            )
        else:
            verdict = (
                f"ENGINE MISMATCH -- TRT!=eager (max {max_e:.3e} > atol {args.atol:.0e}): "
                f"collective not applied / lowering bug."
            )
        _log(
            rank,
            f"[4] correctness over {n} inputs [rtol={args.rtol:.0e}, atol={args.atol:.0e}]: "
            f"TRT==eager {eager_ok}/{n} (max {max_e:.3e}) | "
            f"TRT==reference {ref_ok}/{n} (max {max_r:.3e})  <-- {verdict}",
        )
    except Exception as ex:  # noqa: BLE001
        _log(rank, f"[4] TRT run raised: {type(ex).__name__}: {ex}")
        dist.destroy_process_group()
        return

    # -- 5. benchmark --
    if args.benchmark:
        T = B * S

        # Silence TRT/torch_tensorrt verbose logs so they don't inflate timings.
        # Debug-build torch-tensorrt emits per-forward DEBUG/INFO output plus writes
        # profile traces to /tmp on every call, both of which crush TRT numbers.
        import torch_tensorrt.logging as trt_log

        def _bench(module):
            x = _same_input()
            with torch.no_grad():
                for _ in range(args.warmup):
                    _fwd(module, x)
            torch.cuda.synchronize()
            dist.barrier()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            times = []
            with torch.no_grad():
                for _ in range(args.iters):
                    ev0.record()
                    _fwd(module, x)
                    ev1.record()
                    torch.cuda.synchronize()
                    # System latency = slowest rank. Without the max-reduce,
                    # each rank reports its own ev0-ev1 window, which drifts
                    # with rank skew and doesn't correspond to any real cost.
                    t = torch.tensor([ev0.elapsed_time(ev1)], device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.MAX)
                    times.append(t.item())
            times.sort()
            return times[len(times) // 2]

        # three points: dense (no-EP, single-device compute) | EP eager | EP TRT
        with trt_log.errors():
            dense_ms = _bench(
                block
            )  # NO expert parallel: full dense block, all E experts here
            eager_ms = _bench(wrapped)  # our EP, eager
            trt_ms = _bench(trt_mod)  # our EP, compiled to TRT
        tps = lambda ms: T / ms * 1e3
        _log(
            rank,
            f"[5] benchmark (median of {args.iters}, T={T} tok, ws={ws}, "
            f"E={E}, k={block.gate.top_k if hasattr(block, 'gate') else args.topk}):",
        )
        _log(
            rank,
            f"      dense (no-EP) eager : {dense_ms:8.3f} ms  ({tps(dense_ms):>8.0f} tok/s)",
        )
        _log(
            rank,
            f"      EP            eager : {eager_ms:8.3f} ms  ({tps(eager_ms):>8.0f} tok/s)",
        )
        _log(
            rank,
            f"      EP            TRT   : {trt_ms:8.3f} ms  ({tps(trt_ms):>8.0f} tok/s)",
        )
        _log(
            rank,
            f"      speedup: EP-TRT vs dense-eager {dense_ms / trt_ms:5.2f}x | "
            f"EP-TRT vs EP-eager {eager_ms / trt_ms:5.2f}x",
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
