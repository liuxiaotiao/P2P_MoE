#!/usr/bin/env python3
"""选模型：算清楚哪个真实 MoE 能在你的池子上跑起来。

    python examples/model_fit.py --nodes 15 --mem-gb 24
    python examples/model_fit.py --nodes 15 --mem-gb 40 --config /path/to/config.json

**接真实模型之前先跑这个。** 它回答两件事：

1. **这个模型适不适合本方案** —— 判据是细粒度比 `n_experts / top_k`。比值小的话
   单 token 就用掉一大截专家，一个 task 的驻留集接近全集，「只驻留子集」省不下
   内存，方案失去前提。这一条与你有多少显存无关，是模型自身的性质。

2. **在你的池子上能不能放下** —— 跑真正的规划器（L₀ 选取 + 分档估算），
   给出 L₀、通道数、每条段的驻留量。

不下载权重，不 import torch，只读 config.json 里的几个整数。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2pmoe.planner.capacity import estimate_capacity_by_tier
from p2pmoe.planner.hf_config import (
    PRESETS,
    granularity_verdict,
    model_spec_from_hf,
)
from p2pmoe.planner.memory import choose_l0, make_back_spec, make_front_spec
from p2pmoe.planner.types import Node, TaskProfile

TASKS = [("X", 0.5), ("Y", 0.3), ("Z", 0.2)]


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def build_pool(args) -> list[Node]:
    """统一池或异构池（--pool '16x8,24x7'）。

    异构是常态而不是例外：真实的分散池子里机器是陆续攒起来的，显存档位不一。
    分档估算（II.7.1）本来就是为这种情况设计的 —— 小档可能整档归零，
    大档可能成为某些段形态的唯一承载者。
    """
    if not args.pool:
        return [
            Node(id=f"n{i+1}", tier=f"{args.mem_gb:.0f}GB", mem_gb=args.mem_gb,
                 ms_per_layer=0.35, reserve_gb=args.reserve_gb, avail=0.95)
            for i in range(args.nodes)
        ]
    out: list[Node] = []
    for part in args.pool.split(","):
        mem_s, _, cnt_s = part.strip().partition("x")
        mem, cnt = float(mem_s), int(cnt_s or 1)
        for _ in range(cnt):
            out.append(Node(id=f"n{len(out)+1}", tier=f"{mem:.0f}GB", mem_gb=mem,
                            ms_per_layer=0.35, reserve_gb=args.reserve_gb, avail=0.95))
    if not out:
        raise SystemExit("--pool 解析为空，格式如 '16x8,24x7'")
    return out


def analyse(name: str, cfg: dict, args) -> int:
    spec, info = model_spec_from_hf(cfg, name=name, ctx_max=args.ctx,
                                    dtype_bytes=args.dtype_bytes)

    rule(f"{name}")
    print("  " + info.summary())
    print(f"  全模型 {info.total_gb:.1f}GB（{args.dtype_bytes} 字节/参数）")
    print(f"  每层：attention {info.attn_gb*1000:.0f}MB + "
          f"{info.n_experts} × 专家 {info.expert_gb*1000:.0f}MB = "
          f"{info.layer_full_gb:.2f}GB")
    print(f"  KV：{info.kv_dim} 维（GQA {info.n_kv_heads} 头）→ "
          f"{spec.kv_gb_per_layer*1000:.0f}MB/层 @ ctx={args.ctx}")

    ok, why = granularity_verdict(info)
    print(f"\n  {'✓' if ok else '✗'} {why}")
    if not ok:
        print("\n  —— 跳过放置分析，前提不成立。")
        return 1

    # 驻留集规模按细粒度比推一个合理区间（真实值要跑回放语料才知道）
    for frac in (0.10, 0.20, 0.35):
        n_res = max(info.top_k, round(info.n_experts * frac))
        per = info.layer_gb(n_res)
        print(f"  驻留 {n_res:>3}/{info.n_experts} 专家（{frac:.0%}）→ "
              f"每层 {per*1000:.0f}MB，全部 {info.n_layers} 层 {per*info.n_layers:.1f}GB")

    # ---- 用真正的规划器算 L₀ 与通道数 -------------------------------- #
    n_back = max(info.top_k, round(info.n_experts * args.resident_frac))
    n_union = min(info.n_experts, max(n_back * 2, round(info.n_experts * args.union_frac)))
    tasks = [TaskProfile(name=u, lam=l, experts_per_layer=n_back) for u, l in TASKS]
    nodes = build_pool(args)
    pool_desc = "、".join(
        f"{c} × {m:.0f}GB" for m, c in sorted(
            {n.mem_gb: sum(1 for x in nodes if x.mem_gb == n.mem_gb) for n in nodes}.items()
        )
    )

    rule(f"在 {pool_desc} 的池子上（共 {len(nodes)} 台）")
    print(f"  假设：后段驻留 {n_back}/{info.n_experts}（{args.resident_frac:.0%}），"
          f"前段并集 {n_union}/{info.n_experts}（{n_union/info.n_experts:.0%}）")
    caps = sorted({n.usable_gb for n in nodes})
    print(f"  可用内存/台 {'、'.join(f'{c:.1f}GB' for c in caps)}（预留 {args.reserve_gb}GB）")

    p_curve = {l: min(0.99, 0.60 + 0.03 * l) for l in range(1, info.n_layers)}
    try:
        best, table = choose_l0(spec, tasks, n_union, nodes, p_curve, p_min=args.p_min)
    except ValueError as e:
        print(f"\n  ✗ 放不下：{e}")
        print(f"\n  最小需求：一层前段 {info.layer_gb(n_union)*1000:.0f}MB + KV "
              f"{spec.kv_gb_per_layer*1000:.0f}MB —— 单台至少要装得下一层")
        return 2

    f = make_front_spec(spec, n_union, best.l0)
    bs = {t.name: make_back_spec(spec, t, best.l0) for t in tasks}
    cap = estimate_capacity_by_tier(nodes, f, bs, tasks)

    print(f"\n  L₀ = {best.l0}（前段 1–{best.l0} 层，后段 {best.l0+1}–{info.n_layers} 层）")
    print(f"    前段 {f.total_gb():.1f}GB → 跳数下界 {best.hops_front}，"
          f"能单节点承载的机器 {best.front_single_node_count}/{len(nodes)} 台")
    for u, b in bs.items():
        print(f"    后段 {u} {b.total_gb():.1f}GB → 跳数下界 {best.hops_back[u]}")
    print(f"\n  单通道需求 {cap.channel_demand_gb:.1f}GB；"
          f"活跃供给 {cap.active_supply_gb:.0f}GB"
          + (f"（归零档 {cap.zeroed_supply_gb:.0f}GB）" if cap.zeroed_supply_gb else ""))
    print(f"  通道数：内存上界 {cap.n_max}，精确位配平 {cap.n_max_slots}"
          f" → θ=0.8 打折后可建 {int(0.8 * cap.n_max_slots)} 条")
    n_ch = int(0.8 * cap.n_max_slots)
    if n_ch < 1:
        print("\n  ✗ 建不出完整通道 —— 加机器、加显存，或选更小的模型")
        return 2
    print(f"\n  ✓ 可建 {n_ch} 条通道 → 同时服务 {n_ch} 条请求"
          f"（排他独占；批处理见 TODO.md P1）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=15)
    ap.add_argument("--mem-gb", type=float, default=24.0, help="每台的可用显存（统一池）")
    ap.add_argument("--pool", default=None,
                    help="异构池：'16x8,24x7' = 8 台 16GB + 7 台 24GB。"
                         "给了它就忽略 --nodes/--mem-gb")
    ap.add_argument("--reserve-gb", type=float, default=2.0, help="每台预留")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2=fp16/bf16, 1=int8")
    ap.add_argument("--resident-frac", type=float, default=0.20,
                    help="假设每个 task 的驻留集占全部专家的比例。真实值要跑回放"
                         "语料才知道（corpus.profile_from_corpus）")
    ap.add_argument("--union-frac", type=float, default=0.45,
                    help="假设前段并集占比")
    ap.add_argument("--p-min", type=float, default=0.75)
    ap.add_argument("--config", type=Path, default=None,
                    help="HF config.json 路径。不给则跑内置的三个候选")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    if args.config:
        import json
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        return analyse(args.name or args.config.parent.name, cfg, args)

    rc = 0
    for name, cfg in PRESETS.items():
        rc |= analyse(name, cfg, args)
    rule("怎么用")
    print("  换成你自己的模型：--config /path/to/config.json")
    print("  换成你自己的机器：--nodes 15 --mem-gb 40 --reserve-gb 4")
    print("  驻留集比例（--resident-frac）现在是假设值；真实值要跑回放语料统计，")
    print("  见 runtime/corpus.py::profile_from_corpus。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
