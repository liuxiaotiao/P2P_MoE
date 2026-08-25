#!/usr/bin/env python3
"""一键复现方案文档附录 C 的端到端算例。

    python examples/appendix_c.py [--seed 7] [--eta 0.12] [--rent-policy ...]

输出即离线规划的完整交付物：L₀ 候选表、分档估算、配额与公平比、后段/前段
排布、公共中值域的人口曲线、回环裁剪名单、终审账本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.sim.scenario import UNION_EXPERTS, appendix_c


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eta", type=float, default=0.12)
    ap.add_argument("--j-cap", type=float, default=25.0)
    ap.add_argument("--beta", type=float, default=1.25)
    ap.add_argument("--rent-policy", default="smallest_sufficient",
                    choices=["smallest_sufficient", "largest"])
    args = ap.parse_args()

    st = appendix_c(seed=args.seed, eta=args.eta, j_cap=args.j_cap, beta=args.beta)

    rule("C.0 舞台")
    print(f"模型: {st.model.n_layers} 层, d_model={st.model.d_model}, "
          f"{st.model.n_experts} 专家 top-{st.model.top_k}, ctx_max={st.model.ctx_max}")
    print(f"      KV = {st.model.kv_gb_per_layer:.3f} GB/层, "
          f"并集 {st.model.weight_gb_per_layer(UNION_EXPERTS):.2f} GB/层")
    for t in st.tasks:
        print(f"      task {t.name}: λ={t.lam}, {t.experts_per_layer} 专家 → "
              f"{st.model.weight_gb_per_layer(t.experts_per_layer):.2f} GB/层")
    tiers: dict[str, int] = {}
    for n in st.nodes:
        tiers[n.tier] = tiers.get(n.tier, 0) + 1
    print("节点池: " + ", ".join(f"{v}× {k}" for k, v in tiers.items()))
    print("网络:   " + st.sim.summary())
    print(f"参数:   η={st.cfg.eta}, J_cap={st.cfg.j_cap_ms}ms, β={st.cfg.beta}, "
          f"θ={st.cfg.theta}, κ_over={st.cfg.kappa_over}")

    try:
        res = plan(
            st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve,
            rent_policy=args.rent_policy,
        )
    except PlanningError as e:
        rule("规划失败")
        for line in e.log:
            print("  " + line)
        print(f"\n{e}")
        return 2

    rule("L₀ 候选表 (III.5.3 + 供给约束)")
    print(f"{'L₀':>3} {'p(L₀)':>6} {'前段GB':>7} {'加权跳数':>9} {'通道数':>7} {'前段合格节点':>12}")
    for c in res.l0_table:
        mark = " ←" if c.l0 == res.l0 else ""
        print(f"{c.l0:>3} {c.p_accuracy:>6.2f} {c.front_gb:>7.1f} "
              f"{c.total_hops_weighted:>9.2f} {c.n_channels:>7} "
              f"{c.front_single_node_count:>12}{mark}")

    rule("Step 1 分档估上限 (II.7.1)")
    print(f"{'档位':<10} {'台数':>4} {'可用GB':>7} {'判定':<8} 能承担的段位")
    for r in res.capacity.reports:
        serves = ", ".join(k for k, v in r.serves.items() if v) or "—"
        print(f"{r.tier.name:<10} {r.tier.count:>4} {r.tier.usable_gb:>7.1f} "
              f"{r.verdict:<8} {serves}")
    c = res.capacity
    print(f"\n活跃供给 {c.active_supply_gb:.0f}GB / 归零档 {c.zeroed_supply_gb:.0f}GB "
          f"(废料率 {c.waste_ratio*100:.0f}%) / 单通道需求 {c.channel_demand_gb:.1f}GB")
    print(f"N_max(内存上界) = {c.n_max}, N_max(精确位配平) = {c.n_max_slots}")

    rule("排布")
    for u, segs in res.backs.items():
        for s in segs:
            splits = " ".join(f"{v}[{lo}-{hi}]" for v, (lo, hi) in zip(s.nodes, s.splits))
            print(f"  后段 {u}  {splits:<34} 计算 {s.compute_ms:>5.1f} + 段内跳 "
                  f"{s.hop_ms:>5.1f} = {s.delay_ms:>6.1f}ms  ({s.hops} 跳)")
    print()
    kept = {id(s) for s in res.fronts_final}
    for s in res.fronts_build:
        tag = "保留" if id(s) in kept else "备胎"
        splits = " ".join(f"{v}[{lo}-{hi}]" for v, (lo, hi) in zip(s.nodes, s.splits))
        print(f"  前段 {tag} {splits:<34} 计算 {s.compute_ms:>5.1f} + 段内跳 "
              f"{s.hop_ms:>5.1f} = {s.delay_ms:>6.1f}ms  ({s.hops} 跳)")

    rule("Step 4 公共中值域 —— 人口曲线 (II.3.3)")
    print(f"{'W (ms)':>8} {'|Q_F^out|':>10}  带")
    for W, r in res.band_curve:
        star = " ←" if abs(W - res.band.W) < 1e-6 else ""
        band = f"[{r.w_lo:.1f}, {r.w_hi:.1f}]" if r.population else "—"
        print(f"{W:>8.1f} {r.population:>10}  {band}{star}")

    rule("Step 6 回环裁剪 (III.8.3)")
    keptset = set(res.trim.kept)
    for p in res.trim.profiles:
        tag = "保留" if p.front_index in keptset else "备胎"
        print(f"  {p.front_label:<22} head={p.head:<4} D={p.D:>6.1f}ms  {tag}")
    print(f"\n  最坏回环 {res.trim.worst_D:.1f}ms；保留集极差 {res.trim.spread_kept:.1f}ms "
          f"≤ 全集极差 {res.trim.spread_all:.1f}ms  →  命题 III.8.3(b) "
          f"{'成立' if res.trim.check_iii_8_3()[1] else '不成立'}")

    rule("终审账本 (C.4)")
    for k, v, src in res.audit.ledger():
        print(f"  {k:<26} {v:<22} {src}")
    print(f"\n  终审: {'通过' if res.audit.passed else '未通过 — ' + '；'.join(res.audit.reasons)}")

    rule("单 token 账单示例")
    p = min(res.audit.pairs, key=lambda x: x.t50)
    print(f"  {p.front} → {p.back}")
    print(f"    前段计算 {p.t_front:>6.1f}  正向 w {p.w_fwd:>6.1f}  "
          f"后段 {p.t_back:>6.1f}  回环 {p.d_loop:>6.1f}   = {p.t50:.1f}ms")
    compute = sum(s.compute_ms for s in res.fronts_final if s.label() == p.front)
    compute += sum(s.compute_ms for v in res.backs.values() for s in v if s.label() == p.back)
    print(f"    其中计算 {compute:.1f}ms ({compute/p.t50*100:.0f}%)，"
          f"网络 {p.t50-compute:.1f}ms ({(p.t50-compute)/p.t50*100:.0f}%) —— 分散环境的本质")

    rule("规划日志")
    for line in res.log:
        print("  " + line)
    print(f"\n  累计探测样本数: {res.n_probes}")
    return 0 if res.audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
