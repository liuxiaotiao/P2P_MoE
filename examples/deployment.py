#!/usr/bin/env python3
"""带专家身份的完整部署：逐节点逐层「装哪些专家」+ 前后段组合矩阵。

    python examples/deployment.py [--seed 7] [--coverage 0.97] [--shared-core 1]
    python examples/deployment.py --json plan.json     # 导出可执行清单

与 examples/appendix_c.py 的区别：那一个只算内存（专家用基数），这一个带专家
**身份**，因而能同时给出：
  * 每台节点每一层要加载哪些专家 id（发给节点的加载指令）
  * 前段并集的逐层真实规模（不是拍一个常数）
  * q(u,û) 可检性矩阵与池合并信号（III.7.3 / III.7.4）
  * N_F × N_B 的完整组合矩阵，且逐对过闸校验
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p2pmoe.planner.experts import (
    detectability_matrix,
    expected_detection_tokens,
    merge_candidates,
)
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.sim.scenario import appendix_c


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--coverage", type=float, default=0.97)
    ap.add_argument("--shared-core", type=int, default=1,
                    help="各 task 共享的热门专家数 —— 直接决定 q(u,û) 与是否该合并池")
    ap.add_argument("--json", type=Path, default=None, help="导出部署清单 JSON")
    args = ap.parse_args()

    st = appendix_c(seed=args.seed, with_experts=True,
                    coverage=args.coverage, shared_core=args.shared_core)
    m = st.model
    L = m.n_layers

    rule("专家驻留集（身份，非基数）")
    print(f"覆盖率阈值 {args.coverage}  →  基线 miss 率 = 1 − 覆盖率 = "
          f"{(1-args.coverage)*100:.1f}%（II.5 通道二的告警基线）")
    print(f"{'task':<6} {'λ':>5} {'n_u,l 层均':>10} {'范围':>9} {'每层GB':>8}  第 6 层驻留专家")
    for t in st.tasks:
        p = t.placement
        sz = p.sizes()
        print(f"{t.name:<6} {t.lam:>5} {sum(sz)/len(sz):>10.1f} "
              f"{min(sz):>4}–{max(sz):<4} "
              f"{m.weight_gb_per_layer(sum(sz)/len(sz)):>8.2f}  {sorted(p.at(6))}")
    u = st.union_experts
    usz = u.sizes()
    print(f"\n前段并集: 层均 {sum(usz)/len(usz):.1f} 个专家（范围 {min(usz)}–{max(usz)}）"
          f" → 每层 {m.weight_gb_per_layer(sum(usz)/len(usz)):.2f}GB")
    print(f"          基数相加会得 {sum(t.placement.size_at(6) for t in st.tasks)} 个"
          f"（第 6 层），实际并集 {u.size_at(6)} 个 —— 差额就是 task 间的重叠")

    rule("命题 III.7.3 可检性矩阵 q(u, û)")
    q = detectability_matrix(st.profiles, st.placements, 6, L)
    names = [t.name for t in st.tasks]
    print("行 = 真实 task，列 = 误绑到的池；对角线是基线 miss 率\n")
    print("        " + "".join(f"{v:>10}" for v in names))
    for a in names:
        row = "".join(f"{q[(a, b)]:>10.3f}" for b in names)
        print(f"  {a:<6}" + row)
    print("\n期望检出延迟 = O(1/q) 个 token：")
    for a in names:
        for b in names:
            if a != b:
                print(f"  {a} 误绑到 {b}: q={q[(a,b)]:.3f} → 约 "
                      f"{expected_detection_tokens(q[(a,b)]):.1f} 个 token 后检出")

    mc = merge_candidates(st.profiles, st.placements, 6, L,
                          q_threshold=0.05, expert_gb=m.expert_gb)
    print()
    if mc:
        for s in mc:
            print(f"  [池合并信号] {s.a} 与 {s.b}: q 双向均 ≤ {s.worst_q:.3f}，"
                  f"Jaccard {s.jaccard:.2f} —— 误判几乎无害，合并可省 "
                  f"{s.saved_gb_per_layer:.2f}GB/层（推论 III.7.4）")
    else:
        print("  [池合并信号] 无 —— 各 task 驻留集重叠度低，误绑高度可检，不该合并")

    try:
        res = plan(st.nodes, st.model, st.tasks, st.union_experts,
                   st.net, st.cfg, st.p_curve)
    except PlanningError as e:
        rule("规划失败")
        for line in e.log:
            print("  " + line)
        print(f"\n{e}")
        return 2

    man = res.manifest
    assert man is not None

    rule("部署清单 —— 每台节点装什么")
    print(f"{'节点':<5} {'角色':<14} {'段':<8} {'层区间':>8} {'专家数':>6} "
          f"{'权重GB':>8} {'KV GB':>7} {'合计':>7}  端点")
    for p in sorted(man.nodes, key=lambda x: (x.role, x.segment, x.position)):
        lo, hi = p.layer_range
        ep = ("head" if p.is_head else "") + ("/tail" if p.is_tail else "")
        print(f"{p.node:<5} {p.role:<14} {p.segment:<8} {lo:>3}–{hi:<4} "
              f"{p.n_experts_total:>6} {p.weight_gb:>8.2f} {p.kv_gb:>7.2f} "
              f"{p.total_gb:>7.2f}  {ep or '中间'}")

    sample = man.nodes[0]
    rule(f"加载指令示例 —— 节点 {sample.node}（{sample.role}）")
    for ll in sample.layers[:4]:
        print(f"  layer {ll.layer:>2}: {len(ll.experts):>2} 个专家 "
              f"{sorted(ll.experts)}  ({ll.weight_gb:.2f}GB)")
    if len(sample.layers) > 4:
        print(f"  ... 共 {len(sample.layers)} 层")
    print("\n  这就是要发给该节点的加载清单：配合 safetensors 的逐张量 mmap，")
    print("  直接翻译成「只打开这些 key」，而不是把整层 64 个专家都载进来。")

    rule("组合矩阵 —— 前后段可任意组合")
    cm = man.combination_matrix()
    fronts = sorted(cm)
    backs = sorted({b for v in cm.values() for b in v})
    print("单元格 = 该组合的单 token p50 (ms)；空格 = 该组合不可用\n")
    print("        " + "".join(f"{b:>9}" for b in backs))
    for f in fronts:
        row = "".join(f"{cm[f][b]:>9.1f}" if b in cm[f] else f"{'—':>9}" for b in backs)
        print(f"  {f:<6}" + row)
    vals = [p.t50 for p in man.pairings]
    print(f"\n  {len(fronts)} × {len(backs)} = {len(man.pairings)} 组，全部可用；"
          f"p50 {min(vals):.1f}–{max(vals):.1f}ms，极差 {max(vals)-min(vals):.1f}ms")
    print(f"  正向接口全部落在公共带 [{man.band[0]:.1f}, {man.band[1]:.1f}]ms 内 —— "
          f"这正是「任意组合」成立的依据（命题 III.8.2）")

    rule("清单校验（七项）")
    checks = ["排他：一节点至多一条段", "内存含 KV ≤ M_v − 预留",
              "层区间连续、完整、无重叠", "并集支配：前段 ⊇ 各 task 驻留集",
              "后段驻留集不多不少", "组合矩阵完整无重复", "逐对过闸：公共带 / w_cap / J_cap"]
    if man.ok:
        for c in checks:
            print(f"  ✓ {c}")
    else:
        for v in man.violations:
            print(f"  ✗ {v}")

    if args.json:
        args.json.write_text(man.to_json(), encoding="utf-8")
        print(f"\n清单已导出: {args.json}")

    return 0 if man.ok and res.audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
