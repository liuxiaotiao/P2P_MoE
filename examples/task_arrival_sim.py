#!/usr/bin/env python3
"""随机到达 + 盲绑 + 在线识别：offline 排好的池子，直接拿来打请求。

    python examples/task_arrival_sim.py --data ./task --mix mbpp=5,gsm8k=3 \
        --coverage 0.70 --requests 400

回答的问题
----------
「按 5:3 随机往前段扔 mbpp / gsm8k 的请求，现在的 offline 能不能提前把路组织好、
然后直接测？」

**能，但走的是动态路径而不是静态配对。** 静态模式下一条前段绑死一个 task，
随机来的请求它接不住。文档主线正是为这个设计的：

    到达 → 盲绑一条空闲前段（不比不挑）
         → 前段跑完 1..L₀，本地按激活直方图识别 task
         → 从该 task 的后段池取一条 → 建链
         → decode 绕环 → 两段回池

offline 保证的**不是**「预先排好固定路径」，而是「**任意**前段配**任意**同 task
后段都不后悔」—— 那就是公共中值域（II.3）与组合矩阵在做的事。所以池子建好之后
不需要再为具体的到达序列做任何安排。

这个模拟测什么、不测什么
------------------------
**测得了**：识别准确率、误绑率、通道占用与排队、每条前段实际配过哪些后段
（「任意组合」在到达序列下的样子）、公平比与到达比是否对得上。

**测不了**：真实的逐请求激活轨迹。CSV 只有**聚合**分布，所以这里的每条请求是
从该 task 的分布里**独立同分布采样** n 个 token 生成的直方图。真实请求的 token
之间高度相关（一道数学题的每一步都在同一个语域里），所以真实的逐请求直方图
**方差更大、更容易跨到另一个 task 那边** —— 也就是说：

    这里量出的识别准确率是**乐观上界**。

要拿到真数，得在真模型上按请求粒度采激活（`deploy/run.py --profile-out` 现在
是按 task 聚合的，改成按请求存就行）。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from p2pmoe.planner.experts import build_placement, union_placement
from p2pmoe.planner.from_data import FabricOracle, load_activation_csv, load_topology
from p2pmoe.planner.hf_config import model_spec_from_hf
from p2pmoe.planner.network import MeasurementCache
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.planner.types import PlannerConfig, TaskProfile
from p2pmoe.runtime.identify import HistogramClassifier

from task_deploy import QWEN3_NEXT_80B, identifiability  # noqa: E402


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./task")
    ap.add_argument("--mix", default="mbpp=5,gsm8k=3", help="到达比，如 'mbpp=5,gsm8k=3'")
    ap.add_argument("--coverage", type=float, default=0.70)
    ap.add_argument("--l0", type=int, default=None)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--tokens", type=int, default=64, help="每条请求生成多少 token")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--tau-hi", type=float, default=0.55)
    ap.add_argument("--tau-lo", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    D = Path(args.data)
    mix = {k.strip(): float(v) for k, v in (x.split("=") for x in args.mix.split(","))}
    tot = sum(mix.values())
    lam = {k: v / tot for k, v in mix.items()}
    rng = np.random.default_rng(args.seed)

    # ---------------- 离线：建池 ---------------- #
    rule("1. 离线：按到达比建池")
    prof = {u: load_activation_csv(D / f"{u}_expert_activation.csv", task=u)[0]
            for u in lam}
    topo = load_topology(D / "topo_fabric_mbps.json",
                         d_model=QWEN3_NEXT_80B["hidden_size"], reserve_gb=1.0)
    spec, info = model_spec_from_hf(QWEN3_NEXT_80B, name="qwen3-next-80b",
                                    ctx_max=args.ctx, dtype_bytes=2)
    acc = identifiability(prof, info.n_layers)
    plc = {u: build_placement(p, args.coverage) for u, p in prof.items()}
    uni = union_placement(list(plc.values()))
    tasks = [TaskProfile(name=u, lam=lam[u],
                         experts_per_layer=plc[u].as_experts_per_layer(),
                         placement=plc[u]) for u in sorted(lam)]
    cfg = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                        kappa_over=0.3, n_standby=0, seed=args.seed)
    net = MeasurementCache(FabricOracle(topo), k=cfg.k_probe,
                           j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)
    p_curve = {l: float(acc[l - 1]) for l in range(1, info.n_layers)}
    if args.l0:
        p_curve = {args.l0: p_curve[args.l0]}
    try:
        res = plan(topo.nodes, spec, tasks, uni, net, cfg, p_curve, p_min=0.5)
    except PlanningError as e:
        print(f"  规划失败：{e}")
        return 2
    man = res.manifest
    L0 = res.l0
    n_f = len(res.fronts_final)
    print(f"  到达比 {dict((k, round(v,3)) for k,v in lam.items())}，覆盖率 {args.coverage}")
    print(f"  L₀={L0}（p≥{acc[L0-1]:.2%}），后段每层层均 "
          f"{np.mean([len(plc[u].at(l)) for u in plc for l in range(1, info.n_layers+1)]):.0f}"
          f"/{info.n_experts}")
    print(f"  前段 {n_f} 条：{[s.label() for s in res.fronts_final]}")
    for u, segs in sorted(res.backs.items()):
        print(f"  后段 {u}: {len(segs)} 条 {[s.label() for s in segs]}")
    print(f"  组合矩阵 {len(man.pairings)} 组 —— **这就是「提前组织好的 path」**：")
    print(f"    不是给每条请求排一条固定路，而是保证任意前段配任意同 task 后段都不后悔")
    mat = man.combination_matrix()
    ts = sorted(res.backs)
    real = {u: len(res.backs[u]) for u in ts}
    rtot = sum(real.values())
    print(f"  实建条数 {real} → 实际比 "
          f"{ {u: round(v/rtot,3) for u,v in real.items()} }（目标 "
          f"{ {u: round(lam[u],3) for u in ts} }）")

    # ---------------- 在线：随机到达 ---------------- #
    rule("2. 在线：盲绑 → 识别 → 派发")
    clf = HistogramClassifier.from_profiles(prof, L0, lam,
                                            tau_hi=args.tau_hi, tau_lo=args.tau_lo)
    front_ids = sorted(s for s in man.segments if man.segments[s]["role"] == "front")
    back_ids = {u: sorted(s for s in man.segments
                          if man.segments[s]["role"] == f"back:{u}") for u in ts}
    # 前段激活质量：前 L₀ 层之和，逐 task
    front_mass = {u: np.sum([np.asarray(prof[u].at(l)) for l in range(1, L0 + 1)], 0)
                  for u in ts}
    front_mass = {u: m / m.sum() for u, m in front_mass.items()}

    names = sorted(lam)
    probs = np.array([lam[u] for u in names])
    n_draw = L0 * args.tokens * QWEN3_NEXT_80B["num_experts_per_tok"]

    free_f = list(front_ids)
    free_b = {u: list(back_ids[u]) for u in ts}
    pairs_seen = defaultdict(set)
    conf = Counter()          # (真实, 识别)
    zones = Counter()
    use_f, use_b = Counter(), Counter()
    queued = 0
    for i in range(args.requests):
        true = names[int(rng.choice(len(names), p=probs))]
        # 逐请求直方图：从该 task 的前段分布 i.i.d. 采样（乐观，见文件头）
        hist = rng.multinomial(n_draw, front_mass[true]).astype(float)
        v = clf.predict(hist)
        got, zone = v.task, v.zone
        conf[(true, got)] += 1
        zones[zone] += 1
        if not free_f:
            queued += 1
            free_f = list(front_ids)      # 简化：本模拟不建时间轴，只看配对与识别
        f = free_f.pop(0)
        use_f[f] += 1
        pool = free_b.get(got) or list(back_ids.get(got, []))
        if not pool:
            continue
        b = pool.pop(0)
        free_b[got] = pool or list(back_ids[got])
        use_b[b] += 1
        pairs_seen[f].add(b)

    ok = sum(v for (t, g), v in conf.items() if t == g)
    print(f"  {args.requests} 条请求，到达实际比 "
          f"{ {u: round(sum(v for (t,_),v in conf.items() if t==u)/args.requests,3) for u in names} }")
    print(f"\n  识别准确率 {ok/args.requests:.2%}（p(L₀) 的理论下界 {acc[L0-1]:.2%}）")
    print(f"  混淆矩阵（行=真实，列=识别）:")
    print(f"    {'':<8}" + "".join(f"{g:>10}" for g in names))
    for t in names:
        row = "".join(f"{conf[(t,g)]:>10}" for g in names)
        print(f"    {t:<8}{row}")
    print(f"  置信区间分布: {dict(zones)}")
    print(f"\n  前段使用: {dict(use_f)}")
    print(f"  后段使用: {dict(use_b)}")
    print(f"  出现过的组合:")
    for f in sorted(pairs_seen):
        print(f"    {f} 配过 {sorted(pairs_seen[f])}")
    n_comb = sum(len(v) for v in pairs_seen.values())
    print(f"  → {n_comb}/{len(man.pairings)} 组组合被用到 —— "
          f"「盲绑 + 任意组合」在到达序列下的样子")

    # ---- 负载 vs 产能：这才是 5:3 到底有没有被满足 ----
    rule("3. 负载 vs 产能")
    print(f"  {'task':<8}{'到达占比':>9}{'产能占比':>9}{'负载/产能':>10}{'每条后段接了':>13}")
    over = {}
    for u in ts:
        arr = sum(v for (t, _), v in conf.items() if t == u) / args.requests
        capf = real[u] / rtot
        over[u] = arr / capf if capf else float("inf")
        per = sum(use_b[b] for b in back_ids[u]) / max(real[u], 1)
        print(f"  {u:<8}{arr:>9.1%}{capf:>9.1%}{over[u]:>10.2f}×{per:>12.0f} 条")
    worst = max(over, key=lambda u: over[u])
    if over[worst] > 1.2:
        print(f"\n  ⚠ **{worst} 的产能不够**：它接了 {over[worst]:.2f} 倍于自己产能的负载。")
        print(f"    并发度 = 通道数，所以超出的部分不是变慢，是**排队**。")
        print(f"    根因见规划日志的「实建少于配额」—— 不是配额算错，是**建不出来**：")
        print(f"    {worst} 的驻留集更大 → 后段更重 → 剩下的机器凑不出一条不超跳数下界的段。")
        print(f"    三条路：① 降 {worst} 的覆盖率（只降它，逐 task 可以不同）；")
        print(f"           ② 加机器（扩容第一条按公平比就该加给 {worst}）；")
        print(f"           ③ 接受不均衡，用有界等待兜住 —— 但 {over[worst]:.1f}× 会让队列一直有人。")
    print(f"\n  ⚠ 这里的识别准确率是**乐观上界**：逐请求直方图是从聚合分布 i.i.d. "
          f"采样的，\n    真实请求的 token 高度相关，方差更大、更容易跨界。见文件头。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
