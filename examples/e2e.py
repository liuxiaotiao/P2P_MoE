#!/usr/bin/env python3
"""单机多进程端到端 demo —— 把离线施工图真的跑起来。

    python examples/e2e.py [--seed 3] [--tokens 12] [--latency-scale 0.3]

每个 manifest 节点起一个**独立进程**，通信走本地 socket，发送前按规划期实测到的
(p50, jitter) 注入延迟。模型是 toy MoE（8 层 / 32 专家 / top-2），但走的是真实的
权重选择性加载 —— 每个进程只 materialize 自己那份专家。

跑通的是完整一条请求的生命周期：

    到达 → 盲绑前段 → 前段逐节点 prefill + 捎带激活直方图
         → tail(f) 本地识别 → 派发后段 → 正向接口传 hidden state
         → 后段 prefill → 采样 → 回环把 token 传回 head(f)
         → decode 逐 token 绕环 → 通道二 miss 率检出 → 换绑（前段 KV 不动）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from p2pmoe.planner.experts import build_placement, detectability_matrix, union_placement
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.planner.network import MeasurementCache
from p2pmoe.planner.types import ModelSpec, Node, PlannerConfig, TaskProfile
from p2pmoe.runtime.corpus import (make_corpus, measure_baseline_miss,
                                   measure_front_refs, profile_from_corpus, sample_prompt)
from p2pmoe.runtime.coordinator import LocalCluster
from p2pmoe.runtime.identify import HistogramClassifier
from p2pmoe.runtime.model import ToyMoEConfig
from p2pmoe.runtime.wire import LinkTable
from p2pmoe.sim.network import SimNetwork

TASKS = [("X", 0.5), ("Y", 0.3), ("Z", 0.2)]
CTX_MAX = 64


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def toy_model_spec(cfg: ToyMoEConfig) -> ModelSpec:
    """把 toy 模型的**真实字节数**翻译成规划器的内存模型（单位统一取 MB）。

    这样规划器算出来的层切分，就是这些进程真的装得下的切分 —— 不是两套账。
    """
    mb = 1e6
    return ModelSpec(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        base_gb_per_layer=cfg.base_params * 8 / mb,
        expert_gb=cfg.expert_params * 8 / mb,
        ctx_max=CTX_MAX,
        kv_bytes_per_elem=8,
    )


def build_pool() -> list[Node]:
    """一个小而异构的分散池：5 台大内存 + 8 台小内存（单位 MB）。"""
    out = [Node(id=f"big{i+1}", tier="big", mem_gb=26.0, ms_per_layer=0.35, avail=0.97)
           for i in range(5)]
    out += [Node(id=f"sml{i+1}", tier="small", mem_gb=15.0, ms_per_layer=0.50, avail=0.94)
            for i in range(8)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--latency-scale", type=float, default=0.3,
                    help="延迟整体缩放。1.0 = 真实分散环境量级；小一点跑得快")
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--shared-clusters", type=int, default=0,
                    help="各 task 共用的词簇数。调大 → 驻留集重叠 → q 降低 → 触发池合并信号")
    args = ap.parse_args()

    mcfg = ToyMoEConfig()
    spec = toy_model_spec(mcfg)

    # ---------------- 1. 回放 → 画像 → 驻留集 ---------------------------- #
    rule("1. 回放语料 → 激活画像 → 驻留专家集")
    corpus = make_corpus(mcfg, [t for t, _ in TASKS], seed=args.seed,
                         shared_clusters=args.shared_clusters)
    t0 = time.perf_counter()
    profiles = profile_from_corpus(mcfg, corpus)
    plcs = {u: build_placement(p, args.coverage) for u, p in profiles.items()}
    uni = union_placement(list(plcs.values()))
    print(f"用**全专家**模型跑 {len(corpus['X'].sequences)} 条/task 的语料，"
          f"逐层统计路由质量（{(time.perf_counter()-t0)*1000:.0f}ms）")
    print(f"覆盖率阈值 {args.coverage} → 基线 miss 率 {1-args.coverage:.1%}")
    for u, p in plcs.items():
        print(f"  task {u}: n_u,l = {p.sizes()}  (层均 {sum(p.sizes())/mcfg.n_layers:.1f}"
              f"/{mcfg.n_experts})")
    print(f"  并集   : {uni.sizes()}  (层均 {sum(uni.sizes())/mcfg.n_layers:.1f})")

    q = detectability_matrix(profiles, plcs, 1, mcfg.n_layers)
    print("\nq(u,û) 可检性（行=真实，列=误绑）:")
    names = [t for t, _ in TASKS]
    print("        " + "".join(f"{v:>9}" for v in names))
    for a in names:
        print(f"  {a:<6}" + "".join(f"{q[(a,b)]:>9.3f}" for b in names))

    # ---------------- 2. 规划 -------------------------------------------- #
    rule("2. 离线规划")
    tasks = [TaskProfile(name=u, lam=l, experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u, l in TASKS]
    nodes = build_pool()
    sim = SimNetwork([n.id for n in nodes], seed=args.seed,
                     good_access=(12.0, 16.0), bad_access=(28.0, 33.0),
                     bad_frac=0.2, backbone=(2.0, 5.0), jitter=(4.0, 9.0))
    cfg = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                        kappa_over=0.3, n_standby=0, seed=args.seed)
    net = MeasurementCache(sim, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)
    p_curve = {l: 0.70 + 0.05 * l for l in range(2, mcfg.n_layers)}

    try:
        res = plan(nodes, spec, tasks, uni, net, cfg, p_curve, p_min=0.80)
    except PlanningError as e:
        for line in e.log:
            print("  " + line)
        print(f"\n{e}")
        return 2

    man = res.manifest
    print(f"L₀ = {res.l0}；配额 { {u: len(v) for u, v in res.backs.items()} }；"
          f"前段 {len(res.fronts_final)} 条；组合矩阵 {len(man.pairings)} 组")
    print(f"清单校验: {'通过' if man.ok else man.violations}")
    for p in sorted(man.nodes, key=lambda x: (x.role, x.segment, x.position)):
        lo, hi = p.layer_range
        print(f"  {p.node:<6} {p.role:<10} {p.segment:<5} 层 {lo}-{hi}  "
              f"{p.n_experts_total:>3} 个专家  {p.total_gb:>5.2f}MB")

    # ---------------- 3. 起集群 ------------------------------------------ #
    rule("3. 起进程集群")
    links = LinkTable(
        p50={(a, b): sim.true_p50(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        jitter={(a, b): sim.true_jitter(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        scale=args.latency_scale,
    )
    # 分类器参考也**实测**：在线时前段只装并集，直方图是截断重归一过的，
    # 拿全专家画像当参考会掉准确率（见 corpus.measure_front_refs）
    refs = measure_front_refs(mcfg, corpus, uni, res.l0)
    clf = HistogramClassifier(refs, {u: l for u, l in TASKS}, tau_hi=0.55, tau_lo=0.40)
    # 告警基线**实测**，不用 1−覆盖率（见 corpus.measure_baseline_miss 的说明）
    baselines = measure_baseline_miss(mcfg, corpus, uni, plcs, res.l0)
    nominal = {u: plcs[u].baseline_miss(res.l0 + 1, mcfg.n_layers) for u in plcs}
    print(f"节点进程 {len(man.nodes)} 个，延迟缩放 ×{args.latency_scale}")
    print(f"通道二基线 miss 率（实测）: { {u: f'{v:.1%}' for u, v in baselines.items()} }")
    print(f"  对比「1−覆盖率」的名义值: { {u: f'{v:.1%}' for u, v in nominal.items()} }"
          f"  —— 差 {np.mean([baselines[u]/max(nominal[u],1e-9) for u in baselines]):.1f}×，"
          f"用名义值当告警线会让绑对的池也持续误报")

    with LocalCluster(man, mcfg, links, clf, baselines=baselines,
                      priors={u: l for u, l in TASKS}, alarm_factor=3.0) as cl:
        cl.coord.max_tokens = args.tokens
        print(f"已起 {len(cl.procs)} 个进程，保温连接建立完毕")

        # ---------------- 4. 打请求 -------------------------------------- #
        rule("4. 请求生命周期")
        cases: list[tuple[str, str | None, list[int], str | None]] = []
        for i, (u, _) in enumerate(TASKS):
            cases.append((f"r{i}", u, sample_prompt(corpus, u, 12, seed=args.seed * 10 + i), None))
        # 故障注入：一条 X 的请求强行绑到 Y 池，看通道二能不能自己发现并纠正
        wrong = next(t for t, _ in TASKS if t != "X")
        cases.append(("r-inject", "X", sample_prompt(corpus, "X", 12, seed=777), wrong))

        recs = []
        for req, true_u, ids, force in cases:
            rec = cl.coord.submit(req, ids, true_task=true_u, force_task=force)
            ok = rec.done.wait(timeout=90)
            recs.append((rec, ok))
            tag = "正确" if rec.correct else ("—" if rec.correct is None else "误绑")
            print(f"\n▶ {req}  真实 task = {true_u or '（未知）'}"
                  + (f"  [注入误绑到 {force}]" if force else "") + "  "
                  f"→ 识别 {rec.task} [{tag}]  换绑 {rec.rebinds} 次"
                  + ("" if ok else "  ⚠ 超时"))
            for e in rec.events:
                print("    " + e)

        # ---------------- 5. 账本 ---------------------------------------- #
        rule("5. 账本")
        print(f"{'请求':<7} {'识别':<6} {'区':<8} {'置信':>6} {'换绑':>4} "
              f"{'首token':>9} {'逐token p50':>12} {'token 数':>8}")
        for rec, ok in recs:
            per = sorted(rec.token_ms)
            p50 = per[len(per) // 2] if per else 0.0
            print(f"{rec.req:<7} {str(rec.task):<6} {rec.zone:<8} {rec.conf:>6.2f} "
                  f"{rec.rebinds:>4} {(rec.t_first-rec.t0)*1000:>8.1f}ms "
                  f"{p50:>11.1f}ms {len(rec.tokens):>8}")

        ident = [r for (r, _), c in zip(recs, cases) if r.correct is not None and c[3] is None]
        if ident:
            print(f"\n识别准确率: {sum(r.correct for r in ident)}/{len(ident)}")
        if cl.coord.errors:
            print("\n⚠ 错误:")
            for e in cl.coord.errors[:5]:
                print("  " + e.replace("\n", " ")[:200])
            return 1

    print("\n集群已关闭。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
