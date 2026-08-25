#!/usr/bin/env python3
"""服务循环 demo —— 「打请求进来，前后段自动配对，跑完回池，下一条重新配」。

    python examples/serving.py [--requests 12] [--concurrency 8]

这个脚本回答的问题是：**离线规划完之后，日常怎么用？**

规划的产物是一个「可用池」——若干条前段、若干条后段（按 task 分池）。之后就只有
一个循环：

    请求到达 → 从前段池 pop 一条（盲绑，不比不挑）
             → 前段跑完，本地识别出 task
             → 从该 task 的后段池 pop 一条 → 两段自动建链
             → decode 绕环直到完成
             → 两段各自回池 → 等下一条请求

**池子空了不是错误，是排队**（文档 II.5「池满 → 有界等待」）。15 台节点建出 5 条
前段就只能同时服务 5 条，第 6 条在队列里等 —— 前面哪条一完成，队首立刻被接走，
不用等整批。

跑完会打出配对历史，可以直接核对：同一条前段在不同请求里配了不同的后段，
反之亦然 —— 这正是「任意组合」在服务循环里的样子。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from p2pmoe.planner.experts import build_placement, union_placement
from p2pmoe.planner.network import MeasurementCache
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.planner.types import PlannerConfig, TaskProfile
from p2pmoe.runtime.corpus import (
    make_corpus,
    measure_baseline_miss,
    measure_front_refs,
    profile_from_corpus,
    sample_prompt,
)
from p2pmoe.runtime.coordinator import LocalCluster
from p2pmoe.runtime.identify import HistogramClassifier
from p2pmoe.runtime.model import ToyMoEConfig
from p2pmoe.runtime.wire import LinkTable
from p2pmoe.sim.network import SimNetwork

from e2e import TASKS, build_pool, toy_model_spec  # noqa: E402


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="同时打入多少条。设成大于池子容量才看得到排队")
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--latency-scale", type=float, default=0.25)
    args = ap.parse_args()

    mcfg = ToyMoEConfig()
    spec = toy_model_spec(mcfg)

    # ---- 离线：画像 + 规划（与 e2e.py 相同，这里不是重点） ---------------- #
    corpus = make_corpus(mcfg, [t for t, _ in TASKS], seed=args.seed, shared_clusters=0)
    profiles = profile_from_corpus(mcfg, corpus)
    plcs = {u: build_placement(p, 0.95) for u, p in profiles.items()}
    uni = union_placement(list(plcs.values()))
    tasks = [TaskProfile(name=u, lam=l, experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u, l in TASKS]
    nodes = build_pool()
    sim = SimNetwork([n.id for n in nodes], seed=args.seed, good_access=(12.0, 16.0),
                     bad_access=(28.0, 33.0), bad_frac=0.2, backbone=(2.0, 5.0),
                     jitter=(4.0, 9.0))
    cfg = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                        kappa_over=0.3, n_standby=0, seed=args.seed)
    net = MeasurementCache(sim, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)
    try:
        res = plan(nodes, spec, tasks, uni, net, cfg,
                   {l: 0.70 + 0.05 * l for l in range(2, mcfg.n_layers)}, p_min=0.80)
    except PlanningError as e:
        print(e)
        return 2
    man = res.manifest

    links = LinkTable(
        p50={(a, b): sim.true_p50(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        jitter={(a, b): sim.true_jitter(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        scale=args.latency_scale,
    )
    refs = measure_front_refs(mcfg, corpus, uni, res.l0)
    clf = HistogramClassifier(refs, {u: l for u, l in TASKS})
    baselines = measure_baseline_miss(mcfg, corpus, uni, plcs, res.l0)

    rule("可用池（离线规划的产物）")
    n_f = len(res.fronts_final)
    n_b = {u: len(v) for u, v in res.backs.items()}
    print(f"  前段池: {n_f} 条  {[s.label() for s in res.fronts_final]}")
    for u, segs in res.backs.items():
        print(f"  后段池 {u}: {len(segs)} 条  {[s.label() for s in segs]}")
    print(f"\n  → 同时最多服务 {n_f} 条请求（前段独占，I.2.4）；"
          f"第 {n_f+1} 条起排队")
    print(f"  → 组合矩阵 {len(man.pairings)} 组，任意前段可配任意同 task 后段")

    with LocalCluster(man, mcfg, links, clf, baselines=baselines,
                      priors={u: l for u, l in TASKS}, alarm_factor=3.0) as cl:
        coord = cl.coord
        coord.max_tokens = args.tokens

        rule(f"服务循环：{args.requests} 条请求，并发 {args.concurrency}")
        names = [t for t, _ in TASKS]
        done: list = []
        lock = threading.Lock()

        def submit_batch(lo: int, hi: int):
            batch = []
            for i in range(lo, hi):
                u = names[i % len(names)]
                batch.append(coord.submit(
                    f"r{i:02d}", sample_prompt(corpus, u, 12, seed=2000 + i), true_task=u))
            return batch

        t0 = time.perf_counter()
        i = 0
        while i < args.requests:
            hi = min(i + args.concurrency, args.requests)
            batch = submit_batch(i, hi)
            d = coord.queue_depths()
            print(f"\n  打入 r{i:02d}–r{hi-1:02d}（{len(batch)} 条）→ "
                  f"空闲前段 {d['free_fronts']}，排队 {d['waiting_front']}")
            for rec in batch:
                if not rec.done.wait(timeout=180):
                    print(f"  ⚠ {rec.req} 超时")
                    for e in rec.events:
                        print("      " + e)
                    return 1
                with lock:
                    done.append(rec)
            i = hi
        elapsed = time.perf_counter() - t0

        # ---------------- 结果 ------------------------------------------- #
        rule("每条请求用了哪对段")
        print(f"  {'请求':<7} {'task':<5} {'前段':<10} {'后段':<10} "
              f"{'前段排队':>9} {'后段排队':>9} {'首token':>9}")
        for rec in done:
            print(f"  {rec.req:<7} {str(rec.task):<5} {rec.front:<10} {rec.back:<10} "
                  f"{rec.wait_front_ms:>8.0f}ms {rec.wait_back_ms:>8.0f}ms "
                  f"{(rec.t_first-rec.t0)*1000:>8.0f}ms")

        rule("核对：段是不是被复用了")
        f_use = Counter(r.front for r in done)
        b_use = Counter(r.back for r in done)
        print("  前段使用次数: " + ", ".join(f"{k}×{v}" for k, v in sorted(f_use.items())))
        print("  后段使用次数: " + ", ".join(f"{k}×{v}" for k, v in sorted(b_use.items())))

        pairs = {(r.front, r.back) for r in done}
        print(f"\n  出现过的组合: {len(pairs)} 种")
        for f in sorted(f_use):
            bs = sorted({b for (ff, b) in pairs if ff == f})
            print(f"    {f:<10} 配过 {bs}")
        multi = [f for f in f_use if len({b for (ff, b) in pairs if ff == f}) > 1]
        print(f"\n  → {len(multi)}/{len(f_use)} 条前段在不同请求里配了**不同的后段** "
              f"—— 这就是「跑完回池、下次重新配」")

        rule("账本")
        waited = [r for r in done if r.wait_front_ms > 1]
        per = sorted(m for r in done for m in r.token_ms)
        print(f"  完成 {len(done)} 条，用时 {elapsed:.1f}s")
        print(f"  排过队的: {len(waited)} 条，最长等 "
              f"{max((r.wait_front_ms for r in waited), default=0):.0f}ms")
        print(f"  逐 token p50: {per[len(per)//2]:.0f}ms")
        ok = sum(bool(r.correct) for r in done)
        print(f"  识别正确: {ok}/{len(done)}，换绑 {sum(r.rebinds for r in done)} 次")
        print(f"  最终池深: {coord.queue_depths()}")
        if coord.errors:
            print("\n  ⚠ 错误:")
            for e in coord.errors[:3]:
                print("    " + e.replace("\n", " ")[:200])
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
