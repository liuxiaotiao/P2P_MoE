#!/usr/bin/env python3
"""**静态简化版**：真模型 + 部署时定死的前后段链路，跑出真 token。

    python examples/static_qwen.py                      # 用合成的微型 checkpoint
    python examples/static_qwen.py --model-dir /path/to/Qwen3-30B-A3B --ctx 2048

它和 `serving.py`（文档主线）的区别只有两条，但这两条把在线部分削掉了一大半：

| | 主线（serving.py） | 静态（本脚本） |
|---|---|---|
| 前段驻留 | 各 task 驻留集的**并集** ∪_u S_{u,l} | **全部专家**（并集的保守上界） |
| task 来源 | 前段 tail 在线识别 | 部署时给定，请求自报 |
| 配对 | 到达时从空闲池 pop，任意组合 | 离线算好，写进节点配置 |
| 在线控制面 | 识别 → 问协调器要后段（一个 RTT） | 无 |
| 换绑 | 通道二 miss 检出触发 | 不存在（没有误绑） |

放弃了什么，要说清楚：**「到达时 task 未知」这个前提（I.1.1）被放弃了**，
于是识别、检出、换绑、公共中值域这一整套都不再需要 —— 不是它们被做好了，
是问题被换成了一个更简单的问题。同时负载不能在通道之间流动：某个 task 忽然
变热，只能重新下发配对。

保留了什么：**逐节点逐层的放置、每层只驻留指定专家、段内流水、绕环 decode、
drop-expert 近似、有界等待** —— 也就是这套方案真正吃内存的那部分，一条没少。

前段为什么装全部专家
--------------------
并集要先有各 task 的真实激活画像才算得出来，而画像要跑过真实语料；全集不需要，
部署当天就能装，且将来加 task 时前段不用重装。代价是内存 —— 见
`planner/experts.py::full_placement` 的说明与 `examples/model_fit.py` 的实际账。

关于合成 checkpoint
-------------------
不给 `--model-dir` 时用 `sim/fake_checkpoint.py` 生成一个 4 层 8 专家的微型模型，
**key 命名与真实 Qwen3-MoE 一字不差**。它验证的是*机制*：选择性加载确实只打开
需要的分片与 key、层内结构接得上、KV 语义对、token 真的从环里绕出来。
它验证不了*数值* —— 权重是随机的，输出的 token 没有语义。要看有意义的输出得
`--model-dir` 指到真权重，那时还需要 tokenizer（TODO.md）。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from p2pmoe.planner.experts import build_placement, full_placement, union_placement
from p2pmoe.planner.hf_config import granularity_verdict, model_spec_from_hf
from p2pmoe.planner.memory import choose_l0
from p2pmoe.planner.network import MeasurementCache
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.planner.static_pairing import assign_static_pairs
from p2pmoe.planner.types import Node, PlannerConfig, TaskProfile
from p2pmoe.runtime.coordinator import LocalCluster
from p2pmoe.runtime.text import TextIO
from p2pmoe.runtime.wire import LinkTable
from p2pmoe.sim.network import SimNetwork
from p2pmoe.sim.replay import make_activation_profiles


DEFAULT_PROMPTS = [
    "the quick brown fox",
    "把请求打进前段",
    "front segment back segment",
]


def rec_head(s: str, n: int = 48) -> str:
    return repr(s if len(s) <= n else s[:n] + "…")


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def build_pool(spec_str: str, reserve_gb: float) -> list[Node]:
    """'16x8,24x7' → 8 台 16GB + 7 台 24GB。"""
    out: list[Node] = []
    for part in spec_str.split(","):
        mem_s, _, cnt_s = part.strip().partition("x")
        mem, cnt = float(mem_s), int(cnt_s or 1)
        for _ in range(cnt):
            out.append(Node(id=f"n{len(out)+1}", tier=f"{mem:.0f}GB", mem_gb=mem,
                            ms_per_layer=0.35, reserve_gb=reserve_gb, avail=0.95))
    if not out:
        raise SystemExit("--pool 解析为空，格式如 '16x8,24x7'")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="真实 Qwen3-MoE checkpoint 目录。不给就生成一个微型合成的")
    ap.add_argument("--preset", default=None,
                    help="只用某个已知模型的 config 做规划（qwen3-30b-a3b 等），"
                         "不需要权重。隐含 --plan-only")
    ap.add_argument("--plan-only", action="store_true",
                    help="只跑离线三步（放置 → 建段 → 定死配对），不启节点。"
                         "下载 61GB 之前先用它确认前段装全集到底放不放得下")
    ap.add_argument("--pool", default=None,
                    help="节点池，如 '16x8,24x7'。合成模型下默认用一个微型池")
    ap.add_argument("--reserve-gb", type=float, default=None)
    ap.add_argument("--ctx", type=int, default=None, help="KV 预算的上下文上限")
    ap.add_argument("--tasks", default="X,Y,Z")
    ap.add_argument("--resident-frac", type=float, default=0.30,
                    help="后段每层驻留多少比例的专家（前段固定为全部）")
    ap.add_argument("--prompt", action="append", default=None,
                    help="文本 prompt，可重复给多条。不给就用内置的几条。"
                         "checkpoint 里没有 tokenizer.json 时退回随机 token id")
    ap.add_argument("--chat", action="store_true",
                    help="套 checkpoint 自带的对话模板。指令模型**必须**加这个 —— "
                         "不加是 completion 语义，输出会像坏了")
    ap.add_argument("--system", default=None, help="--chat 时的 system 提示")
    ap.add_argument("--no-text", action="store_true",
                    help="强制走 id 进 id 出（只验协议，不要文本层）")
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--latency-scale", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    tasks_n = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tmp: tempfile.TemporaryDirectory | None = None

    # ---------------- 0. checkpoint ------------------------------------- #
    rule("0. checkpoint")
    if args.preset:
        from p2pmoe.planner.hf_config import PRESETS
        if args.preset not in PRESETS:
            raise SystemExit(f"未知 preset {args.preset}；有 {sorted(PRESETS)}")
        args.plan_only = True
        mdir, synthetic, hf = Path(args.preset), False, dict(PRESETS[args.preset])
    elif args.model_dir:
        mdir = Path(args.model_dir)
        synthetic = False
    else:
        from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint
        tmp = tempfile.TemporaryDirectory(prefix="p2pmoe-ckpt-")
        mdir = write_fake_checkpoint(tmp.name, TINY_QWEN3_MOE, seed=args.seed)
        synthetic = True
    if not args.preset:
        hf = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    ctx = args.ctx or (128 if synthetic else 2048)
    pool_s = args.pool or ("1x9" if synthetic else "16x8,24x7")
    reserve = args.reserve_gb if args.reserve_gb is not None else (0.0 if synthetic else 1.0)

    spec, info = model_spec_from_hf(hf, name=mdir.name, ctx_max=ctx,
                                    dtype_bytes=2 if not synthetic else 4)
    print(f"  {mdir}" + ("   ⚠ 合成权重，输出无语义，只验机制" if synthetic
                          else ("   （只有 config，不加载权重）" if args.preset else "")))
    print("  " + info.summary())
    ok, why = granularity_verdict(info)
    print(f"  {'✓' if ok else '✗'} {why}")
    if not ok and not synthetic:
        print("\n  细粒度比不够 —— 「只驻留子集」省不下内存，换模型。")
        return 2

    # ---------------- 1. 离线：放置 -------------------------------------- #
    rule("1. 离线：前段装全部专家，后段按 task 装子集")
    n_back = max(info.top_k, round(args.resident_frac * info.n_experts))
    profiles = make_activation_profiles(
        tasks_n, {u: n_back for u in tasks_n}, n_layers=info.n_layers,
        n_experts=info.n_experts, seed=args.seed, shared_core=1,
    )
    plcs = {u: build_placement(p, 0.95) for u, p in profiles.items()}
    front_plc = full_placement(info.n_layers, info.n_experts)
    lam = {u: v / sum(range(1, len(tasks_n) + 1))
           for u, v in zip(tasks_n, range(len(tasks_n), 0, -1))}
    tasks = [TaskProfile(name=u, lam=lam[u],
                         experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u in tasks_n]
    print(f"  前段每层 {info.n_experts}/{info.n_experts} 个专家（全集，覆盖率 1.0）")
    for u in tasks_n:
        print(f"  后段 {u} 每层 {plcs[u].sizes()[:6]}… "
              f"（层均 {np.mean(plcs[u].as_experts_per_layer()):.1f}/{info.n_experts}）")

    # ---------------- 2. 离线：规划 -------------------------------------- #
    rule("2. 离线：选 L₀ → 建段")
    nodes = build_pool(pool_s, reserve)
    print(f"  池：{len(nodes)} 台（{pool_s}），预留 {reserve}GB/台，ctx={ctx}")
    sim = SimNetwork([n.id for n in nodes], seed=args.seed, good_access=(12.0, 16.0),
                     bad_access=(28.0, 33.0), bad_frac=0.2, backbone=(2.0, 5.0),
                     jitter=(4.0, 9.0))
    cfg = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                        kappa_over=0.3, n_standby=0, seed=args.seed)
    net = MeasurementCache(sim, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)
    p_curve = {l: min(0.97, 0.70 + 0.05 * l) for l in range(1, info.n_layers)}
    try:
        res = plan(nodes, spec, tasks, front_plc, net, cfg, p_curve, p_min=0.75)
    except PlanningError as e:
        print(e)
        print("\n  前段装全集是有代价的 —— 内存不够时先降 --resident-frac 或加大机器。")
        return 2
    man = res.manifest
    print(f"  L₀ = {res.l0}（前段 1..{res.l0}，后段 {res.l0+1}..{info.n_layers}）")
    chosen = next(c for c in res.l0_table if c.l0 == res.l0)
    print(f"  前段驻留 {chosen.front_gb:.1f}GB（全集），可单节点承载的机器 "
          f"{chosen.front_single_node_count}/{len(nodes)} 台")

    # 装全集到底贵在哪 —— 拿并集做同一次 L₀ 选择，差额就是这个简化的价码。
    uni = union_placement(list(plcs.values()))
    try:
        u_best, _ = choose_l0(spec, tasks, uni, nodes, p_curve, p_min=0.75)
        u_gb = u_best.front_gb
        print(f"  ── 对照：若前段只装并集（{np.mean([len(uni.at(l)) for l in range(1, res.l0+1)]):.0f}"
              f"/{info.n_experts} 个/层），L₀ 可到 {u_best.l0}、前段 {u_gb:.1f}GB、"
              f"通道上限 {u_best.n_channels}（现在 {chosen.n_channels}）")
        print("     差额就是「不用画像、加 task 不重装前段」这份省事的价码。")
    except Exception:
        print("  ── 对照：并集口径下无可行 L₀（合成画像的规模不适配）")
    print(f"  前段 {len(res.fronts_final)} 条：{[s.label() for s in res.fronts_final]}")
    for u, segs in res.backs.items():
        print(f"  后段 {u}: {len(segs)} 条 {[s.label() for s in segs]}")

    # ---------------- 3. 离线：定死配对 ---------------------------------- #
    rule("3. 离线：把链路定死")
    front_ids = [f"F{i}" for i in range(len(res.fronts_final))]
    wiring = assign_static_pairs(res.fronts_final, res.backs, net,
                                 k=cfg.k_audit, front_ids=front_ids)
    print("  " + wiring.summary())
    print(f"\n  {'通道':<8} {'task':<6} {'前段':<6} {'后段':<8} {'正向':>8} "
          f"{'回环':>8} {'组合p50':>9}")
    for i, p in enumerate(wiring.pairs):
        print(f"  ch{i:<6} {p.task:<6} {p.front_id:<6} {p.back_id:<8} "
              f"{p.w_fwd:>7.1f}ms {p.d_loop:>7.1f}ms {p.t50:>8.1f}ms")
    if not wiring.pairs:
        print("  没配出通道 —— 池子太小。")
        return 2
    wired = wiring.as_map()
    served = sorted({t for _, t in wired.values()})
    print(f"\n  → 有通道的 task: {served}"
          + (f"；⚠ 没通道的: {sorted(set(tasks_n) - set(served))}"
             if set(tasks_n) - set(served) else ""))
    print("  → 这张表在节点加载时就写进配置了：前段 tail 只认识自己那一个后段 head，"
          "\n    不问协调器、不识别、不换绑。")

    if args.plan_only:
        print("\n  --plan-only：离线部分到此为止。把上面这张表交给 "
              "deploy/control.py 就能下发到真机。")
        if tmp:
            tmp.cleanup()
        return 0

    # ---------------- 4. 在线：写死 task 的循环测试 ---------------------- #
    rule(f"4. 在线：{args.requests} 条请求轮着打（task 由脚本写死）")
    links = LinkTable(
        p50={(a, b): sim.true_p50(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        jitter={(a, b): sim.true_jitter(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        scale=args.latency_scale,
    )
    # ---- 文本层（控制机侧，节点不受影响） ---- #
    textio = None
    if not args.no_text:
        try:
            textio = TextIO.from_model_dir(mdir, chat=args.chat, system=args.system)
            print(f"  tokenizer: vocab {textio.tok.vocab_size}，"
                  f"停止 token {sorted(textio.stop.ids)}"
                  + ("，套对话模板" if args.chat else "，completion 模式"))
        except (FileNotFoundError, ImportError) as e:
            print(f"  ⚠ 没有文本层（{type(e).__name__}: {str(e)[:60]}…）→ 退回 id 进 id 出")
    prompts = args.prompt or DEFAULT_PROMPTS

    rng = np.random.default_rng(args.seed)
    t_load = time.perf_counter()

    def stream(rec, delta: str) -> None:
        print(delta, end="", flush=True)

    with LocalCluster(man, None, links, static_wiring=wired, backend="torch",
                      model_dir=str(mdir), model_hf=hf, device=args.device,
                      textio=textio, on_text=stream if textio else None) as cl:
        print(f"  {len(man.nodes)} 个节点加载完毕，用时 {time.perf_counter()-t_load:.1f}s")
        coord = cl.coord
        coord.max_tokens = args.tokens

        done = []
        for i in range(args.requests):
            u = served[i % len(served)]
            if textio:
                prompt = prompts[i % len(prompts)]
                print(f"  r{i:02d} [{u}] {rec_head(prompt)}\n      ", end="", flush=True)
                rec = coord.submit(f"r{i:02d}", text=prompt, task=u)
            else:
                ids = rng.integers(0, info.vocab, size=args.prompt_len).tolist()
                rec = coord.submit(f"r{i:02d}", ids, task=u)   # ← task 写死，不识别
            if not rec.done.wait(timeout=300):
                print(f"\n  ⚠ {rec.req} 超时")
                for e in rec.events:
                    print("      " + e)
                return 1
            done.append(rec)
            if textio:
                print(f"\n      └ {rec.front}×{rec.back}  {len(rec.tokens)} token  "
                      f"首token {(rec.t_first-rec.t0)*1000:.0f}ms  停于 {rec.stop_reason}")
            else:
                print(f"  r{i:02d}  task={u}  {rec.front}×{rec.back}  "
                      f"首token {(rec.t_first-rec.t0)*1000:6.0f}ms  "
                      f"出 {len(rec.tokens)} 个 token: {rec.tokens[:6]}"
                      f"{'…' if len(rec.tokens) > 6 else ''}")

        rule("账本")
        per = sorted(m for r in done for m in r.token_ms)
        n_tok = sum(len(r.tokens) for r in done)
        print(f"  {len(done)} 条请求，共 {n_tok} 个 token，逐 token p50 "
              f"{per[len(per)//2]:.0f}ms（延迟按 ×{args.latency_scale} 缩放）")
        used = {(r.front, r.back) for r in done}
        print(f"  出现过的组合 {len(used)} 种 = 通道数 {len(wiring.pairs)} 中被打到的部分"
              f" —— 静态模式下组合是定死的，不会有新组合")
        print(f"  换绑 {sum(r.rebinds for r in done)} 次（静态模式恒为 0）")
        if textio:
            why = {}
            for r in done:
                why[r.stop_reason] = why.get(r.stop_reason, 0) + 1
            print(f"  停止原因: {why}")
        print(f"  最终池深: {coord.queue_depths()}")
        if coord.errors:
            print("\n  ⚠ 错误:")
            for e in coord.errors[:3]:
                print("    " + e.replace("\n", " ")[:300])
            return 1

    if synthetic:
        print("\n  提醒：权重是随机的，所以上面那些字是乱码 —— 它说明的是"
              "**文本进得去、token 绕得出环、字节拼得回字符**，不是模型对。")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
