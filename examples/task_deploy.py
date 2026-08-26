#!/usr/bin/env python3
"""把文档第二部分的算法，跑在**真实数据**上。

    python examples/task_deploy.py --data ./task --tasks gsm8k=3,mbpp=5 \
        --coverage 0.90 --save-spec deploy.json

输入三样东西，全是量出来的、不是编的：

* `*_expert_activation.csv` —— 逐层逐专家的激活（Qwen3-Next-80B，48 层 × 512 专家）
* `topo_fabric_mbps.json`   —— 15 节点的算力/显存/带宽/传播时延
* 模型 config                —— 决定内存形状与层型

产出：L₀、逐 task 的后段驻留集、逐节点的加载清单、前后段连线。

它与 `deploy/control.py` 的分工：那个是**在线**跑（探测真机、下发清单），
这个是**离线**跑（数据已经在手上，只算放置）。两者产出同一种
`DeploymentManifest`，所以算完可以直接 `--save-spec` 交给 `deploy/run.py` 部署。

p(L₀) 是从数据里**算出来**的，不是假设的
------------------------------------------
文档 III.5.3 用一条 p(L₀) 曲线（用前 L₀ 层能不能认出 task）来选切点，但没说
那条曲线从哪来。这里从激活数据直接推：两个 task 在每层的门控质量分布差多少
（Bhattacharyya 距离），逐层累加，转成两类等先验下的判别准确率下界。

各层近似独立是个**简化**：真实的路由沿层相关（前面选了什么会影响后面），
所以累加会**高估**可分性。要更准就得逐请求的激活轨迹，而 CSV 只有聚合量。
这条曲线因此是「乐观上界」，用它选出的 L₀ 偏小 —— 保守起见该往大了取一点。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from p2pmoe.planner.experts import ExpertPlacement, build_placement, union_placement
from p2pmoe.planner.from_data import FabricOracle, load_activation_csv, load_topology
from p2pmoe.planner.hf_config import granularity_verdict, model_spec_from_hf
from p2pmoe.planner.network import MeasurementCache
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.planner.static_pairing import assign_static_pairs
from p2pmoe.planner.types import PlannerConfig, TaskProfile

# Qwen3-Next-80B-A3B 的 config（取自 HF 的 config.json）
QWEN3_NEXT_80B = {
    "architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
    "num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 2,
    "head_dim": 256, "full_attention_interval": 4,
    "linear_num_value_heads": 32, "linear_num_key_heads": 16,
    "linear_value_head_dim": 128, "linear_key_head_dim": 128,
    "linear_conv_kernel_dim": 4, "num_experts": 512, "num_experts_per_tok": 10,
    "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    "hidden_size": 2048, "intermediate_size": 5120, "vocab_size": 151936,
    "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "norm_topk_prob": True,
}


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def identifiability(profiles: dict[str, "object"], n_layers: int) -> np.ndarray:
    """p(L₀)：用前 L₀ 层的激活特征区分各 task 的准确率下界。

    两两之间算 Bhattacharyya 距离逐层累加，取**最难分的那一对**（下界要对
    最坏情况成立），再用 Chernoff 界 P(err) ≤ ½·exp(−D_B) 转成准确率。
    """
    names = sorted(profiles)
    worst = np.full(n_layers, np.inf)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa = np.array([profiles[a].at(l + 1) for l in range(n_layers)])
            pb = np.array([profiles[b].at(l + 1) for l in range(n_layers)])
            bc = np.sqrt(pa * pb).sum(1)                    # 逐层 BC
            worst = np.minimum(worst, np.cumsum(-np.log(np.clip(bc, 1e-12, 1))))
    return 1.0 - 0.5 * np.exp(-worst)


def knee(x: np.ndarray, y: np.ndarray) -> int:
    """最大弦距法找膝点：曲线上离首末连线最远的那点。"""
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / np.hypot(y1 - y0, x1 - x0)
    return int(d.argmax())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./task", help="放 CSV 与 topo JSON 的目录")
    ap.add_argument("--tasks", default="gsm8k=3,mbpp=5",
                    help="task 名与到达率比，如 'gsm8k=3,mbpp=5'")
    ap.add_argument("--coverage", type=float, default=0.90,
                    help="后段每层保留多少激活质量")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--reserve-gb", type=float, default=1.0)
    ap.add_argument("--phase", default="decode", choices=("decode", "prefill"))
    ap.add_argument("--l0", type=int, default=None, help="强制切点（默认让规划器选）")
    ap.add_argument("--jitter-frac", type=float, default=0.0,
                    help="人为注入与 p50 成比例的抖动做敏感性分析。"
                         "拓扑本身没有抖动维度 —— 默认 0，见 from_data.FabricOracle")
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--j-cap", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--save-plan", type=Path, default=None)
    ap.add_argument("--save-spec", type=Path, default=None,
                    help="导出 deploy/run.py 能吃的布局文件")
    args = ap.parse_args()

    D = Path(args.data)
    pairs = [x.split("=") for x in args.tasks.split(",")]
    weights = {k.strip(): float(v) for k, v in pairs}
    tot = sum(weights.values())
    lam = {k: v / tot for k, v in weights.items()}

    # ---------------- 1. 数据 ---------------- #
    rule("1. 读数据")
    profiles, diags = {}, {}
    for name in lam:
        f = D / f"{name}_expert_activation.csv"
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        profiles[name], diags[name] = load_activation_csv(f, task=name, phase=args.phase)
        d = diags[name]
        warn = ""
        if d["n_suspect"]:
            sh = max(d["suspect_count_share"].values())
            warn = (f"   ⚠ {d['n_suspect']} 个格子「计数非零但权重是 NaN」"
                    f"（专家 {d['suspect_experts'][:4]}…，最多占该层 {sh:.0%} 的计数）"
                    f"—— 采集端的 bug，这些专家会被当成没激活")
        print(f"  {name:<8} {d['n_layers']} 层 × {d['n_experts']} 专家，{args.phase} 相{warn}")
    topo = load_topology(D / "topo_fabric_mbps.json", d_model=QWEN3_NEXT_80B["hidden_size"],
                         dtype_bytes=args.dtype_bytes, reserve_gb=args.reserve_gb)
    v = np.array(list(topo.p50_ms.values()))
    tiers = {}
    for n in topo.nodes:
        tiers.setdefault((n.tier, n.mem_gb), []).append(n.id)
    print(f"  拓扑     {len(topo.nodes)} 节点 / {topo.reachable_pairs} 个有向对；"
          f"单向 p50 {v.min():.2f}–{v.max():.2f}ms（中位 {np.median(v):.2f}）")
    for (t, m), ids in sorted(tiers.items(), key=lambda kv: -kv[0][1]):
        print(f"           {t:<8} {m:.0f}GB × {len(ids)}  {ids}")
    if args.jitter_frac == 0:
        print("  ⚠ 拓扑没有抖动维度（只有确定的传播时延）→ 尾闸与抖动闸恒定通过，"
              "\n    规划出的通道数偏乐观。--jitter-frac 可注入抖动做敏感性分析。")

    # ---------------- 2. 模型 ---------------- #
    rule("2. 模型")
    spec, info = model_spec_from_hf(QWEN3_NEXT_80B, name="qwen3-next-80b",
                                    ctx_max=args.ctx, dtype_bytes=args.dtype_bytes)
    print(f"  {info.summary()}")
    print(f"  混合层：{sum(1 for t in info.layer_types if t=='linear_attention')} 线性 "
          f"+ {sum(1 for t in info.layer_types if t=='full_attention')} 标准；"
          f"全模型 {info.total_gb:.1f}GB")
    ok, why = granularity_verdict(info)
    print(f"  {'✓' if ok else '✗'} {why}")

    # ---------------- 3. 拐点 ---------------- #
    rule("3. 前段该到哪一层：p(L₀) 与拐点")
    L = info.n_layers
    acc = identifiability(profiles, L)
    plc = {u: build_placement(p, args.coverage) for u, p in profiles.items()}
    uni_sz = np.array([len(union_placement(list(plc.values())).at(l + 1)) for l in range(L)])
    k_acc = knee(np.arange(1, L + 1, dtype=float), acc) + 1
    print(f"  {'L₀':>3} {'p(L₀)≥':>8} {'边际增益':>9} {'本层前段并集':>12} {'累积':>8}")
    cum = np.cumsum(uni_sz)
    prev = 1.0 / (len(lam) or 1)
    for l in range(1, min(L, 18) + 1):
        g = acc[l - 1] - prev
        prev = acc[l - 1]
        star = " ← 膝点" if l == k_acc else ""
        print(f"  {l:>3} {acc[l-1]:>7.2%} {g:>+9.2%} {uni_sz[l-1]:>12} {cum[l-1]:>8}{star}")
    print(f"\n  膝点（最大弦距法）：L₀ = {k_acc}，此时 p ≥ {acc[k_acc-1]:.2%}")
    print(f"  达到 99% 需要 L₀ = {int(np.argmax(acc>=0.99))+1}；"
          f"99.9% 需要 L₀ = {int(np.argmax(acc>=0.999))+1}")
    sep = [len(plc[u].at(l+1)) for u in plc for l in range(L)]
    print(f"  各 task 单独装：层均 {np.mean(sep):.0f}/{info.n_experts}；"
          f"并集层均 {uni_sz.mean():.0f} → 共用比各装省 "
          f"{1 - uni_sz.sum()/sum(sep):.1%}")

    # ---------------- 4. 规划 ---------------- #
    rule("4. 六步流水线（II.4）")
    uni = union_placement(list(plc.values()))
    tasks = [TaskProfile(name=u, lam=lam[u],
                         experts_per_layer=plc[u].as_experts_per_layer(),
                         placement=plc[u]) for u in sorted(lam)]
    cfg = PlannerConfig(eta=args.eta, beta=1.3, j_cap_ms=args.j_cap, theta=0.8,
                        kappa_over=0.3, n_standby=0, seed=args.seed)
    oracle = FabricOracle(topo, jitter_frac=args.jitter_frac)
    net = MeasurementCache(oracle, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms,
                           k_gate=cfg.k_gate)
    p_curve = {l: float(acc[l - 1]) for l in range(1, L)}
    if args.l0:
        # choose_l0 遍历的就是 p_curve 的键 —— 只留一个即等于强制切点
        p_curve = {args.l0: p_curve[args.l0]}
    try:
        res = plan(topo.nodes, spec, tasks, uni, net, cfg, p_curve, p_min=0.80)
    except PlanningError as e:
        for line in e.log:
            print("  " + line)
        print(f"\n规划失败：{e}")
        return 2
    man = res.manifest
    for line in res.log:
        print("  " + line)

    rule("5. 结果")
    print(f"  L₀ = {res.l0}（前段 1..{res.l0}，后段 {res.l0+1}..{L}）")
    print(f"  配额 {res.quota}（目标比 {dict((k, round(v,3)) for k,v in lam.items())}）")
    print(f"  前段 {len(res.fronts_final)} 条：{[s.label() for s in res.fronts_final]}")
    for u, segs in sorted(res.backs.items()):
        print(f"  后段 {u}: {len(segs)} 条 {[s.label() for s in segs]}")
    print(f"  组合矩阵 {len(man.pairings)} 组；清单校验 "
          f"{'通过' if man.ok else f'未通过 {man.violations[:2]}'}")

    wiring = assign_static_pairs(res.fronts_final, res.backs, net, k=cfg.k_audit)
    print(f"\n  静态链路：{wiring.summary()}")
    for i, p in enumerate(wiring.pairs):
        print(f"    ch{i}  {p.task:<7} {p.front_id}{tuple(p.front.nodes)} → "
              f"{p.back_id}{tuple(p.back.nodes)}   组合 p50 {p.t50:.1f}ms")

    print(f"\n  {'节点':<6}{'角色':<16}{'层':<10}{'专家数':>7}{'权重GB':>9}{'合计GB':>9}")
    for p in man.nodes:
        n = next(x for x in topo.nodes if x.id == p.node)
        print(f"  {p.node:<6}{p.role:<16}{p.layer_range[0]}–{p.layer_range[1]:<7}"
              f"{sum(len(l.experts) for l in p.layers):>7}"
              f"{p.weight_gb:>9.2f}{p.total_gb:>9.2f}   / {n.mem_gb:.0f}GB")

    if args.save_plan:
        args.save_plan.write_text(man.to_json(), encoding="utf-8")
        print(f"\n  清单已存 {args.save_plan}")
    if args.save_spec:
        by_seg = {}
        for p in man.nodes:
            by_seg.setdefault(p.segment, []).append(p)
        chans = []
        for f, (b, u) in sorted(wiring.as_map().items()):
            def side(sid):
                ps = sorted(by_seg[sid], key=lambda x: x.position)
                return [{"node": x.node, "layers": list(x.layer_range)} for x in ps]
            chans.append({"front": side(f), "back": side(b), "task": u})
        exp = {"model_dir": "/data/qwen3-next-part", "channels": chans,
               "experts": {}}
        for u, segs in res.backs.items():
            pass
        args.save_spec.write_text(json.dumps(exp, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"  布局已存 {args.save_spec}（交给 deploy/run.py）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
