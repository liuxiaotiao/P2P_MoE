#!/usr/bin/env python3
"""量一量「后段只装 hot expert」到底损失多少。

    python examples/drop_expert_impact.py                      # 合成模型（验工具）
    python examples/drop_expert_impact.py --model-dir /data/qwen3-30b-a3b \\
        --profile prof.json --coverage 0.9 0.95 0.99

这是 TODO 里挂了很久的 P3：drop-expert 被文档标注为「运维近似，非无损」，
但**从来没量过它有多不无损**。接了真模型、有了参考实现之后，这件事可以做了。

量三件不同的事
--------------
**1. miss 率**：被路由到的专家有多少不在本地。这是通道二的观测量，也是最直接的
指标 —— 但它不等于质量损失，因为丢掉的可能是门控权重很小的那个。

**2. 丢掉的门控质量（miss_mass）**：比 miss 率诚实。top-k 里排第 8 的那个专家
权重可能只有 0.02，丢了几乎没影响；排第 1 的丢了就伤筋动骨。

**3. 输出偏差**：与全装的输出比。这才是真正要的答案，前两个都是代理指标。
分 prefill 与 decode 两种口径 —— 后者更重要，理由见下。

为什么 decode 比 prefill 危险
-----------------------------
prefill 是一次前向：误差进去、出来、结束。decode 不是 —— 第 t 步的输入是第 t-1
步被扰动过的输出，**误差会沿着生成步累积**，而且路由本身也会跟着漂：被扰动的
hidden state 可能选出不同的专家，于是走上一条越来越偏的轨迹。

所以「单次前向的偏差很小」推不出「生成 64 个 token 没问题」。这里显式测生成
在第几步开始分叉。

**每条 path 一个 request 不影响这里的任何结论** —— 排他独占（I.2.4）意味着
一条通道同时只服务一条请求，drop-expert 的影响是逐请求独立的，不存在批内互相
干扰。并发度是另一回事（= 通道数），与本文件量的东西无关。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def rule(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "─" * 78)


def build(cfg: dict, model_dir: Path, layer_experts: dict, dtype,
          policy: str = "drop"):
    from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel
    from p2pmoe.runtime.weights import (
        KeyPlan, SelectiveLoader, WeightIndex, qwen_moe_keys,
    )

    keys = qwen_moe_keys(KeyPlan(layer_experts=layer_experts, with_embed=True,
                                 with_lm_head=True))
    tensors, _ = SelectiveLoader(WeightIndex(model_dir)).load(keys, dtype=dtype)
    return TorchSegmentModel(TorchModelConfig.from_hf(cfg), layer_experts, tensors,
                             miss_policy=policy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None,
                    help="真实 checkpoint。不给就造一个合成的（只验工具，数字无意义）")
    ap.add_argument("--profile", default=None,
                    help="激活画像。不给就用「同一批输入自己的路由」当画像 —— "
                         "那是乐观上界，见输出里的提醒")
    ap.add_argument("--coverage", type=float, nargs="+",
                    default=[0.8, 0.9, 0.95, 0.99])
    ap.add_argument("--l0", type=int, default=None,
                    help="前后段切点。前段全装，只裁后段（与部署一致）")
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=24, help="生成多少步")
    ap.add_argument("--policy", nargs="+",
                    default=["drop", "drop_noscale", "local_topk"],
                    help="缺专家时的补救策略，可给多个横向对比")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    # ---------------- 模型 ---------------- #
    if args.model_dir:
        d = Path(args.model_dir)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        synthetic = False
    else:
        import tempfile

        from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint

        cfg = dict(TINY_QWEN3_MOE, num_hidden_layers=6)
        d = Path(write_fake_checkpoint(tempfile.mkdtemp(), cfg, seed=args.seed))
        synthetic = True

    L, E, K = (cfg["num_hidden_layers"], cfg["num_experts"],
               cfg["num_experts_per_tok"])
    l0 = args.l0 if args.l0 is not None else max(1, L // 4)
    dtype = torch.float32
    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, cfg["vocab_size"], size=args.prompt_len).tolist()

    rule("配置")
    print(f"  {d}{'   ⚠ 合成权重' if synthetic else ''}")
    print(f"  {L} 层 × {E} 专家 top-{K}；前段 1..{l0}（全装），后段 {l0+1}..{L}（要裁的）")

    # ---------------- 基准：全装 ---------------- #
    full_le = {l: list(range(E)) for l in range(1, L + 1)}
    full = build(cfg, d, full_le, dtype)
    with torch.no_grad():
        h, _ = full.forward("ref", full.embed_tokens(ids))
        base_logits = full.logits(h)
        # 记下每层实际被路由到的专家与质量 —— 没有画像时拿它当画像
        full.enable_profiling()
        full.drop_kv("p")
        full.forward("p", full.embed_tokens(ids))
        observed = {l: full.profiler.mass[l] / full.profiler.mass[l].sum()
                    for l in full.profiler.mass}
        # 全装的贪心生成，当作参考轨迹
        full.drop_kv("g")
        hh, _ = full.forward("g", full.embed_tokens(ids))
        tok = int(full.logits(hh[-1:])[0].argmax())
        base_seq = [tok]
        for _ in range(args.tokens - 1):
            hh, _ = full.forward("g", full.embed_tokens([tok]))
            tok = int(full.logits(hh[-1:])[0].argmax())
            base_seq.append(tok)

    if args.profile:
        from p2pmoe.runtime.profile import load_profile, placement_from_profile

        raw = load_profile(args.profile)
        source = f"画像 {args.profile}"
    else:
        raw = {"n_experts": E, "n_layers": L, "tasks": {"self": {
            "n_tokens": len(ids),
            "layers": {str(l): [float(x) for x in m] for l, m in observed.items()}}}}
        source = "**这批输入自己的路由**（乐观上界）"

    rule(f"逐覆盖率的损失（驻留集来自 {source}）")
    head = f"  {'覆盖率':>6} {'后段每层':>9} {'miss率':>8} {'丢门控':>8}"
    for pol in args.policy:
        head += f" {('KL·' + pol):>18}"
    print(head + f" {'最好的':>13}")

    from p2pmoe.runtime.profile import placement_from_profile

    for cov in args.coverage:
        sets = placement_from_profile(raw, next(iter(raw["tasks"])), coverage=cov,
                                      n_experts=E, min_experts=K,
                                      layers=list(range(l0 + 1, L + 1)))
        le = {l: (list(range(E)) if l <= l0 else list(sets[l]))
              for l in range(1, L + 1)}
        kls: dict[str, float] = {}
        st = None
        for pol in args.policy:
            thin = build(cfg, d, le, dtype, pol)
            with torch.no_grad():
                h, st = thin.forward(f"t{pol}", thin.embed_tokens(ids))
                lg = thin.logits(h)
                p_full = torch.softmax(base_logits.float(), dim=-1)
                p_thin = torch.log_softmax(lg.float(), dim=-1)
                kls[pol] = float((p_full * (torch.log(p_full.clamp_min(1e-12))
                                            - p_thin)).sum(-1).mean())
        back_sizes = [len(le[l]) for l in range(l0 + 1, L + 1)]
        best = min(kls, key=lambda k: kls[k])
        row = (f"  {cov:>6.2f} {np.mean(back_sizes):>6.1f}/{E:<2} "
               f"{st.miss_rate:>7.1%} {st.miss_mass/max(st.n_token_layer,1):>8.3f}")
        for pol in args.policy:
            row += f" {kls[pol]:>18.4f}"
        print(row + f" {best:>13}")

    rule("生成轨迹（只对第一个策略跑，看误差沿 decode 步怎么累积）")
    for cov in args.coverage:
        sets = placement_from_profile(raw, next(iter(raw["tasks"])), coverage=cov,
                                      n_experts=E, min_experts=K,
                                      layers=list(range(l0 + 1, L + 1)))
        le = {l: (list(range(E)) if l <= l0 else list(sets[l]))
              for l in range(1, L + 1)}
        thin = build(cfg, d, le, dtype, args.policy[0])
        with torch.no_grad():
            h, st = thin.forward("t", thin.embed_tokens(ids))
            lg = thin.logits(h)
            # KL(全装 ‖ 裁过的)：连续指标，不像 argmax 那样一票定生死。
            # argmax 分叉在权重随机时几乎必然发生（top-1 与 top-2 的间距接近 0），
            # 那说明的是「决策很接近」，不是「输出很不同」。KL 分得清这两件事。
            top1 = float((lg.argmax(-1) == base_logits.argmax(-1)).float().mean())

            thin.drop_kv("g")
            hh, _ = thin.forward("g", thin.embed_tokens(ids))
            t2 = int(thin.logits(hh[-1:])[0].argmax())
            seq = [t2]
            for _ in range(args.tokens - 1):
                hh, _ = thin.forward("g", thin.embed_tokens([t2]))
                t2 = int(thin.logits(hh[-1:])[0].argmax())
                seq.append(t2)
        fork = next((i for i, (a, b) in enumerate(zip(seq, base_seq)) if a != b), None)
        print(f"  覆盖率 {cov:.2f}：top1 一致 {top1:.0%}，生成"
              + (f"在第 {fork} 步分叉" if fork is not None else "未分叉"))

    rule("怎么读这张表")
    print("  · **miss 率**是通道二的观测量，但它不等于质量损失 —— 丢掉的可能是")
    print("    门控权重 0.02 的那个。「丢门控」才是诚实的代理指标。")
    print("  · **KL** 是输出分布的距离，连续、稳健。argmax 一致率与「分叉于第几步」")
    print("    很脆：top-1 与 top-2 间距小时，再小的扰动也会翻，那说明「决策很接近」")
    print("    而不是「输出很不同」。看 KL 判断质量，看分叉判断可复现性。")
    print("  · **生成分叉于**：prefill 偏差小推不出生成没问题 —— decode 的误差会")
    print("    沿步累积，而且被扰动的 hidden state 会选出不同的专家，越走越偏。")
    if not args.profile:
        print("  · ⚠ 驻留集是拿**这批输入自己的路由**算的 —— 相当于「考前看过答案」。")
        print("    真实情况要用别的语料统计出画像，再在**没见过的**输入上评估。")
        print("    给 --profile 就是那个口径，数字会明显更差。")
    if synthetic:
        print("  · ⚠ 合成权重的路由接近均匀，hot expert 这个概念在这里是退化的：")
        print("    覆盖率 0.95 要装掉几乎所有专家。真模型上分布集中得多，")
        print("    同样覆盖率只要 10–20% 的专家。**这张表验的是工具，不是结论。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
