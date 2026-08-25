"""回放语料的模拟：逐 task、逐层的专家激活质量画像。

真实系统里这张表由回放语料统计得出（I.1.1）。这里生成一份具备同样定性特征的
合成画像，用来跑通「覆盖率阈值 → 驻留集 → 可检性 → 池合并」这条链路：

* 每层的激活质量高度集中在少数专家上（MoE 路由的经验事实），用几何衰减建模；
* 不同 task 的「热门专家」大部分不同，但共享一个小的公共核 —— 重叠度可调，
  因为它直接决定 q(u,û) 与是否该合并池（III.7.4）；
* 逐层重新洗牌，所以 n_{u,l} 天然逐层异构（文档说的「规模 n_{u,l} 异构」）。

衰减率由「希望 top-n 覆盖多少」反解：几何分布下 top-n 覆盖 1 − r^n，故
r = (1 − coverage)^(1/n)。这样可以直接按算例 C 的口径要求「X 约 8 个专家覆盖
97%」，而不用手调参数。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..planner.experts import ActivationProfile

__all__ = ["make_activation_profiles"]


def make_activation_profiles(
    tasks: Sequence[str],
    target_sizes: Mapping[str, int],
    *,
    n_layers: int,
    n_experts: int,
    coverage: float = 0.97,
    shared_core: int = 1,
    seed: int = 0,
    size_jitter: int = 1,
) -> dict[str, ActivationProfile]:
    """构造逐 task 的激活质量画像。

    Parameters
    ----------
    target_sizes : 每个 task 在 `coverage` 阈值下期望的驻留集规模（层平均）
    shared_core : 各 task 共享的热门专家数。0 = 完全不重叠（可检性最高、
        绝不该合并池）；大 = 高度重叠（误判几乎无害，该合并，推论 III.7.4）
    size_jitter : 逐层规模抖动幅度，用来产生 n_{u,l} 的逐层异构
    """
    rng = np.random.default_rng(seed)
    profiles: dict[str, ActivationProfile] = {}
    n_tasks = len(tasks)

    # 逐层的排布：公共核 + 各 task 私有池（互不相交）
    per_layer_rank: list[dict[str, list[int]]] = []
    for _ in range(n_layers):
        perm = rng.permutation(n_experts).tolist()
        core = perm[:shared_core]
        rest = perm[shared_core:]
        ranks: dict[str, list[int]] = {}
        for j, u in enumerate(tasks):
            private = rest[j::n_tasks]
            ranks[u] = list(core) + list(private)
        per_layer_rank.append(ranks)

    for u in tasks:
        n_target = target_sizes[u]
        mass_all: list[tuple[float, ...]] = []
        for l in range(n_layers):
            # 逐层微调目标规模 → n_{u,l} 异构
            n_l = max(1, n_target + int(rng.integers(-size_jitter, size_jitter + 1)))
            r = (1.0 - coverage) ** (1.0 / n_l)
            order = per_layer_rank[l][u]
            w = np.array([(1 - r) * r**i for i in range(len(order))], dtype=float)
            m = np.zeros(n_experts, dtype=float)
            m[np.array(order, dtype=int)] = w
            m /= m.sum()
            mass_all.append(tuple(float(x) for x in m))
        profiles[u] = ActivationProfile(
            task=u, n_layers=n_layers, n_experts=n_experts, mass=tuple(mass_all)
        )
    return profiles
