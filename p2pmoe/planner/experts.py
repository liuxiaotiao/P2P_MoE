"""专家身份层：逐层驻留专家集、并集、可检性与池合并信号。

在这一层之前，规划器只知道 n_{u,l}（**基数**）—— 那足够算内存，但不足以做别的
任何事。以下四件事全都需要专家的**身份**：

* 前段并集 mem_F(l) = base + |∪_u S_{u,l}| × expert_gb
  —— 基数相加会高估：三个 task 各 8/6/7 个专家，并集可能是 21 也可能是 9。
* 命题 III.7.3 的可检性下界 q(u,û)
  —— 定义就是「u 的激活质量落在 û 驻留集外的比例」，没有集合就无从谈起。
* 推论 III.7.4 的池合并信号
  —— 判据是两个 task 驻留集高度重叠。
* 在线的 miss 检出与 drop-expert 近似（II.5）
  —— 运行时要知道「这个 token 路由到的专家在不在本地」。

对应文档：
* I.1.1  各 task 逐层驻留专家集由回放语料按覆盖率阈值统计得出，规模 n_{u,l} 异构
* II.5   基线 miss 率 = 1 − 覆盖率
* III.7.3 可检性下界（离线可算）
* III.7.4 低可检性 = 池合并信号
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ActivationProfile",
    "ExpertPlacement",
    "build_placement",
    "union_placement",
    "detectability",
    "detectability_matrix",
    "expected_detection_tokens",
    "merge_candidates",
]


# --------------------------------------------------------------------------- #
# 回放语料画像
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActivationProfile:
    """一个 task 的逐层专家激活质量分布（由回放语料统计）。

    mass[l-1][e] = 在第 l 层，路由质量落到专家 e 上的比例；每层归一化到 1。
    「质量」而非「次数」是有意的：top-k 路由带门控权重，III.7.3 的 q 定义用的
    就是质量占比。
    """

    task: str
    n_layers: int
    n_experts: int
    mass: tuple[tuple[float, ...], ...]

    def at(self, layer: int) -> tuple[float, ...]:
        return self.mass[layer - 1]

    def ranked(self, layer: int) -> list[int]:
        m = self.at(layer)
        return sorted(range(self.n_experts), key=lambda e: (-m[e], e))

    def mass_outside(self, layer: int, resident: Iterable[int]) -> float:
        """本层激活质量落在 resident 之外的比例。"""
        keep = set(resident)
        m = self.at(layer)
        return max(0.0, 1.0 - sum(m[e] for e in keep if e < self.n_experts))


# --------------------------------------------------------------------------- #
# 驻留集
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExpertPlacement:
    """一个 task（或前段并集）的逐层驻留专家集 —— 身份，不只是基数。"""

    name: str
    n_layers: int
    sets: tuple[frozenset[int], ...]
    achieved_coverage: tuple[float, ...]
    """逐层实际覆盖率。基线 miss 率 = 1 − 该值（II.5）。"""
    coverage_target: float | None = None

    def at(self, layer: int) -> frozenset[int]:
        return self.sets[layer - 1]

    def size_at(self, layer: int) -> int:
        return len(self.sets[layer - 1])

    def sizes(self, lo: int = 1, hi: int | None = None) -> list[int]:
        hi = self.n_layers if hi is None else hi
        return [self.size_at(l) for l in range(lo, hi + 1)]

    def baseline_miss(self, lo: int, hi: int) -> float:
        """层区间上的基线 miss 率 —— 在线告警线就设在它之上（II.5 通道二）。"""
        vals = [1.0 - self.achieved_coverage[l - 1] for l in range(lo, hi + 1)]
        return sum(vals) / len(vals) if vals else 0.0

    def as_experts_per_layer(self) -> list[int]:
        """降级成 TaskProfile 需要的基数序列（索引 0 对应 layer 1）。"""
        return [len(s) for s in self.sets]


def build_placement(
    profile: ActivationProfile, coverage: float, *, name: str | None = None
) -> ExpertPlacement:
    """按覆盖率阈值取驻留集（I.1.1）。

    逐层按激活质量降序累加，直到累计 ≥ coverage 为止。层与层之间独立，
    所以 n_{u,l} 天然异构 —— 这正是文档说的「规模 n_{u,l} 异构」。
    """
    sets: list[frozenset[int]] = []
    got: list[float] = []
    for l in range(1, profile.n_layers + 1):
        m = profile.at(l)
        acc = 0.0
        chosen: list[int] = []
        for e in profile.ranked(l):
            chosen.append(e)
            acc += m[e]
            if acc >= coverage - 1e-12:
                break
        sets.append(frozenset(chosen))
        got.append(acc)
    return ExpertPlacement(
        name=name or profile.task,
        n_layers=profile.n_layers,
        sets=tuple(sets),
        achieved_coverage=tuple(got),
        coverage_target=coverage,
    )


def full_placement(
    n_layers: int, n_experts: int, *, name: str = "front-full"
) -> ExpertPlacement:
    """前段逐层驻留**全部**专家。

    这不是文档的主线 —— I.1.1 说的是并集 ∪_u S_{u,l}，比全集小。全集是它的
    一个**保守上界**，用在静态简化模式里，理由是工程性的而非理论性的：

    * 并集要先有各 task 的真实激活画像才算得出来；全集不需要画像，
      部署当天就能装；
    * 前段是 task 无关的（I.1.1），将来加一个新 task 时并集会变、前段要重装，
      全集不会 —— 前段一次装好，后段随 task 增删；
    * 覆盖率恒为 1，通道二在前段侧的基线 miss 归零，少一个要标定的量。

    **代价是内存**：Qwen3-30B-A3B 上前段每层从「并集约 40%」涨到 100%，
    L₀ 处的前段总量按比例上去，直接压缩可建的通道数。所以它只适合
    「大内存节点富余、task 数多到并集本来就接近全集」的池子 —— 对 Qwen3-30B-A3B
    的 16/24GB 混合池，跑 `examples/model_fit.py` 看 L₀ 与通道数的实际代价。
    """
    full = frozenset(range(n_experts))
    return ExpertPlacement(
        name=name,
        n_layers=n_layers,
        sets=tuple(full for _ in range(n_layers)),
        achieved_coverage=tuple(1.0 for _ in range(n_layers)),
        coverage_target=1.0,
    )


def union_placement(
    placements: Sequence[ExpertPlacement], *, name: str = "front-union"
) -> ExpertPlacement:
    """前段的逐层并集 ∪_u S_{u,l}（I.1.1：前段每层驻留全 task 专家并集）。

    并集支配性（III.5.4 用到）：mem_F(l) ≥ max_u mem_u(l) 逐层成立，因为
    ∪_u S_{u,l} ⊇ S_{u,l}。这是构造性的，不需要额外假设。

    注意并集的**覆盖率不是各 task 覆盖率的并** —— 对任一 task u，并集至少
    覆盖 S_{u,l} 覆盖的那部分，故这里逐层取 max，是对每个 task 都成立的下界。
    """
    if not placements:
        raise ValueError("并集至少需要一个 placement")
    n = placements[0].n_layers
    if any(p.n_layers != n for p in placements):
        raise ValueError("各 placement 的层数不一致")
    sets = tuple(
        frozenset().union(*(p.at(l) for p in placements)) for l in range(1, n + 1)
    )
    cov = tuple(
        max(p.achieved_coverage[l - 1] for p in placements) for l in range(1, n + 1)
    )
    return ExpertPlacement(name=name, n_layers=n, sets=sets, achieved_coverage=cov)


# --------------------------------------------------------------------------- #
# III.7.3 可检性 / III.7.4 池合并
# --------------------------------------------------------------------------- #
def detectability(
    profile_true: ActivationProfile,
    placement_wrong: ExpertPlacement,
    lo: int,
    hi: int,
) -> float:
    """q(u, û) —— 真实 task u 被误绑到 û 时，每个 decode token 在 l ∈ [lo,hi]
    触发 expert miss 的概率下界（命题 III.7.3）。

    定义即「u 的激活质量落在 û 驻留集外的比例」，逐层取均值。

    [口径说明] 文档用的是**质量占比**。若模型是 top-k 路由，「该 token 至少有
    一个被选专家缺失」的概率会高于质量占比（k 次独立机会），所以质量占比是
    保守的下界 —— 与命题里写的「≥」一致。
    """
    vals = [
        profile_true.mass_outside(l, placement_wrong.at(l)) for l in range(lo, hi + 1)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def detectability_matrix(
    profiles: Mapping[str, ActivationProfile],
    placements: Mapping[str, ExpertPlacement],
    lo: int,
    hi: int,
) -> dict[tuple[str, str], float]:
    """全部有序对 (真实 u, 误绑 û) 的 q 矩阵。对角线是基线 miss 率。"""
    out: dict[tuple[str, str], float] = {}
    for u, prof in profiles.items():
        for v, plc in placements.items():
            out[(u, v)] = detectability(prof, plc, lo, hi)
    return out


def expected_detection_tokens(q: float) -> float:
    """期望检出延迟 = O(1/q) 个 token（命题 III.7.3）。

    q = 0 表示该误绑在结构上不可检 —— 两池驻留集完全覆盖对方的激活质量，
    只能靠通道一（前段的统计后验）兜底。
    """
    return float("inf") if q <= 1e-12 else 1.0 / q


@dataclass(frozen=True)
class MergeSignal:
    a: str
    b: str
    q_ab: float
    q_ba: float
    jaccard: float
    saved_gb_per_layer: float

    @property
    def worst_q(self) -> float:
        return max(self.q_ab, self.q_ba)


def merge_candidates(
    profiles: Mapping[str, ActivationProfile],
    placements: Mapping[str, ExpertPlacement],
    lo: int,
    hi: int,
    *,
    q_threshold: float = 0.05,
    expert_gb: float = 0.0,
) -> list[MergeSignal]:
    """推论 III.7.4：q(u,û) 与 q(û,u) 均小 ⟺ 两 task 驻留集高度重叠
    ⟺ (a) 误判几乎无害，(b) 合并两池可省内存并消除该混淆对。

    分散环境下池合并的价值更大：它减少池数、提高每池条数、从而抬升最坏公平比。
    返回按「最坏 q」升序排列的候选对（越靠前越该合并）。
    """
    names = sorted(placements)
    out: list[MergeSignal] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a not in profiles or b not in profiles:
                continue
            q_ab = detectability(profiles[a], placements[b], lo, hi)
            q_ba = detectability(profiles[b], placements[a], lo, hi)
            if max(q_ab, q_ba) > q_threshold:
                continue
            inter = union = 0
            for l in range(lo, hi + 1):
                sa, sb = placements[a].at(l), placements[b].at(l)
                inter += len(sa & sb)
                union += len(sa | sb)
            n_layers = hi - lo + 1
            # 合并后每层驻留 |S_a ∪ S_b|，原本两池各驻留一份 —— 省下的是
            # 「两份之和 − 一份并集」，按层平均
            two = sum(
                len(placements[a].at(l)) + len(placements[b].at(l))
                for l in range(lo, hi + 1)
            )
            saved = (two - union) / n_layers * expert_gb if n_layers else 0.0
            out.append(
                MergeSignal(
                    a=a,
                    b=b,
                    q_ab=q_ab,
                    q_ba=q_ba,
                    jaccard=inter / union if union else 0.0,
                    saved_gb_per_layer=saved,
                )
            )
    out.sort(key=lambda s: s.worst_q)
    return out
