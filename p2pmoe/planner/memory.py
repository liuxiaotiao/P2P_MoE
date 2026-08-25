"""内存模型、跳数整数下界与 L₀ 选取。

对应文档：
* I.2.2  硬约束（含 KV 与单节点成段）
* III.5.1 含跳数地板的松下界
* III.5.2 跳数下界（整数）—— 分散环境下延迟的主项
* III.5.3 跳数平衡是分散环境的 L₀ 准则
* III.5.4 含 KV 的可行性必要条件
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Mapping, Sequence

from .types import ModelSpec, Node, SegmentSpec, TaskProfile

__all__ = [
    "hops_min",
    "front_gb_per_layer",
    "back_gb_per_layer",
    "make_front_spec",
    "make_back_spec",
    "feasibility_necessary",
    "L0Candidate",
    "choose_l0",
]


# --------------------------------------------------------------------------- #
# III.5.2 跳数整数下界
# --------------------------------------------------------------------------- #
def hops_min(total_gb: float, max_usable_gb: float) -> int:
    """hops_min(S) = ⌈ Σ_{l∈S} m(l) / max_v (M_v − 预留) ⌉ − 1

    最小节点数由「总驻留量 ÷ 最大可用单机内存」的上整给出；链上 n 节点有
    n−1 跳。分散环境下 h 为数十毫秒，故该整数下界直接决定延迟的主项。
    """
    if max_usable_gb <= 0:
        raise ValueError("max_usable_gb must be positive")
    return max(0, ceil(total_gb / max_usable_gb - 1e-9) - 1)


def max_usable(nodes: Sequence[Node]) -> float:
    return max(n.usable_gb for n in nodes)


# --------------------------------------------------------------------------- #
# 逐层内存形状
# --------------------------------------------------------------------------- #
def front_gb_per_layer(model: ModelSpec, union_experts) -> Callable[[int], float]:
    """mem_F(l)：前段驻留全 task 专家并集（I.1.1）。

    union_experts 可以是：
      * int              —— 逐层同质的并集规模（粗估口径，无专家身份）
      * ExpertPlacement  —— 逐层并集（有身份）。真实语料下并集规模逐层不同，
                            用一个常数会同时产生高估与低估。
    """
    if isinstance(union_experts, int):
        gb = model.weight_gb_per_layer(union_experts)
        return lambda _l: gb
    return lambda l: model.weight_gb_per_layer(union_experts.size_at(l))


def back_gb_per_layer(model: ModelSpec, task: TaskProfile) -> Callable[[int], float]:
    """mem_u(l)：后段驻留 task 专用集，规模 n_{u,l} 异构。"""
    return lambda l: model.weight_gb_per_layer(task.n_experts_at(l))


def make_front_spec(
    model: ModelSpec,
    union_experts,
    l0: int,
    *,
    tail_allowed: frozenset[str] | None = None,
) -> SegmentSpec:
    return SegmentSpec(
        kind="front",
        task=None,
        layer_lo=1,
        layer_hi=l0,
        gb_per_layer=front_gb_per_layer(model, union_experts),
        kv_gb_per_layer=model.kv_gb_per_layer,
        tail_allowed=tail_allowed,
    )


def make_back_spec(model: ModelSpec, task: TaskProfile, l0: int) -> SegmentSpec:
    return SegmentSpec(
        kind="back",
        task=task.name,
        layer_lo=l0 + 1,
        layer_hi=model.n_layers,
        gb_per_layer=back_gb_per_layer(model, task),
        kv_gb_per_layer=model.kv_gb_per_layer,
        tail_allowed=None,  # 分散环境主线：后段端点无约束（I.2.2）
    )


# --------------------------------------------------------------------------- #
# III.5.4 可行性必要条件
# --------------------------------------------------------------------------- #
@dataclass
class FeasibilityReport:
    supply_gb: float
    demand_gb: float
    ok: bool
    detail: str


def feasibility_necessary(
    nodes: Sequence[Node],
    front_spec: SegmentSpec,
    back_specs: Mapping[str, SegmentSpec],
    n_front: int,
    n_back: Mapping[str, int],
) -> FeasibilityReport:
    """Σ_v (M_v − 预留) ≥ N_F·[前段驻留 + kv_F] + Σ_u N_B(u)·[后段驻留 + kv_B]

    必要非充分 —— 它忽略排他与连续切分造成的碎片，也忽略网络约束。
    """
    supply = sum(n.usable_gb for n in nodes)
    demand = n_front * front_spec.total_gb()
    for u, spec in back_specs.items():
        demand += n_back.get(u, 0) * spec.total_gb()
    ok = supply >= demand
    return FeasibilityReport(
        supply_gb=supply,
        demand_gb=demand,
        ok=ok,
        detail=f"供给 {supply:.1f}GB {'≥' if ok else '<'} 需求 {demand:.1f}GB",
    )


# --------------------------------------------------------------------------- #
# III.5.3 L₀ 选取：跳数平衡
# --------------------------------------------------------------------------- #
@dataclass
class L0Candidate:
    l0: int
    p_accuracy: float
    front_gb: float
    back_gb: dict[str, float]
    hops_front: int
    hops_back: dict[str, int]
    total_hops_weighted: float
    """按 λ_u 加权的总跳数 = hops_F + Σ_u λ_u·hops_B(u) + 2

    诚实标注：这是 III.5.2 的**乐观**下界 —— 它用 max_v(M_v − 预留) 作分母，
    等于假设最大内存档的供给无限。当最大档稀缺时该下界不可达，见 n_channels。
    """
    front_single_node_count: int
    """能单节点承载整个前段的节点数。"""
    n_channels: int
    """按节点位做精确配平后真正能建出的通道数（II.7.1）。

    这一项是必要的：单看 III.5.2 的跳数下界会得出「L₀ 越大越好」的错误结论 ——
    算例 C 的池子里 L₀=7 时 Y 池后段恰好缩进单张 A40（跳数从 1 降到 0，加权
    总跳数 3.00→2.70 看起来更优），但同时前段也被迫只能用 A40，于是 4 张 A40
    要同时供前段和 X/Y 后段，通道数反而塌掉。跳数下界不含供给约束，容量估算
    含 —— 两者必须一起看。
    """
    feasible: bool

    def key(self) -> tuple:
        # 字典序：
        #   1. 最大化真正能建出的通道数（供给约束，压倒一切）
        #   2. 最小化加权总跳数（每多一跳即数十毫秒/token，III.5.3）
        #   3. 最大化识别准确率（前两项都不变时，多一层是免费的，C.1 的取法）
        return (-self.n_channels, self.total_hops_weighted, -self.p_accuracy, self.l0)


def choose_l0(
    model: ModelSpec,
    tasks: Sequence[TaskProfile],
    union_experts: int,
    nodes: Sequence[Node],
    p_curve: Mapping[int, float],
    *,
    p_min: float = 0.85,
    l0_range: Sequence[int] | None = None,
) -> tuple[L0Candidate, list[L0Candidate]]:
    """选 L₀，返回 (推荐, 全部候选表)。

    规则（字典序，见 L0Candidate.key）：在 p(L₀) ≥ p_min 的候选中
      1. 最大化真正能建出的通道数
      2. 最小化按 λ 加权的总跳数（III.5.3）
      3. 最大化 p(L₀)

    与文档的两点分歧，均已在候选表里显式暴露：

    * 文档 III.5.3 正文写「取最小 L₀」，算例 C.1 实际取的是跳数不变前提下的
      更大 L₀（"这一层是免费的"）。本实现采用后者。
    * 文档只用跳数下界做 L₀ 准则。但跳数下界（III.5.2）以 max_v(M_v−预留) 为
      分母，隐含「最大内存档供给无限」；当最大档稀缺时，单看跳数会选出通道数
      更少的 L₀。本实现把容量估算并入准则，理由见 L0Candidate.n_channels。
      若要复现文档口径，按 (total_hops_weighted, -p_accuracy) 排序即可。
    """
    from .capacity import estimate_capacity_by_tier  # 局部导入避免顶层循环依赖

    mx = max_usable(nodes)
    l0s = list(l0_range) if l0_range is not None else sorted(p_curve)
    out: list[L0Candidate] = []

    for l0 in l0s:
        if not (1 <= l0 < model.n_layers):
            continue
        fspec = make_front_spec(model, union_experts, l0)
        front_gb = fspec.total_gb()
        h_f = hops_min(front_gb, mx)

        back_specs: dict[str, SegmentSpec] = {}
        back_gb: dict[str, float] = {}
        h_b: dict[str, int] = {}
        for t in tasks:
            bspec = make_back_spec(model, t, l0)
            back_specs[t.name] = bspec
            g = bspec.total_gb()
            back_gb[t.name] = g
            h_b[t.name] = hops_min(g, mx)

        weighted = h_f + sum(t.lam * h_b[t.name] for t in tasks) + 2.0
        n_single = sum(1 for n in nodes if n.usable_gb >= front_gb)

        n_ch = 0
        if n_single > 0:
            try:
                n_ch = estimate_capacity_by_tier(nodes, fspec, back_specs, tasks).n_max_slots
            except Exception:  # 形态枚举失败 ⇒ 该 L₀ 不可建
                n_ch = 0

        out.append(
            L0Candidate(
                l0=l0,
                p_accuracy=p_curve.get(l0, 0.0),
                front_gb=front_gb,
                back_gb=back_gb,
                hops_front=h_f,
                hops_back=h_b,
                total_hops_weighted=weighted,
                front_single_node_count=n_single,
                n_channels=n_ch,
                feasible=n_single > 0 and n_ch > 0,
            )
        )

    eligible = [c for c in out if c.feasible and c.p_accuracy >= p_min]
    if not eligible:
        raise ValueError(
            f"没有 L₀ 同时满足 p ≥ {p_min}、前段可单节点承载、且通道数 > 0；"
            f"候选表: {[(c.l0, c.p_accuracy, c.front_single_node_count, c.n_channels) for c in out]}"
        )
    best = min(eligible, key=lambda c: c.key())
    return best, out
