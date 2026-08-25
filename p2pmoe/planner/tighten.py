"""II.2 两轮筛：间隙检测去慢尾 + 字典序收紧 (break-and-reconstruct)。

对应文档：
* II.2.1 间隙检测去慢尾
* II.2.2 字典序收紧 tighten_lex
* III.4.1 字典序单调（最坏偏离单调不增）
* III.4.2 有限终止（由 max_steps 硬上限保证，不是收敛速度）
* III.4.3 局部性（每轮只触碰区间外段与空闲节点）

分散环境下收紧的杠杆变了：集群环境靠换更快的机器，分散环境靠消跳与挑接入
好的对端。「过快者拆掉」在此有具体含义 —— 独占一台大内存节点把整段装下
（零跳）的段，若同池其他段被迫两节点（一跳），其延迟差就是数十毫秒；拆掉它
交出大节点让慢者也能零跳，是唯一能真正拉平的动作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Mapping, Sequence

from .network import MeasurementCache
from .solver import deploy_path
from .types import Node, Objective, PlannerConfig, Segment, SegmentSpec

__all__ = [
    "GapResult",
    "detect_gap_robust",
    "blame_slow_tail",
    "band_from_median",
    "dist_to_band",
    "TightenResult",
    "tighten_lex",
]


# --------------------------------------------------------------------------- #
# II.2.1 间隙检测
# --------------------------------------------------------------------------- #
@dataclass
class GapResult:
    body: list[int]
    """主体段的索引。"""
    tail: list[int]
    """慢尾段的索引（可能为空）。"""
    gap: float
    cut_at: float | None


def detect_gap_robust(
    delays: Sequence[float], *, theta1: float = 0.35, k: float = 1.5
) -> GapResult:
    """在延迟序列里找出「与主体断开」的慢尾。

    双条件避免把连续分布误切：
      cond1  最大相邻间隙相对中位足够大
      cond2  尾部均值高出主体均值 k 个主体标准差
    两者同时成立才判定为慢尾。
    """
    n = len(delays)
    idx = sorted(range(n), key=lambda i: delays[i])
    s = [delays[i] for i in idx]
    if n < 3:
        return GapResult(body=list(range(n)), tail=[], gap=0.0, cut_at=None)

    gaps = [(s[i + 1] - s[i], i) for i in range(n - 1)]
    g_star, cut = max(gaps)
    body_i, tail_i = idx[: cut + 1], idx[cut + 1 :]
    med = median(s) or 1.0

    body_vals = [delays[i] for i in body_i]
    tail_vals = [delays[i] for i in tail_i]
    cond1 = g_star / med > theta1
    sd = pstdev(body_vals) if len(body_vals) > 1 else 0.0
    cond2 = bool(tail_vals) and mean(tail_vals) > mean(body_vals) + k * sd

    if cond1 and cond2:
        return GapResult(body=body_i, tail=tail_i, gap=g_star, cut_at=s[cut])
    return GapResult(body=list(range(n)), tail=[], gap=g_star, cut_at=None)


def blame_slow_tail(segs: Sequence[Segment], gap: GapResult, net: MeasurementCache) -> list[str]:
    """慢尾诊断。分散环境下慢尾几乎总由多余的跳或坏接入节点造成，
    故诊断优先看跳数：hops 比主体多 1 即为首要 blame（II.2.1）。"""
    if not gap.tail:
        return []
    body_hops = [segs[i].hops for i in gap.body]
    ref = min(body_hops) if body_hops else 0
    out: list[str] = []
    for i in gap.tail:
        s = segs[i]
        if s.hops > ref:
            out.append(
                f"{s.label()}: 跳数 {s.hops} 比主体 {ref} 多 {s.hops - ref} —— "
                f"抢救方案是换用内存更大的节点合并层区间以消跳"
            )
        else:
            worst = max(
                ((net.p50(a, b), f"{a}→{b}") for a, b in zip(s.nodes, s.nodes[1:])),
                default=(0.0, "—"),
            )
            out.append(f"{s.label()}: 跳数与主体相同，最慢链路 {worst[1]} = {worst[0]:.1f}ms（接入劣化）")
    return out


# --------------------------------------------------------------------------- #
# 目标区间
# --------------------------------------------------------------------------- #
def band_from_median(delays: Sequence[float], beta: float) -> tuple[float, float]:
    """目标区间取中位邻域，按 β 定宽（II.2.2）。

    β 是文档里「全局唯一需人工拍板的延迟政策」：愿意接受比中位慢多少倍。
    """
    if not delays:
        return (0.0, 0.0)
    m = median(delays)
    half = m * (beta - 1.0)
    return (m - half, m + half)


def dist_to_band(x: float, band: tuple[float, float]) -> float:
    a, b = band
    if x < a:
        return a - x
    if x > b:
        return x - b
    return 0.0


# --------------------------------------------------------------------------- #
# II.2.2 字典序收紧
# --------------------------------------------------------------------------- #
@dataclass
class TightenResult:
    segments: list[Segment]
    free_nodes: set[str]
    steps: int
    committed: int
    rolled_back: int
    history: list[tuple[float, float]] = field(default_factory=list)
    """每轮验收后的 (D_max, D_Σ)。命题 III.4.1 保证这个序列字典序单调不增。"""


def tighten_lex(
    segments: Sequence[Segment],
    spec: SegmentSpec,
    band: tuple[float, float],
    free_nodes: set[str],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
    cfg: PlannerConfig,
    *,
    good_nodes: frozenset[str] | None = None,
    rent_nodes: frozenset[str] = frozenset(),
    head_cost: Mapping[str, float] | None = None,
    tail_cost: Mapping[str, float] | None = None,
) -> TightenResult:
    """区间外两头都拆、区间内不动；整轮字典序验收，不达标则整轮回退。

    设计要点（II.2.2）：
      * 两头都破坏 —— fast 与 slow 都进重建池，否则「过快者」独占的好节点
        永远交不出来；
      * 先富后贫 —— fast 按「占大内存好节点数」降序先重建，把好节点释放到
        pool 里给后面的 slow 用；
      * 字典序验收 —— 先保最坏偏离不变差，再看总量（命题 III.4.1）。
    """
    a, b = band
    segs = list(segments)
    free = set(free_nodes)
    if good_nodes is None:
        top = max((n.usable_gb for n in nodes.values()), default=0.0)
        good_nodes = frozenset(v for v, n in nodes.items() if n.usable_gb >= top - 1e-9)

    history: list[tuple[float, float]] = []
    committed = rolled_back = 0
    prev_key = _lex_key(segs, band)
    history.append(prev_key)

    step = 0
    for step in range(1, cfg.tighten_max_steps + 1):
        fast = [i for i, s in enumerate(segs) if s.delay_ms < a - 1e-9]
        slow = [i for i, s in enumerate(segs) if s.delay_ms > b + 1e-9]
        if not fast and not slow:
            break

        pool = set(free)
        for i in fast + slow:
            pool |= set(segs[i].nodes)

        new: dict[int, Segment] = {}

        # 先富后贫：占好节点最多的过快者先拆
        for i in sorted(fast, key=lambda i: -len(set(segs[i].nodes) & good_nodes)):
            obj = Objective(
                mu_ms=cfg.mu_ms,
                jitter_w=cfg.jitter_w,
                rent_nu_ms=cfg.rent_nu_ms if rent_nodes else 0.0,
                rent_nodes=rent_nodes,
                good_gamma_ms=cfg.good_gamma_ms,
                good_nodes=good_nodes,
                target_ms=a,
                head_cost=head_cost,
                tail_cost=tail_cost,
                endpoint_w=cfg.endpoint_w if head_cost else 0.0,
            )
            got = deploy_path(
                spec, sorted(pool), nodes, net, obj,
                beam_width=cfg.beam_width, prune_topk=cfg.prune_topk,
            ) or segs[i]
            new[i] = got
            pool -= set(got.nodes)

        # 慢者纯求快：优先消跳
        for i in sorted(slow, key=lambda i: -segs[i].delay_ms):
            obj = Objective(
                mu_ms=cfg.mu_ms,
                jitter_w=cfg.jitter_w,
                rent_nu_ms=cfg.rent_nu_ms if rent_nodes else 0.0,
                rent_nodes=rent_nodes,
                target_ms=None,
                head_cost=head_cost,
                tail_cost=tail_cost,
                endpoint_w=cfg.endpoint_w if head_cost else 0.0,
            )
            got = deploy_path(
                spec, sorted(pool), nodes, net, obj,
                beam_width=cfg.beam_width, prune_topk=cfg.prune_topk,
            ) or segs[i]
            new[i] = got
            pool -= set(got.nodes)

        cand = [new.get(i, s) for i, s in enumerate(segs)]
        cand_key = _lex_key(cand, band)

        if cand_key <= prev_key:  # 字典序验收
            improvement = (prev_key[0] - cand_key[0]) + (prev_key[1] - cand_key[1])
            segs = cand
            free = pool
            committed += 1
            prev_key = cand_key
            history.append(cand_key)
            if improvement < cfg.tighten_eps_ms:
                break
        else:
            rolled_back += 1  # 整轮回退制
            break

    return TightenResult(
        segments=segs,
        free_nodes=free,
        steps=step,
        committed=committed,
        rolled_back=rolled_back,
        history=history,
    )


def _lex_key(segs: Sequence[Segment], band: tuple[float, float]) -> tuple[float, float]:
    """(D_max, D_Σ) —— 接受判据的字典序键。"""
    d = [dist_to_band(s.delay_ms, band) for s in segs]
    return (max(d) if d else 0.0, sum(d))
