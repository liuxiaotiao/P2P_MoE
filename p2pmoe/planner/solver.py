"""II.1 核心求解器 deploy_path —— 跳数优先的 beam search。

前后段共用，差异全部收进 SegmentSpec。与集群版的三点差异（II.1）：

1. 单节点/少跳解显式优先 —— 每省一跳即省数十毫秒，远超任何计算优化。
   实现上体现为：目标函数里 hop_ms 与 compute_ms 同量纲相加，而分散环境下
   前者是后者的十倍以上；且「把剩余层全部放在当前节点」总是被显式枚举。
2. 抖动进硬屏蔽 —— p95 − p50 > J_cap 的链路不参与（MeasurementCache.blocked）。
3. 稳定性 μ 项升为一等 —— churn 频繁，可用率低的节点即使快也不值。

与文档伪码的一处结构差异（等价改写）：文档写 `p.finish_here()` 与
`p.extend(v′, seg)` 两类动作，本实现把「本节点承载多少层」在节点入链时就
决定，于是 finish_here 只是 seg_len == 剩余层数 的那个候选。两者枚举的解
空间相同，但状态定义更干净（层区间与节点一一对应）。

复杂度：单条段的最优解本可由 DP 多项式求得（命题 III.1.2，O(L²|V|²)），
但那需要「已占节点固定」且不含稳定性/租金这类非可加项。beam 是对含全部
评分项的版本的启发式，宽度 W 可调。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .network import MeasurementCache
from .types import Node, Objective, Segment, SegmentSpec

__all__ = ["deploy_path", "prune_neighbors", "segment_from_nodes"]


@dataclass(frozen=True)
class _Partial:
    nodes: tuple[str, ...]
    splits: tuple[tuple[int, int], ...]
    placed: int
    compute_ms: float
    hop_ms: float
    jitter_ms: float

    @property
    def last(self) -> str:
        return self.nodes[-1]


def prune_neighbors(
    v: str,
    avail: Sequence[str],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
    top_k: int,
    peers_for_offset: Sequence[str],
) -> list[str]:
    """预剪枝 N*(v)：按 (M_v, 接入偏移 δ̂_v, 可用率 r_v) 取前 K（II.1）。

    先剔除被抖动闸屏蔽的对端，再按「大内存优先、接入好优先、稳定优先」排序。
    大内存排在最前是因为分散环境下多装一层就可能少一跳。
    """
    cands = [u for u in avail if u != v and not net.blocked(v, u)]
    cands.sort(
        key=lambda u: (
            -nodes[u].usable_gb,
            net.p50(v, u),
            -nodes[u].avail,
        )
    )
    return cands[:top_k]


def _mem_ok(spec: SegmentSpec, lo: int, hi: int, node: Node) -> bool:
    """I.2.2 内存（含 KV）：Σ mem(l) + kv(S, ctx_max) ≤ M_v − 预留。"""
    return spec.resident_gb(lo, hi) <= node.usable_gb + 1e-9


def segment_from_nodes(
    spec: SegmentSpec,
    node_ids: Sequence[str],
    splits: Sequence[tuple[int, int]],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
) -> Segment:
    """由给定的节点链与层切分重建一条 Segment（用于测试与手工方案导入）。"""
    compute = 0.0
    for v, (lo, hi) in zip(node_ids, splits):
        compute += (hi - lo + 1) * nodes[v].ms_per_layer * spec.ms_per_layer_scale
    hop = sum(net.p50(a, b) for a, b in zip(node_ids, node_ids[1:]))
    jit = sum(net.jitter(a, b) for a, b in zip(node_ids, node_ids[1:]))
    return Segment(
        kind=spec.kind,
        task=spec.task,
        nodes=tuple(node_ids),
        splits=tuple(splits),
        compute_ms=compute,
        hop_ms=hop,
        jitter_ms=jit,
    )


def deploy_path(
    spec: SegmentSpec,
    avail: Sequence[str],
    nodes: Mapping[str, Node],
    net: MeasurementCache,
    objective: Objective,
    *,
    beam_width: int = 12,
    prune_topk: int = 10,
) -> Segment | None:
    """在可用节点集上放置 spec 描述的层区间，返回最优段；不可行返回 None。"""
    lo0, hi0 = spec.layer_lo, spec.layer_hi
    n_layers = spec.n_layers
    if n_layers <= 0 or not avail:
        return None

    avail = list(avail)
    peers = avail  # δ̂ 的对端取全体可用节点

    # --- 完成下界 -------------------------------------------------------- #
    # 部分解不能按「已累计代价」直接比较：那样「少放几层」永远分数更低，beam 会
    # 被一堆刚起步的前缀塞满而永远收敛不到完整解。加一个可容许的完成下界，
    # 让不同进度的前缀在同一尺度上比较（A* 式）。
    #
    # 剩余层必须落在**新**节点上（本实现里节点入链时其层数即已确定），故
    #   剩余计算 ≥ 剩余层数 × min ms_per_layer
    #   剩余跳数 ≥ ⌈剩余驻留量 / max(M_v − 预留)⌉      （命题 III.5.2 的同构式）
    # 两项都取可用节点上的极值，保证是下界。
    min_ms = min(nodes[v].ms_per_layer for v in avail) * spec.ms_per_layer_scale
    max_cap = max(nodes[v].usable_gb for v in avail)
    min_hop = net.min_hop_p50(avail)

    def completion_lb(placed: int) -> tuple[float, float]:
        rem = n_layers - placed
        if rem <= 0:
            return (0.0, 0.0)
        rem_gb = spec.resident_gb(lo0 + placed, hi0)
        from math import ceil as _ceil

        n_more = max(1, _ceil(rem_gb / max_cap - 1e-9))
        return (rem * min_ms, n_more * min_hop)

    def score_partial(p: _Partial) -> float:
        lb_c, lb_h = completion_lb(p.placed)
        return objective.score(
            p.compute_ms + lb_c, p.hop_ms + lb_h, p.jitter_ms, p.nodes, nodes, partial=True
        )

    def score_final(p: _Partial) -> float:
        return objective.score(
            p.compute_ms, p.hop_ms, p.jitter_ms, p.nodes, nodes, partial=False
        )

    # --- 起点：每个候选起点 × 每种首段长度 ------------------------------- #
    starts = [v for v in avail if spec.head_ok(v)]
    starts.sort(key=lambda v: (-nodes[v].usable_gb, -nodes[v].avail))
    starts = starts[: max(prune_topk, beam_width)]

    beam: list[_Partial] = []
    for v in starts:
        node = nodes[v]
        for seg_len in range(1, n_layers + 1):
            lo, hi = lo0, lo0 + seg_len - 1
            if not _mem_ok(spec, lo, hi, node):
                break  # 内存单调：更长只会更不可行
            if seg_len == n_layers and not spec.tail_ok(v):
                continue
            beam.append(
                _Partial(
                    nodes=(v,),
                    splits=((lo, hi),),
                    placed=seg_len,
                    compute_ms=seg_len * node.ms_per_layer * spec.ms_per_layer_scale,
                    hop_ms=0.0,
                    jitter_ms=0.0,
                )
            )
    if not beam:
        return None
    beam.sort(key=score_partial)
    beam = beam[:beam_width]

    finished: list[_Partial] = [p for p in beam if p.placed == n_layers]
    frontier = [p for p in beam if p.placed < n_layers]

    # --- 逐轮扩展 --------------------------------------------------------- #
    while frontier:
        cand: list[_Partial] = []
        for p in frontier:
            remaining = n_layers - p.placed
            nxt_lo = lo0 + p.placed
            used = set(p.nodes)
            neigh = prune_neighbors(p.last, avail, nodes, net, prune_topk, peers)
            for v2 in neigh:
                if v2 in used:
                    continue
                node2 = nodes[v2]
                hop = net.p50(p.last, v2)
                jit = net.jitter(p.last, v2)
                for seg_len in range(1, remaining + 1):
                    lo, hi = nxt_lo, nxt_lo + seg_len - 1
                    if not _mem_ok(spec, lo, hi, node2):
                        break
                    is_last = seg_len == remaining
                    if is_last and not spec.tail_ok(v2):
                        continue
                    cand.append(
                        _Partial(
                            nodes=p.nodes + (v2,),
                            splits=p.splits + ((lo, hi),),
                            placed=p.placed + seg_len,
                            compute_ms=p.compute_ms
                            + seg_len * node2.ms_per_layer * spec.ms_per_layer_scale,
                            hop_ms=p.hop_ms + hop,
                            jitter_ms=p.jitter_ms + jit,
                        )
                    )
        if not cand:
            break
        cand.sort(key=score_partial)
        cand = cand[:beam_width]
        finished.extend(p for p in cand if p.placed == n_layers)
        frontier = [p for p in cand if p.placed < n_layers]

    if not finished:
        return None
    best = min(finished, key=score_final)
    return Segment(
        kind=spec.kind,
        task=spec.task,
        nodes=best.nodes,
        splits=best.splits,
        compute_ms=best.compute_ms,
        hop_ms=best.hop_ms,
        jitter_ms=best.jitter_ms,
    )
