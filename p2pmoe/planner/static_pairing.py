"""静态配对：把每条后段固定绑到一条前段，配对在部署时定死。

这是**简化模式**，与文档的主线（盲绑 + 在线识别 + 任意组合）是两条路：

| | 动态（文档主线） | 静态（本模块） |
|---|---|---|
| 到达时 task | 未知，由前段识别 | 调用方给定 |
| 配对 | 在线 pop，任意组合 | 部署时定死 |
| 控制面 RTT | 识别完要问协调器要后段 | 无 |
| 换绑 | 有（通道二检出） | 无（不存在误绑） |
| 公共中值域 | 必需 —— 它保证任意组合成立 | 退化为「只需选中的那几对好」|

**代价**：放弃「到达时 task 未知」这个前提（I.1.1），也放弃了负载重分配 ——
某个 task 流量涨了不能临时借别的通道，只能重新下发配对。

**收益**：少一个控制面 RTT，少一整套分类器/检出/换绑，好调试。

配对怎么选
----------
每条后段配一条前段，最小化组合延迟之和。这是一个矩形指派问题；这里用贪心：
把所有 (f,b) 对按延迟升序取，两边都空闲就成交。贪心不保证最优，但在这个规模
（个位数条数）与这个目标（延迟差异被抖动淹没，命题 III.3.3）下，与最优的差距
远小于抖动本身 —— 花力气上匈牙利算法不值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .network import MeasurementCache
from .types import Segment

__all__ = ["StaticPair", "StaticWiring", "assign_static_pairs"]


@dataclass(frozen=True)
class StaticPair:
    front_id: str
    back_id: str
    task: str
    front: Segment
    back: Segment
    w_fwd: float
    """正向接口 p50：tail(f) → head(b)。"""
    d_loop: float
    """回环 p50：tail(b) → head(f)。"""

    @property
    def t50(self) -> float:
        return self.front.delay_ms + self.w_fwd + self.back.delay_ms + self.d_loop


@dataclass
class StaticWiring:
    pairs: list[StaticPair] = field(default_factory=list)
    spare_fronts: list[str] = field(default_factory=list)
    """没配上的前段 —— 天然的备胎，churn 时改一条配置就能顶上。"""
    unpaired_backs: list[str] = field(default_factory=list)
    """没配上的后段 —— 前段不够，说明该加机器或降条数。"""

    def as_map(self) -> dict[str, tuple[str, str]]:
        """{前段 id: (后段 id, task)} —— 直接喂给 Coordinator(static_wiring=...)。"""
        return {p.front_id: (p.back_id, p.task) for p in self.pairs}

    @property
    def spread_ms(self) -> float:
        """选中的这几对之间的组合延迟极差。

        注意口径变了：动态模式下要求**所有** N_F × N_B 组合齐（这是公共中值域
        的职责），静态模式只需要**被选中的这几对**齐 —— 约束弱得多，所以同一个
        池子在静态模式下更容易达标。这不是方法变好了，是问题变简单了。
        """
        if not self.pairs:
            return 0.0
        ts = [p.t50 for p in self.pairs]
        return max(ts) - min(ts)

    def summary(self) -> str:
        if not self.pairs:
            return "没有配出任何通道"
        ts = [p.t50 for p in self.pairs]
        return (
            f"{len(self.pairs)} 条静态通道，组合 p50 {min(ts):.0f}–{max(ts):.0f}ms"
            f"（极差 {self.spread_ms:.0f}ms）"
            + (f"；备胎前段 {len(self.spare_fronts)} 条" if self.spare_fronts else "")
            + (f"；⚠ {len(self.unpaired_backs)} 条后段没配上" if self.unpaired_backs else "")
        )


def assign_static_pairs(
    fronts: Sequence[Segment],
    backs: Mapping[str, Sequence[Segment]],
    net: MeasurementCache,
    *,
    k: int = 16,
    front_ids: Sequence[str] | None = None,
) -> StaticWiring:
    """给每条后段配一条前段，贪心最小化组合延迟。

    front_ids / back id 的命名与 `DeploymentManifest` 保持一致（F0、BX0 …），
    这样配对结果能直接对上清单里的段。
    """
    f_ids = list(front_ids) if front_ids else [f"F{i}" for i in range(len(fronts))]
    f_by_id = dict(zip(f_ids, fronts))

    b_by_id: dict[str, tuple[str, Segment]] = {}
    for task, segs in backs.items():
        for i, b in enumerate(segs):
            b_by_id[f"B{task}{i}"] = (task, b)

    # 所有候选对，按组合延迟升序
    cands: list[StaticPair] = []
    for fid, f in f_by_id.items():
        for bid, (task, b) in b_by_id.items():
            w = net.get(f.tail, b.head, k).p50
            d = net.get(b.tail, f.head, k).p50
            cands.append(StaticPair(fid, bid, task, f, b, w, d))
    cands.sort(key=lambda p: p.t50)

    used_f: set[str] = set()
    used_b: set[str] = set()
    chosen: list[StaticPair] = []
    for p in cands:
        if p.front_id in used_f or p.back_id in used_b:
            continue
        chosen.append(p)
        used_f.add(p.front_id)
        used_b.add(p.back_id)

    chosen.sort(key=lambda p: (p.task, p.back_id))
    return StaticWiring(
        pairs=chosen,
        spare_fronts=sorted(set(f_by_id) - used_f),
        unpaired_backs=sorted(set(b_by_id) - used_b),
    )
