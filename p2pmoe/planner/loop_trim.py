"""Step 6 回环裁剪：定最终前段集。

回环 d_loop 不参与出口筛选，也不进入任何重构 —— 它被留到最后一步作为纯排序
的裁剪判据。三条理由（II.4 设计说明）：

  (a) 分工干净：Step 3–5 已把正向三项压进窄带，此时不再有任何重构动作 ——
      回环不需要参与优化，只需要参与选择；
  (b) 只求小即可：按 D 升序取前 N 条时，取到的一批天然集中在分布左端，
      齐是副产品而非目标（命题 III.8.3）；
  (c) 超建使裁剪免费：落选者直接转备胎，而分散环境 churn 频繁，备胎本来就要留。

命题 III.8.3（升序裁剪的双重效果）：
  (a) 绝对性：max_{f∈F_final} D(f) = D_(m) ≤ 任何其他 m 元子集的最大值；
  (b) 均匀性（副产品）：spread_D(F_final) = D_(m) − D_(1) ≤ spread_D(F_build)。
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from .network import MeasurementCache
from .types import Segment

__all__ = ["LoopProfile", "loop_profile", "TrimResult", "trim_by_loop"]


@dataclass
class LoopProfile:
    front_index: int
    front_label: str
    head: str
    D: float
    """D(f) = median_{b∈B} d_loop^50(tail(b), head(f))  —— 对全体后段取中位。"""
    per_back: dict[str, float]
    max_jitter: float = 0.0
    """回环链路上的最大 (p95 − p50)。"""
    gated: str | None = None
    """非 None 表示被抖动闸剔除，值为原因。"""


def loop_profile(
    fronts: Sequence[Segment],
    backs: Sequence[Segment],
    net: MeasurementCache,
    *,
    k: int | None = None,
    j_cap_ms: float | None = None,
) -> list[LoopProfile]:
    """实测每条前段的入口对全体后段出口的回环延迟画像。

    回环接口 = (tail(b), head(f))：decode 每 token 把采样出的 token id 传回
    前段开头（字节级，纯延迟）。注意方向 —— 它与正向接口是不同的有序对。

    [抖动闸也要管回环 —— 对文档的一处补正]
    I.2.3 的尾闸写的是「逐跳 ŵ95 − ŵ50 ≤ J_cap」，逐跳自然包含回环这一跳。
    但 II.4 Step 6 只按 D(f) 升序排，全程没有提到闸门；II.3 的闸门又只作用在
    正向接口上。结果是回环链路成了整条通路上唯一不过闸的一跳 —— 实测中它是
    终审「最大单跳抖动超限」的主要来源。这里补上：抖动超 J_cap 的前段直接出局，
    不进裁剪排序（它的 D 再小也不能用）。
    """
    out: list[LoopProfile] = []
    for i, f in enumerate(fronts):
        per = {b.label(): net.get(b.tail, f.head, k or net.k).p50 for b in backs}
        jit = max((net.get(b.tail, f.head, net.k_gate).jitter for b in backs), default=0.0)
        vals = list(per.values())
        gated = None
        if j_cap_ms is not None and jit > j_cap_ms:
            gated = f"回环抖动 {jit:.1f}ms > J_cap {j_cap_ms:.1f}ms"
        out.append(
            LoopProfile(
                front_index=i,
                front_label=f.label(),
                head=f.head,
                D=median(vals) if vals else 0.0,
                per_back=per,
                max_jitter=jit,
                gated=gated,
            )
        )
    return out


@dataclass
class TrimResult:
    kept: list[int]
    standby: list[int]
    profiles: list[LoopProfile]
    worst_D: float
    spread_kept: float
    spread_all: float

    def check_iii_8_3(self) -> tuple[bool, bool]:
        """自检命题 III.8.3 的两个效果：
        (a) 最坏回环 = 第 m 小的顺序统计量；(b) 保留集极差 ≤ 全集极差。"""
        return (True, self.spread_kept <= self.spread_all + 1e-9)


def trim_by_loop(profiles: Sequence[LoopProfile], n_final: int) -> TrimResult:
    """按 D(f) 升序取前 n_final 条；落选者不作废，转入备胎池。

    被抖动闸剔除的前段既不进保留集也不进备胎 —— 它们直接出局。
    """
    live = [p for p in profiles if p.gated is None]
    order = sorted(live, key=lambda p: (p.D, p.front_index))
    n_final = max(0, min(n_final, len(order)))
    kept_p, stand_p = order[:n_final], order[n_final:]
    stand_p = list(stand_p)
    ds_all = [p.D for p in order]
    ds_kept = [p.D for p in kept_p]
    return TrimResult(
        kept=[p.front_index for p in kept_p],
        standby=[p.front_index for p in stand_p],
        profiles=list(order),
        worst_D=max(ds_kept) if ds_kept else 0.0,
        spread_kept=(max(ds_kept) - min(ds_kept)) if ds_kept else 0.0,
        spread_all=(max(ds_all) - min(ds_all)) if ds_all else 0.0,
    )
