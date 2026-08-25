"""II.3 公共中值域：从后段入口反推前段出口。

后段 cluster 固定后，前段出口的资格是一个被动条件：它必须对全部后段入口都
表现得又快又一致。这可以直接以实测表达，无需假设任何网络结构。

    求带 [w_lo, w_hi] 与出口集 Q_F^out，使
      ∀t ∈ Q_F^out, ∀v ∈ H:  ŵ50(t,v) ∈ [w_lo, w_hi]
      且 w_hi ≤ w_cap,  带宽 W = w_hi − w_lo ≤ η·median T50
    在满足上述前提下最大化 |Q_F^out|。

三点性质（II.3.1）：
  (a) 纯实测驱动 —— 不依赖加法可分离、块状结构或任何先验；
  (b) 目标是人口不是速度 —— 绝对速度由 w_cap 兜住、带宽由 W 兜住，剩下的
      自由度应换成人口，因为人口是 Step 5/6 的余量来源；
  (c) 带宽与人口此消彼长 —— 故实际操作是扫 W 画人口曲线取拐点。

命题 III.8.1：滑窗解是精确最优的（窗左端只需取遍各 lo_t）。这是全流程中
仅有的两处精确最优结论之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Iterable, Mapping, Sequence

from .network import MeasurementCache
from .types import PlannerConfig

__all__ = [
    "ExitProbe",
    "probe_exits",
    "BandResult",
    "common_band",
    "sweep_bandwidth",
    "pick_bandwidth",
    "diagnose_outlier_entries",
]


@dataclass
class ExitProbe:
    """一个候选前段出口 t 对全体后段入口 H 的实测画像。"""

    exit_id: str
    p50: dict[str, float]
    p95: dict[str, float]
    dropped: str | None = None
    """非 None 表示被闸门剔除，值为原因。"""

    def interval(self, entries: Sequence[str]) -> tuple[float, float]:
        """[lo_t, hi_t] = [min_v ŵ50(t,v), max_v ŵ50(t,v)]（II.3.2）。"""
        vals = [self.p50[v] for v in entries if v in self.p50]
        return (min(vals), max(vals)) if vals else (0.0, 0.0)


def probe_exits(
    candidates: Sequence[str],
    entries: Sequence[str],
    net: MeasurementCache,
    cfg: PlannerConfig,
    *,
    w_cap95: float | None = None,
) -> dict[str, ExitProbe]:
    """实测每个候选出口 × 每个入口，k ≥ 8 次采样，并过尾闸与抖动闸。

    复杂度 O(|T|·|H|·k) 次实测（II.3.2）。
    """
    w95 = w_cap95 if w_cap95 is not None else cfg.w_cap95_ms
    out: dict[str, ExitProbe] = {}
    for t in candidates:
        p50: dict[str, float] = {}
        p95: dict[str, float] = {}
        for v in entries:
            p50[v] = net.get(t, v, cfg.k_probe).p50
            # 闸门判定另用 k_gate 采样：p95 在 k=8 时估计不住（见 PlannerConfig.k_gate）
            p95[v] = net.get(t, v, cfg.k_gate).p95
        ep = ExitProbe(exit_id=t, p50=p50, p95=p95)
        if w95 is not None and max(p95.values(), default=0.0) > w95:
            ep.dropped = f"尾闸: max p95 {max(p95.values()):.1f} > {w95:.1f}"
        else:
            jit = max((net.get(t, v, cfg.k_gate).jitter for v in entries), default=0.0)
            if jit > cfg.j_cap_ms:
                ep.dropped = f"抖动闸: max (p95−p50) {jit:.1f} > J_cap {cfg.j_cap_ms:.1f}"
        out[t] = ep
    return out


@dataclass
class BandResult:
    exits: list[str]
    w_lo: float
    w_hi: float
    W: float

    @property
    def population(self) -> int:
        return len(self.exits)


def common_band(
    probes: Mapping[str, ExitProbe],
    entries: Sequence[str],
    W: float,
    w_cap: float,
) -> BandResult:
    """滑窗求最大公共带内出口集 —— 精确最优（命题 III.8.1）。

    问题等价于：在数轴上放一个长度 W 的窗，使被**完全覆盖**的区间数最多。
    最优窗必可在某个 lo_t 处左对齐，故只需取遍各 lo_t。
    """
    ivs: dict[str, tuple[float, float]] = {}
    for t, ep in probes.items():
        if ep.dropped:
            continue
        lo, hi = ep.interval(entries)
        if hi - lo > W + 1e-9:
            continue  # 自身跨度已超带宽
        ivs[t] = (lo, hi)

    best = BandResult(exits=[], w_lo=0.0, w_hi=0.0, W=W)
    for w_lo in sorted({lo for lo, _ in ivs.values()}):
        w_hi = w_lo + W
        if w_hi > w_cap + 1e-9:
            break
        S = sorted(t for t, (lo, hi) in ivs.items() if lo >= w_lo - 1e-9 and hi <= w_hi + 1e-9)
        if len(S) > best.population:
            best = BandResult(exits=S, w_lo=w_lo, w_hi=w_hi, W=W)
    return best


def sweep_bandwidth(
    probes: Mapping[str, ExitProbe],
    entries: Sequence[str],
    w_cap: float,
    W_grid: Sequence[float],
) -> list[tuple[float, BandResult]]:
    """II.3.3：W 不预先拍定，而是由人口需求反解 —— 扫 W 画 (W, |S|) 曲线。"""
    return [(W, common_band(probes, entries, W, w_cap)) for W in W_grid]


def pick_bandwidth(
    curve: Sequence[tuple[float, BandResult]],
    n_target: int,
    w_max: float,
) -> tuple[BandResult, str]:
    """在硬上限 W ≤ η·median T50 内取拐点。

    规则：
      1. 若存在 W ≤ w_max 使 |S| ≥ n_target，取**最小**的这样的 W
         （多余的带宽换不来人口，只会放大极差）；
      2. 否则取 W ≤ w_max 内人口最大者中最小的 W，并返回「人口不足」提示 ——
         调用方应转入异类入口诊断（diagnose_outlier_entries），而不是急着放宽带宽。
    """
    inside = [(W, r) for W, r in curve if W <= w_max + 1e-9]
    if not inside:
        raise ValueError("W 网格里没有任何点落在 η·median 上限内")
    ok = [(W, r) for W, r in inside if r.population >= n_target]
    if ok:
        W, r = min(ok, key=lambda x: x[0])
        return r, f"取 W={W:.1f}ms（满足人口目标 {n_target} 的最小带宽）"
    best_pop = max(r.population for _, r in inside)
    W, r = min((x for x in inside if x[1].population == best_pop), key=lambda x: x[0])
    return r, (
        f"取 W={W:.1f}ms，人口 {r.population} < 目标 {n_target} —— "
        f"在 W ≤ {w_max:.1f}ms 内凑不够，须先做异类入口诊断"
    )


def diagnose_outlier_entries(
    probes: Mapping[str, ExitProbe],
    entries: Sequence[str],
    W: float,
    w_cap: float,
) -> list[tuple[str, int, int]]:
    """异类入口诊断（II.3.3）。

    入口里若存在异类（对全网普遍偏差的接入点），它会同时抬高所有候选出口的
    hi_t、压缩公共带。诊断方法：逐个剔除入口看 |S| 的跃升，跃升最大者即异类。

    返回 [(入口, 剔除后的人口, 跃升幅度)]，按跃升降序。
    """
    base = common_band(probes, entries, W, w_cap).population
    out: list[tuple[str, int, int]] = []
    for v in entries:
        rest = [x for x in entries if x != v]
        if not rest:
            continue
        pop = common_band(probes, rest, W, w_cap).population
        out.append((v, pop, pop - base))
    out.sort(key=lambda x: -x[2])
    return out


def pick_outlier(
    diag: Sequence[tuple[str, int, int]],
    *,
    exclude: Sequence[str] = (),
    min_jump: int = 2,
) -> tuple[str, int, int] | None:
    """从诊断结果里挑出**真正的**异类入口，挑不出就返回 None。

    为什么需要判据而不是直接取第一名：剔除任何一个入口都会放松约束，人口多半
    会涨一点。把「涨了 1」当成异类会引发反馈震荡 —— 实测中它会把一条好后段
    拆掉重建成三跳的怪物，下一轮再换回来。真正的异类有两个特征：
      * 跃升有绝对幅度（≥ min_jump）；
      * 跃升明显高于第二名（否则只是普遍性的约束放松）。
    两条都不满足时，人口不足是池群的结构问题，应转向放宽 η 或域分裂（II.6）。
    """
    ex = set(exclude)
    cands = [d for d in diag if d[0] not in ex]
    if not cands:
        return None
    best = cands[0]
    if best[2] < min_jump:
        return None
    second = max((d[2] for d in diag[1:]), default=0)
    if best[2] <= second:
        return None
    return best
