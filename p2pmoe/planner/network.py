"""网络测量层：分位数探测、缓存与闸门。

对应文档：
* 第〇部分「分位数记号」：所有配对代价均为 k 次采样的 p50 / p95，
  规划一律用 p50，尾部由 p95 单独设闸。
* I.2.3 尾闸与抖动闸
* II.1  抖动进硬屏蔽（p95 − p50 > J_cap 的链路不参与求解）

设计要点：规划器只通过 NetworkOracle 这一个接口看网络，它不假设任何拓扑
结构（II.3.1(a)：纯实测驱动，不依赖加法可分离、块状结构或任何先验）。
真实部署时把 SimNetwork 换成打真实探测包的实现即可，规划器代码不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

__all__ = ["Probe", "NetworkOracle", "MeasurementCache"]


@dataclass(frozen=True)
class Probe:
    """一次 k 采样探测的结果。"""

    p50: float
    p95: float
    k: int

    @property
    def jitter(self) -> float:
        """p95 − p50。与 p50 同量级是分散环境的常态（I.1.1）。"""
        return self.p95 - self.p50


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


class NetworkOracle(Protocol):
    """逐对代价的实测来源。"""

    def probe(self, a: str, b: str, k: int) -> Probe:
        """对有序对 (a, b) 做 k 次采样，返回 p50 / p95。

        方向性：正向接口 (tail(f), head(b)) 与回环接口 (tail(b), head(f)) 是
        不同的有序对，实现可以对称也可以不对称。
        """
        ...


class MeasurementCache:
    """按需测量 + 缓存 + 闸门判定。

    II.1 的注释「代价 h(v,v′) 取 p50(k 次采样); 按需测量+缓存; p95 超 J_cap 的
    对直接屏蔽」在这里落地：求解器只调 p50() 与 blocked()，看不到采样细节。
    """

    def __init__(
        self,
        oracle: NetworkOracle,
        k: int = 8,
        j_cap_ms: float = 25.0,
        k_gate: int = 32,
    ):
        self.oracle = oracle
        self.k = k
        self.k_gate = k_gate
        self.j_cap = j_cap_ms
        self._cache: dict[tuple[str, str, int], Probe] = {}
        self._min_hop: dict[frozenset[str], float] = {}
        self.n_probes = 0
        """累计探测次数 —— 复杂度是 O(|T|·|H|·k) 次实测（II.3.2），值得计量。"""

    # -- 基本查询 ---------------------------------------------------------- #
    def get(self, a: str, b: str, k: int | None = None) -> Probe:
        kk = self.k if k is None else k
        key = (a, b, kk)
        hit = self._cache.get(key)
        if hit is None:
            hit = self.oracle.probe(a, b, kk)
            self._cache[key] = hit
            self.n_probes += kk
        return hit

    def p50(self, a: str, b: str) -> float:
        return self.get(a, b).p50

    def p95(self, a: str, b: str) -> float:
        return self.get(a, b).p95

    def jitter(self, a: str, b: str) -> float:
        return self.get(a, b).jitter

    # -- 闸门 -------------------------------------------------------------- #
    def blocked(self, a: str, b: str) -> bool:
        """抖动硬屏蔽：避免 p50 漂亮但尾部灾难的解（II.1）。

        用 k_gate（默认 32）而非 k_probe（默认 8）采样 —— p95 的经验估计在
        k=8 时系统性偏低且方差极大，用它过闸等于形同虚设，见 PlannerConfig.k_gate。
        """
        if a == b:
            return False
        return self.get(a, b, self.k_gate).jitter > self.j_cap

    # -- 派生量 ------------------------------------------------------------ #
    def access_offset(self, v: str, peers: Sequence[str]) -> float:
        """δ̂_v：节点 v 对一组对端的 p50 中位，作为「接入偏移」的实测代理。

        用于求解器预剪枝 N*(v) 的排序（II.1）。这不是对加法结构的假设 ——
        它只是一个排序用的启发式统计量，公共带（II.3）才是判定资格的地方。
        """
        vals = sorted(self.p50(v, p) for p in peers if p != v)
        if not vals:
            return 0.0
        n = len(vals)
        return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    def min_hop_p50(self, nodes: Sequence[str]) -> float:
        """一组节点内最快的一跳。用作 beam 搜索完成下界里的每跳代价（solver.py）。

        必须是真正的下界，否则 A* 式剪枝会误杀最优解；故对小规模池取全对最小值。
        结果按节点集合缓存，且底层 probe 本身也有缓存，代价只付一次。
        """
        key = frozenset(nodes)
        hit = self._min_hop.get(key)
        if hit is not None:
            return hit
        ids = sorted(nodes)
        best = float("inf")
        if len(ids) <= 40:
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    best = min(best, self.p50(a, b), self.p50(b, a))
        else:
            step = max(1, len(ids) // 40)
            probe_set = ids[::step]
            for i, a in enumerate(probe_set):
                for b in probe_set[i + 1 :]:
                    best = min(best, self.p50(a, b), self.p50(b, a))
        best = 0.0 if best == float("inf") else best
        self._min_hop[key] = best
        return best

    def endpoint_costs(
        self, nodes: Sequence[str], reference: Sequence[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """逐节点的入口/出口接入代价画像。

        head_cost[v] = median_{u∈ref} ŵ50(u, v)   —— v 做入口时正向接口的代价中位
        tail_cost[v] = median_{u∈ref} ŵ50(v, u)   —— v 做出口时回环的代价中位

        ref 通常取「能承载前段的节点」集合 —— 它们是未来所有接口的另一端。
        这两张表是纯实测统计量，不含任何拓扑假设（II.3.1(a)）。
        """
        head: dict[str, float] = {}
        tail: dict[str, float] = {}
        for v in nodes:
            ins = sorted(self.p50(u, v) for u in reference if u != v)
            outs = sorted(self.p50(v, u) for u in reference if u != v)
            head[v] = _median(ins)
            tail[v] = _median(outs)
        return head, tail

    def typical_hop_p50(self, nodes: Sequence[str], sample: int = 200) -> float:
        """典型跳 p50，用于 w_cap 的比例锚（I.2.3）。"""
        vals: list[float] = []
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                vals.append(self.p50(a, b))
                if len(vals) >= sample:
                    break
            if len(vals) >= sample:
                break
        if not vals:
            return 0.0
        vals.sort()
        n = len(vals)
        return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    def clear(self) -> None:
        self._cache.clear()
        self._min_hop.clear()
