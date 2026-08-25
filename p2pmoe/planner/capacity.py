"""II.7 条数的确定：分档估上限 → 打折 → 配额（不使用 Erlang）。

为什么不用 Erlang：其输入是到达率与平均占用时长，二者均非本问题给定量
（问题只给 λ_u），且平均占用时长依赖每 token 延迟 —— 而后者在放置完成前
不存在。分散环境下条数的实质是装箱问题。

对应文档：
* II.7.1 分档估上限（整档归零）
* II.7.2 打折得工作值
* II.7.3 配额与公平比
* III.6.1 最大余额法的两条性质
* III.8.5 分档估算的上界性
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from math import floor
from typing import Mapping, Sequence

from .types import Node, SegmentSpec, TaskProfile

__all__ = [
    "Tier",
    "TierReport",
    "classify_tiers",
    "enumerate_forms",
    "CapacityEstimate",
    "estimate_capacity_by_tier",
    "largest_remainder",
    "fair_ratios",
]


# --------------------------------------------------------------------------- #
# 分档
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tier:
    name: str
    usable_gb: float
    count: int


def classify_tiers(nodes: Sequence[Node], *, round_to: float = 0.5) -> list[Tier]:
    """按 (M_v − 预留) 分档。同档节点在内存判据上不可区分。"""
    buckets: dict[tuple[str, float], int] = {}
    for n in nodes:
        key = (n.tier, round(n.usable_gb / round_to) * round_to)
        buckets[key] = buckets.get(key, 0) + 1
    tiers = [Tier(name=k[0], usable_gb=k[1], count=v) for k, v in buckets.items()]
    tiers.sort(key=lambda t: -t.usable_gb)
    return tiers


# --------------------------------------------------------------------------- #
# 段形态
# --------------------------------------------------------------------------- #
def _can_fit(spec: SegmentSpec, caps: Sequence[float]) -> bool:
    """给定按链序排列的节点容量，问是否存在连续层切分把整段装下。

    层区间必须连续且有序（I.2.2），所以这是一个一维切分可行性问题。
    """
    n_slots = len(caps)
    lo0, hi0 = spec.layer_lo, spec.layer_hi

    def rec(start: int, idx: int) -> bool:
        if start > hi0:
            return idx == n_slots  # 层放完且槽位恰好用尽
        if idx >= n_slots:
            return False
        # 本槽尽可能多装（贪心上界），再向下回溯
        best = start - 1
        for hi in range(start, hi0 + 1):
            if spec.resident_gb(start, hi) <= caps[idx] + 1e-9:
                best = hi
            else:
                break
        if best < start:
            return False
        for hi in range(best, start - 1, -1):
            if rec(hi + 1, idx + 1):
                return True
        return False

    return rec(lo0, 0)


def enumerate_forms(
    spec: SegmentSpec,
    tiers: Sequence[Tier],
    *,
    max_nodes: int = 4,
    extra_nodes: int = 0,
) -> list[tuple[str, ...]]:
    """枚举 spec 的可行「段形态」——一条段占用哪些档的节点位。

    只保留最小节点数（= 最少跳数）的形态，外加 extra_nodes 层冗余。
    分散环境下多一跳即数十毫秒，故默认 extra_nodes = 0：形态集就是最优跳数形态。
    """
    names = [t.name for t in tiers]
    cap = {t.name: t.usable_gb for t in tiers}
    forms: list[tuple[str, ...]] = []
    n_min: int | None = None

    for n in range(1, max_nodes + 1):
        if n_min is not None and n > n_min + extra_nodes:
            break
        found_this_n = False
        for combo in combinations_with_replacement(names, n):
            ok = any(_can_fit(spec, [cap[c] for c in perm]) for perm in set(permutations(combo)))
            if ok:
                forms.append(tuple(sorted(combo, key=lambda c: -cap[c])))
                found_this_n = True
        if found_this_n and n_min is None:
            n_min = n
    return forms


@dataclass
class TierReport:
    tier: Tier
    serves: dict[str, bool]
    """spec 名 → 该档能否承担该 spec 某个形态里的至少一个节点位。"""

    @property
    def zeroed(self) -> bool:
        """整档归零：不能承担任何形态的任一节点位（II.7.1）。

        III.8.5：归零档对可行段数的贡献恰为零，故其内存计入总量必然导致高估。
        """
        return not any(self.serves.values())

    @property
    def verdict(self) -> str:
        n = sum(self.serves.values())
        if n == 0:
            return "整档归零"
        return "全能" if n == len(self.serves) else "部分"


# --------------------------------------------------------------------------- #
# 上限估算
# --------------------------------------------------------------------------- #
@dataclass
class CapacityEstimate:
    tiers: list[Tier]
    reports: list[TierReport]
    active_supply_gb: float
    """未归零档的可用内存总量。"""
    zeroed_supply_gb: float
    """归零档的内存量 —— 朴素估法会把它计入，这是最大的一个高估源。"""
    channel_demand_gb: float
    """单通道驻留量 = 前段 + 按 λ 加权的后段。"""
    n_max: int
    """文档口径的上限：活跃供给 ÷ 单通道需求，再受形态耦合约束封顶。
    III.8.5 明确它是上界而非可达值（忽略网络与配对可行性）。"""
    n_max_slots: int
    """按节点位做精确整数配平的可达上限。n_max − n_max_slots 的差额正是
    θ 折扣要吸收的碎片之一。"""
    waste_ratio: float
    """归零档占全池内存的比例 —— 朴素估法的废料率。"""
    forms: dict[str, list[tuple[str, ...]]]
    notes: list[str] = field(default_factory=list)


def estimate_capacity_by_tier(
    nodes: Sequence[Node],
    front_spec: SegmentSpec,
    back_specs: Mapping[str, SegmentSpec],
    tasks: Sequence[TaskProfile],
    *,
    max_nodes: int = 4,
) -> CapacityEstimate:
    """II.7.1 Step 1：分档估上限。

    朴素做法「全网总内存 ÷ 单通道驻留量」会系统性高估，因为节点排他 + 层区间
    连续切分导致废料不可回收，且整档机器可能一台都用不上却仍被计入总量。
    """
    tiers = classify_tiers(nodes)
    specs: dict[str, SegmentSpec] = {"front": front_spec}
    specs.update({f"back:{u}": s for u, s in back_specs.items()})

    forms = {name: enumerate_forms(s, tiers, max_nodes=max_nodes) for name, s in specs.items()}

    reports: list[TierReport] = []
    for t in tiers:
        serves = {
            name: any(t.name in form for form in forms[name]) for name in specs
        }
        reports.append(TierReport(tier=t, serves=serves))

    active = sum(r.tier.usable_gb * r.tier.count for r in reports if not r.zeroed)
    zeroed = sum(r.tier.usable_gb * r.tier.count for r in reports if r.zeroed)
    total = active + zeroed

    lam = {t.name: t.lam for t in tasks}
    demand = front_spec.total_gb() + sum(
        lam[u] * s.total_gb() for u, s in back_specs.items()
    )

    n_mem = int(floor(active / demand)) if demand > 0 else 0

    # 形态耦合封顶：若某 task 的每条后段都必须占用某个稀缺档的位，则该 task
    # 的条数受该档台数限制，进而通过 λ 配额反向封住总条数。
    notes: list[str] = []
    supply_by_tier = {r.tier.name: r.tier.count for r in reports}
    n_coupled = n_mem
    for u, s in back_specs.items():
        fs = forms[f"back:{u}"]
        if not fs:
            notes.append(f"task {u} 无可行形态 —— 该池不可建")
            return CapacityEstimate(
                tiers, reports, active, zeroed, demand, 0, 0, zeroed / total if total else 0.0,
                forms, notes,
            )
        # 每种形态对各档的需求；取「最省稀缺档」的形态作为该 task 的下界需求
        best_cap = 0
        for form in fs:
            cap_this = min(
                supply_by_tier.get(tn, 0) // form.count(tn)
                for tn in set(form)
            )
            best_cap = max(best_cap, cap_this)
        if lam[u] > 0:
            implied = int(floor(best_cap / lam[u]))
            if implied < n_coupled:
                n_coupled = implied
                notes.append(
                    f"总条数受 task {u} 的形态耦合封顶：该池最多 {best_cap} 条 "
                    f"→ 按 λ={lam[u]} 反解总量 ≤ {implied}"
                )

    n_max = max(0, n_coupled)
    n_slots = _exact_slot_packing(reports, forms, tasks, n_max)

    if n_slots < n_max:
        notes.append(
            f"精确位配平只能建 {n_slots} 条，而内存上界给出 {n_max} 条；"
            f"差额由 θ 折扣吸收（II.7.2 / III.8.5 注）"
        )

    return CapacityEstimate(
        tiers=tiers,
        reports=reports,
        active_supply_gb=active,
        zeroed_supply_gb=zeroed,
        channel_demand_gb=demand,
        n_max=n_max,
        n_max_slots=n_slots,
        waste_ratio=zeroed / total if total else 0.0,
        forms=forms,
        notes=notes,
    )


def _exact_slot_packing(
    reports: Sequence[TierReport],
    forms: Mapping[str, list[tuple[str, ...]]],
    tasks: Sequence[TaskProfile],
    n_upper: int,
) -> int:
    """精确整数配平：能否用现有节点位真的建出 N 个通道（N 前段 + 按配额的后段）。

    从 n_upper 向下试，返回第一个可行的 N。这是 N_max 的可达版本；它仍忽略
    网络约束（同档两个节点可能因链路不合格而不能组成一条段），所以也只是
    比 n_max 更紧的上界，不是保证值。
    """
    supply0 = tuple(sorted((r.tier.name, r.tier.count) for r in reports if not r.zeroed))
    tier_names = [n for n, _ in supply0]

    def feasible(n: int) -> bool:
        if n <= 0:
            return True
        quota = largest_remainder([t.lam for t in tasks], n)
        need: list[str] = ["front"] * n
        for t, q in zip(tasks, quota):
            need += [f"back:{t.name}"] * q
        # 稀缺优先：可行形态少的段先安排
        need.sort(key=lambda s: len(forms.get(s, [])))
        supply = dict(supply0)

        def dfs(i: int) -> bool:
            if i >= len(need):
                return True
            for form in sorted(forms[need[i]], key=len):
                if all(supply.get(tn, 0) >= form.count(tn) for tn in set(form)):
                    for tn in form:
                        supply[tn] -= 1
                    if dfs(i + 1):
                        return True
                    for tn in form:
                        supply[tn] += 1
            return False

        return dfs(0)

    for n in range(n_upper, -1, -1):
        if feasible(n):
            return n
    return 0


# --------------------------------------------------------------------------- #
# II.7.3 配额与公平比
# --------------------------------------------------------------------------- #
def largest_remainder(lams: Sequence[float], n: int, *, min_one: bool = True) -> list[int]:
    """最大余额法（命题 III.6.1）。

    (a) 配额性质：∀u, N_u ∈ {⌊q_u⌋, ⌈q_u⌉}
    (b) 在 Σ N_u = n 的整数分配中最小化 max_u |N_u − q_u|

    min_one=True 时按文档口径：对 q_u < 1 的 task 强制置 1，再在余下名额上
    重跑；性质 (b) 在剩余集合上成立。
    """
    m = len(lams)
    if m == 0 or n <= 0:
        return [0] * m
    total_lam = sum(lams)
    if total_lam <= 0:
        return [0] * m

    if min_one:
        forced = [i for i, lam in enumerate(lams) if lam / total_lam * n < 1.0]
        if forced and len(forced) < m and n > len(forced):
            out = [0] * m
            for i in forced:
                out[i] = 1
            rest = [i for i in range(m) if i not in set(forced)]
            sub = largest_remainder([lams[i] for i in rest], n - len(forced), min_one=False)
            for i, v in zip(rest, sub):
                out[i] = v
            return out

    q = [lam / total_lam * n for lam in lams]
    base = [int(floor(x)) for x in q]
    rem = n - sum(base)
    order = sorted(range(m), key=lambda i: (-(q[i] - base[i]), i))
    for i in order[:rem]:
        base[i] += 1
    return base


def fair_ratios(quota: Mapping[str, int], lams: Mapping[str, float]) -> dict[str, float]:
    """公平比 fair(u) = (N_B(u)/N_B^total) / λ_u  （II.7.3）。

    fair 越接近 1 越均衡；min_u fair(u) 是系统最坏委屈度，argmin 即瓶颈池 ——
    下一条链加给它。全程不需要到达率。
    """
    total = sum(quota.values())
    if total == 0:
        return {u: 0.0 for u in quota}
    return {
        u: (quota[u] / total) / lams[u] if lams[u] > 0 else float("inf")
        for u in quota
    }
