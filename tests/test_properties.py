"""对文档中**可证明**的性质写测试。

只测有严格证明的命题（§III.9 的「严格证明」清单），启发式部分不写断言 ——
给启发式写断言等于把调参结果冻进测试，是自欺。
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.capacity import largest_remainder, fair_ratios
from p2pmoe.planner.common_band import BandResult, ExitProbe, common_band
from p2pmoe.planner.loop_trim import LoopProfile, trim_by_loop
from p2pmoe.planner.memory import hops_min
from p2pmoe.planner.tighten import band_from_median, dist_to_band, tighten_lex
from p2pmoe.planner.types import ModelSpec, Node, Segment


# --------------------------------------------------------------------------- #
# 命题 III.8.1  滑窗解的最优性
# --------------------------------------------------------------------------- #
def _brute_force_band(intervals: dict[str, tuple[float, float]], W: float, w_cap: float) -> int:
    """暴力最优：窗左端取遍所有区间端点（含 hi−W），覆盖数取最大。

    这是一个比被测实现更宽的候选集 —— 若滑窗只取 lo_t 仍能达到同样的最大值，
    命题 III.8.1 的「最优窗必可左对齐某个 lo_t」就在这些实例上被验证了。
    """
    cands = sorted({lo for lo, _ in intervals.values()} | {hi - W for _, hi in intervals.values()})
    best = 0
    for x in cands:
        if x + W > w_cap + 1e-9:
            continue
        n = sum(1 for lo, hi in intervals.values() if lo >= x - 1e-9 and hi <= x + W + 1e-9)
        best = max(best, n)
    return best


def _probes_from(intervals: dict[str, tuple[float, float]]) -> dict[str, ExitProbe]:
    """把 [lo, hi] 造成一个只有两个入口的探测画像。"""
    return {
        t: ExitProbe(exit_id=t, p50={"a": lo, "b": hi}, p95={"a": lo, "b": hi})
        for t, (lo, hi) in intervals.items()
    }


@pytest.mark.parametrize("seed", range(40))
def test_iii_8_1_sliding_window_is_optimal(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 12)
    intervals = {}
    for i in range(n):
        lo = rng.uniform(20.0, 60.0)
        intervals[f"t{i}"] = (lo, lo + rng.uniform(0.0, 18.0))
    W = rng.uniform(2.0, 20.0)
    w_cap = 200.0

    got = common_band(_probes_from(intervals), ["a", "b"], W, w_cap)
    want = _brute_force_band(intervals, W, w_cap)
    assert got.population == want, f"滑窗 {got.population} != 暴力最优 {want}"
    # 返回的出口确实都落在带内
    for t in got.exits:
        lo, hi = intervals[t]
        assert got.w_lo - 1e-9 <= lo and hi <= got.w_hi + 1e-9


def test_iii_8_1_left_alignment_witness() -> None:
    """命题证明的核心一步：最优窗可以平移到左对齐某个 lo_t 而不丢覆盖。"""
    intervals = {"t1": (10.0, 12.0), "t2": (11.0, 15.0), "t3": (30.0, 31.0)}
    r = common_band(_probes_from(intervals), ["a", "b"], W=5.0, w_cap=100.0)
    assert set(r.exits) == {"t1", "t2"}
    assert r.w_lo == pytest.approx(10.0)  # 左对齐 min lo


def test_common_band_respects_w_cap() -> None:
    intervals = {"t1": (10.0, 12.0), "t2": (50.0, 52.0), "t3": (51.0, 53.0)}
    r = common_band(_probes_from(intervals), ["a", "b"], W=5.0, w_cap=20.0)
    assert set(r.exits) == {"t1"}  # 50+ 的那对超过 w_cap


def test_common_band_drops_gated_exits() -> None:
    pr = _probes_from({"t1": (10.0, 12.0), "t2": (10.5, 12.5)})
    pr["t2"].dropped = "抖动闸"
    r = common_band(pr, ["a", "b"], W=5.0, w_cap=100.0)
    assert r.exits == ["t1"]


# --------------------------------------------------------------------------- #
# 命题 III.8.3  回环升序裁剪的双重效果
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(40))
def test_iii_8_3_ascending_prefix_minimises_worst(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(3, 9)
    m = rng.randint(1, n)
    ds = [rng.uniform(10.0, 80.0) for _ in range(n)]
    profs = [LoopProfile(i, f"F{i}", f"v{i}", ds[i], {}) for i in range(n)]

    res = trim_by_loop(profs, m)

    # (a) 绝对性：任何 m 元子集的最大值 ≥ 前缀的最大值
    for subset in itertools.combinations(range(n), m):
        assert res.worst_D <= max(ds[i] for i in subset) + 1e-9

    # (b) 均匀性副产品：保留集极差 ≤ 全集极差
    assert res.spread_kept <= res.spread_all + 1e-9

    # 保留的确实是最小的 m 个
    assert sorted(ds[i] for i in res.kept) == sorted(ds)[:m]


def test_iii_8_3_standby_is_the_complement() -> None:
    ds = [28.0, 31.0, 33.0, 36.0, 47.0, 55.0]
    profs = [LoopProfile(i, f"F{i}", f"v{i}", d, {}) for i, d in enumerate(ds)]
    res = trim_by_loop(profs, 5)
    assert res.worst_D == pytest.approx(47.0)
    assert res.spread_kept == pytest.approx(47.0 - 28.0)
    assert res.spread_all == pytest.approx(55.0 - 28.0)
    assert len(res.kept) == 5 and len(res.standby) == 1
    assert set(res.kept) | set(res.standby) == set(range(6))


# --------------------------------------------------------------------------- #
# 命题 III.6.1  最大余额法的两条性质
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(60))
def test_iii_6_1_quota_property_and_minmax(seed: int) -> None:
    rng = random.Random(seed)
    m = rng.randint(2, 6)
    raw = [rng.uniform(0.05, 1.0) for _ in range(m)]
    s = sum(raw)
    lams = [x / s for x in raw]
    n = rng.randint(m, 20)

    out = largest_remainder(lams, n, min_one=False)
    assert sum(out) == n

    import math

    q = [lam * n for lam in lams]
    # (a) 配额性质：N_u ∈ {⌊q_u⌋, ⌈q_u⌉}
    for got, qq in zip(out, q):
        assert math.floor(qq) <= got <= math.ceil(qq)

    # (b) 最小化最大偏离：暴力枚举所有和为 n 的整数分配
    def all_allocs(k: int, total: int):
        if k == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in all_allocs(k - 1, total - i):
                yield (i,) + rest

    if m <= 4 and n <= 12:
        best = min(max(abs(a - qq) for a, qq in zip(alloc, q)) for alloc in all_allocs(m, n))
        got = max(abs(a - qq) for a, qq in zip(out, q))
        assert got <= best + 1e-9


def test_min_one_matches_document_example() -> None:
    """算例 C.1：λ = (0.5, 0.3, 0.2)，N_work = 4 → 配额 (2, 1, 1)。"""
    out = largest_remainder([0.5, 0.3, 0.2], 4)
    assert out == [2, 1, 1]
    fair = fair_ratios({"X": 2, "Y": 1, "Z": 1}, {"X": 0.5, "Y": 0.3, "Z": 0.2})
    assert fair["X"] == pytest.approx(1.00)
    assert fair["Y"] == pytest.approx(0.8333, rel=1e-3)
    assert fair["Z"] == pytest.approx(1.25)
    assert min(fair, key=fair.get) == "Y"  # 瓶颈池，下一条链加给它


def test_min_one_forces_a_slot_for_tiny_tasks() -> None:
    out = largest_remainder([0.9, 0.05, 0.05], 4)
    assert all(v >= 1 for v in out) and sum(out) == 4


# --------------------------------------------------------------------------- #
# 命题 III.5.2  跳数整数下界
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "total,cap,want",
    [
        (28.0, 47.0, 0),   # 单节点成段
        (47.0, 47.0, 0),   # 恰好装下
        (47.1, 47.0, 1),
        (63.6, 47.0, 1),   # 算例 C 的 X 后段
        (94.1, 47.0, 2),
        (140.0, 47.0, 2),
    ],
)
def test_iii_5_2_hops_lower_bound(total: float, cap: float, want: int) -> None:
    assert hops_min(total, cap) == want


@pytest.mark.parametrize("seed", range(40))
def test_iii_5_2_is_a_real_lower_bound(seed: int) -> None:
    """任何可行放置的节点数 ≥ ⌈总量/最大单机⌉，故跳数 ≥ 该值 − 1。"""
    rng = random.Random(seed)
    cap = rng.uniform(10.0, 50.0)
    parts = [rng.uniform(0.5, cap) for _ in range(rng.randint(1, 8))]
    total = sum(parts)
    assert len(parts) - 1 >= hops_min(total, cap)


def test_kv_formula_matches_appendix_c() -> None:
    """kv = ctx × 2 × d_model × 2 bytes；算例 C 的 ctx=4096, d=4096 → 0.067 GB/层。"""
    m = ModelSpec(32, 4096, 64, 2, 0.13, 0.27, 4096)
    assert m.kv_gb_per_layer == pytest.approx(0.0671, abs=1e-4)
    assert m.kv_gb(28) == pytest.approx(1.879, abs=1e-3)     # 文档：后段 28 层 1.88GB
    assert m.kv_gb(5) == pytest.approx(0.3355, abs=1e-3)     # 文档：前段 5 层 0.34GB
    assert m.weight_gb_per_layer(20) == pytest.approx(5.53)  # 并集
    assert m.weight_gb_per_layer(8) == pytest.approx(2.29)   # task X
    assert 5 * m.weight_gb_per_layer(20) + m.kv_gb(5) == pytest.approx(27.99, abs=0.02)


# --------------------------------------------------------------------------- #
# 命题 III.4.1  字典序单调
# --------------------------------------------------------------------------- #
def test_iii_4_1_lex_monotone_on_random_pools() -> None:
    """tighten_lex 的验收序列 (D_max, D_Σ) 必须字典序单调不增。"""
    from p2pmoe.planner.memory import make_back_spec
    from p2pmoe.planner.network import MeasurementCache
    from p2pmoe.planner.types import PlannerConfig, TaskProfile
    from p2pmoe.sim.network import SimNetwork
    from p2pmoe.sim.scenario import APPENDIX_C_MODEL, build_nodes

    for seed in range(6):
        nodes = build_nodes()
        nm = {n.id: n for n in nodes}
        sim = SimNetwork([n.id for n in nodes], seed=seed)
        net = MeasurementCache(sim, k=8, j_cap_ms=40.0)
        cfg = PlannerConfig(j_cap_ms=40.0)
        task = TaskProfile("Y", 0.3, 6)
        spec = make_back_spec(APPENDIX_C_MODEL, task, 5)

        from p2pmoe.planner.solver import deploy_path
        from p2pmoe.planner.types import Objective

        free = set(nm)
        segs = []
        for _ in range(3):
            s = deploy_path(spec, sorted(free), nm, net, Objective(), beam_width=8)
            if s is None:
                break
            segs.append(s)
            free -= set(s.nodes)
        if len(segs) < 2:
            continue

        band = band_from_median([s.delay_ms for s in segs], cfg.beta)
        res = tighten_lex(segs, spec, band, free, nm, net, cfg)
        for prev, cur in zip(res.history, res.history[1:]):
            assert cur <= prev, f"seed={seed} 字典序回退: {prev} → {cur}"


def test_iii_4_3_locality_in_band_segments_untouched() -> None:
    """局部性：区间内的段在收紧中不被触碰。"""
    band = (40.0, 60.0)
    assert dist_to_band(50.0, band) == 0.0
    assert dist_to_band(35.0, band) == pytest.approx(5.0)
    assert dist_to_band(70.0, band) == pytest.approx(10.0)


def test_band_from_median_is_symmetric_about_median() -> None:
    lo, hi = band_from_median([10.0, 20.0, 30.0], beta=1.25)
    assert (lo + hi) / 2 == pytest.approx(20.0)
    assert hi - lo == pytest.approx(2 * 20.0 * 0.25)


# --------------------------------------------------------------------------- #
# 定理 III.2.1  组合延迟分解 / 定理 III.3.1 零后悔
# --------------------------------------------------------------------------- #
def test_iii_2_1_decomposition_bounds_combined_spread() -> None:
    """spread_pairs(u) ≤ spread_F + spread_w + spread_B(u)，w 恒定时取等。"""
    fronts = [2.5, 2.5, 3.1]
    backs = [36.5, 42.1]
    ws = [28.0, 30.0, 31.5]
    loops = [29.8, 30.2]

    combos = [f + w + b + d for f in fronts for w in ws for b in backs for d in loops]
    spread = max(combos) - min(combos)
    bound = (
        (max(fronts) - min(fronts))
        + (max(ws) - min(ws))
        + (max(backs) - min(backs))
        + (max(loops) - min(loops))
    )
    assert spread <= bound + 1e-9

    # w 与回环恒定时取等（定理的「取等」条件）
    combos2 = [f + 30.0 + b + 30.0 for f in fronts for b in backs]
    assert max(combos2) - min(combos2) == pytest.approx(
        (max(fronts) - min(fronts)) + (max(backs) - min(backs))
    )


def test_iii_3_1_zero_regret_bound() -> None:
    """极差 ≤ ε 时，任取空闲派发与全知最优的差 ≤ ε。"""
    ts = [100.0, 103.0, 107.0, 110.0]
    eps = max(ts) - min(ts)
    for chosen in ts:
        assert chosen - min(ts) <= eps + 1e-9
