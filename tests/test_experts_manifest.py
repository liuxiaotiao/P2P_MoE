"""专家身份层与部署清单的测试。

同样只断言可证明/构造性成立的性质：覆盖率定义、并集支配性、q 的定义、
清单七项校验能真的抓到违规。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.experts import (
    ActivationProfile,
    build_placement,
    detectability,
    detectability_matrix,
    expected_detection_tokens,
    merge_candidates,
    union_placement,
)
from p2pmoe.planner.manifest import LayerLoad, NodePlan
from p2pmoe.planner.pipeline import plan
from p2pmoe.sim.replay import make_activation_profiles
from p2pmoe.sim.scenario import appendix_c


# --------------------------------------------------------------------------- #
# 覆盖率与驻留集
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coverage", [0.80, 0.90, 0.95, 0.97, 0.99])
def test_placement_reaches_coverage(coverage: float) -> None:
    """按覆盖率阈值取驻留集：每层实际覆盖率必须 ≥ 阈值，且**去掉任一个**就不足
    —— 即它是达到该覆盖率的最小前缀。"""
    profs = make_activation_profiles(
        ["A", "B"], {"A": 8, "B": 5}, n_layers=6, n_experts=32, coverage=coverage, seed=3
    )
    for u, prof in profs.items():
        plc = build_placement(prof, coverage)
        for l in range(1, prof.n_layers + 1):
            got = plc.achieved_coverage[l - 1]
            assert got >= coverage - 1e-9
            # 最小性：去掉集合里最小的那个，覆盖率就掉到阈值以下
            s = plc.at(l)
            if len(s) > 1:
                m = prof.at(l)
                weakest = min(s, key=lambda e: m[e])
                assert got - m[weakest] < coverage


def test_baseline_miss_is_one_minus_coverage() -> None:
    """II.5：基线 miss 率 = 1 − 覆盖率。滑窗告警线设在它之上。"""
    profs = make_activation_profiles(
        ["A"], {"A": 8}, n_layers=10, n_experts=64, coverage=0.97, seed=1
    )
    plc = build_placement(profs["A"], 0.97)
    assert plc.baseline_miss(1, 10) == pytest.approx(0.03, abs=5e-3)
    # 对角线：绑对了池，miss 就是基线
    assert detectability(profs["A"], plc, 1, 10) == pytest.approx(
        plc.baseline_miss(1, 10), abs=1e-9
    )


def test_sizes_are_heterogeneous_across_layers() -> None:
    """文档 I.1.1：规模 n_{u,l} 异构。逐层独立取阈值，天然如此。"""
    profs = make_activation_profiles(
        ["A"], {"A": 8}, n_layers=32, n_experts=64, coverage=0.97, seed=5
    )
    sizes = build_placement(profs["A"], 0.97).sizes()
    assert len(set(sizes)) > 1, "逐层规模应当异构"


# --------------------------------------------------------------------------- #
# 并集支配性（I.1.1 / III.5.4 用到）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shared", [0, 1, 3, 6])
def test_union_dominates_every_task_layerwise(shared: int) -> None:
    """mem_F(l) ≥ max_u mem_u(l) 逐层成立 —— 因为 ∪_u S_{u,l} ⊇ S_{u,l}。
    这是构造性的，不需要任何额外假设。"""
    profs = make_activation_profiles(
        ["X", "Y", "Z"], {"X": 8, "Y": 6, "Z": 7},
        n_layers=12, n_experts=64, coverage=0.97, shared_core=shared, seed=11,
    )
    plcs = {u: build_placement(p, 0.97) for u, p in profs.items()}
    uni = union_placement(list(plcs.values()))
    for l in range(1, 13):
        for u, p in plcs.items():
            assert p.at(l) <= uni.at(l)
            assert uni.size_at(l) >= p.size_at(l)
        # 并集不超过基数之和，等号当且仅当两两不交
        assert uni.size_at(l) <= sum(p.size_at(l) for p in plcs.values())


def test_union_size_beats_naive_sum_when_tasks_overlap() -> None:
    """基数相加会高估并集 —— 这正是「只有基数」做不了前段内存的原因。"""
    profs = make_activation_profiles(
        ["X", "Y", "Z"], {"X": 8, "Y": 6, "Z": 7},
        n_layers=8, n_experts=64, coverage=0.97, shared_core=4, seed=2,
    )
    plcs = {u: build_placement(p, 0.97) for u, p in profs.items()}
    uni = union_placement(list(plcs.values()))
    naive = sum(p.size_at(1) for p in plcs.values())
    assert uni.size_at(1) < naive


# --------------------------------------------------------------------------- #
# III.7.3 / III.7.4
# --------------------------------------------------------------------------- #
def test_q_is_monotone_in_overlap() -> None:
    """重叠越多，误绑越难检出 —— q 应随 shared_core 单调下降。

    这条同时给出了文档 C.0 与 C.5 的矛盾的量化形式：C.0 说并集 20（≈ 不重叠），
    C.5 说误绑 miss 率 19%。二者不能同时成立。
    """
    qs = []
    for shared in range(0, 7):
        profs = make_activation_profiles(
            ["X", "Y"], {"X": 8, "Y": 6},
            n_layers=16, n_experts=64, coverage=0.97, shared_core=shared, seed=7,
        )
        plcs = {u: build_placement(p, 0.97) for u, p in profs.items()}
        qs.append(detectability(profs["X"], plcs["Y"], 1, 16))
    assert all(a >= b - 1e-9 for a, b in zip(qs, qs[1:])), f"q 应随重叠单调下降: {qs}"
    assert qs[0] > 0.9      # 完全不重叠 ⇒ 几乎必然 miss
    assert qs[-1] < 0.2     # 高度重叠 ⇒ 难检出


def test_merge_signal_fires_only_on_mutual_low_q() -> None:
    """推论 III.7.4：q(u,û) 与 q(û,u) **均**小才是合并信号。"""
    low = make_activation_profiles(
        ["A", "B"], {"A": 8, "B": 8},
        n_layers=10, n_experts=32, coverage=0.97, shared_core=8, seed=4,
    )
    low_p = {u: build_placement(p, 0.97) for u, p in low.items()}
    got = merge_candidates(low, low_p, 1, 10, q_threshold=0.10, expert_gb=0.27)
    assert got and got[0].worst_q <= 0.10
    assert got[0].jaccard > 0.5

    high = make_activation_profiles(
        ["A", "B"], {"A": 8, "B": 8},
        n_layers=10, n_experts=64, coverage=0.97, shared_core=0, seed=4,
    )
    high_p = {u: build_placement(p, 0.97) for u, p in high.items()}
    assert not merge_candidates(high, high_p, 1, 10, q_threshold=0.10)


def test_expected_detection_tokens() -> None:
    assert expected_detection_tokens(0.5) == pytest.approx(2.0)
    assert expected_detection_tokens(0.0) == float("inf")


# --------------------------------------------------------------------------- #
# 部署清单
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built():
    st = appendix_c(seed=7, with_experts=True)
    res = plan(st.nodes, st.model, st.tasks, st.union_experts, st.net, st.cfg, st.p_curve)
    return st, res


def test_manifest_is_produced_and_clean(built) -> None:
    st, res = built
    man = res.manifest
    assert man is not None, "带专家身份时必须产出部署清单"
    assert man.ok, man.violations


def test_manifest_covers_every_layer_exactly_once(built) -> None:
    """每条段的层区间连续、完整、无重叠 —— 逐段重建覆盖集验证。"""
    st, res = built
    for sid, info in res.manifest.segments.items():
        covered: list[int] = []
        for lo, hi in info["splits"]:
            covered += list(range(lo, hi + 1))
        assert covered == sorted(covered), f"{sid} 层序不单调"
        assert len(covered) == len(set(covered)), f"{sid} 层有重叠"
        want = (range(1, res.l0 + 1) if info["role"].startswith("front")
                else range(res.l0 + 1, st.model.n_layers + 1))
        assert covered == list(want), f"{sid} 覆盖不完整"


def test_manifest_expert_sets_match_placements(built) -> None:
    """后段每层装的就是该 task 的驻留集；前段装的是并集。"""
    st, res = built
    for p in res.manifest.nodes:
        for ll in p.layers:
            if p.role.startswith("back:"):
                u = p.role.split(":", 1)[1]
                assert set(ll.experts) == st.placements[u].at(ll.layer)
            elif p.role.startswith("front"):
                assert set(ll.experts) == st.union_experts.at(ll.layer)


def test_manifest_memory_fits_every_node(built) -> None:
    st, res = built
    cap = {n.id: n.usable_gb for n in st.nodes}
    for p in res.manifest.nodes:
        assert p.total_gb <= cap[p.node] + 1e-6


def test_combination_matrix_is_complete(built) -> None:
    """「前后 cluster 任意组合」的形式化：每条前段 × 每条同 task 后段都在矩阵里，
    且每一对的正向接口都落在公共带内（命题 III.8.2）。"""
    st, res = built
    man = res.manifest
    n_front = len(res.fronts_final)
    n_back = res.n_back_total
    assert len(man.pairings) == n_front * n_back
    cm = man.combination_matrix()
    assert len(cm) == n_front
    assert all(len(v) == n_back for v in cm.values())
    lo, hi = man.band
    for p in man.pairings:
        assert lo - 1e-6 <= p.w_p50 <= hi + 1e-6
        assert p.w_jitter <= st.cfg.j_cap_ms + 1e-6
        assert p.d_loop_jitter <= st.cfg.j_cap_ms + 1e-6


def test_manifest_json_roundtrip(built) -> None:
    import json

    st, res = built
    d = json.loads(res.manifest.to_json())
    assert d["l0"] == res.l0
    assert len(d["nodes"]) == len(res.manifest.nodes)
    assert len(d["pairings"]) == len(res.manifest.pairings)
    n0 = d["nodes"][0]
    assert n0["layers"] and isinstance(n0["layers"][0]["experts"], list)


# --------------------------------------------------------------------------- #
# 校验器本身要能抓到违规（否则「校验通过」没有意义）
# --------------------------------------------------------------------------- #
def test_validator_catches_memory_overflow(built) -> None:
    from p2pmoe.planner.manifest import _validate

    st, res = built
    man = res.manifest
    victim = man.nodes[0]
    fat = replace(
        victim,
        layers=victim.layers + (LayerLoad(layer=999, experts=tuple(range(64)),
                                          weight_gb=999.0, kv_gb=0.0),),
    )
    hacked = [fat] + list(man.nodes[1:])
    bad = _validate(
        replace_nodes(man, hacked), st.model, {n.id: n for n in st.nodes},
        st.union_experts, st.placements, st.cfg, res.w_cap,
        n_fronts=len(res.fronts_final),
        back_counts={u: len(v) for u, v in res.backs.items()},
    )
    assert any(v.startswith("[内存]") for v in bad), bad


def test_validator_catches_exclusivity_violation(built) -> None:
    from p2pmoe.planner.manifest import _validate

    st, res = built
    man = res.manifest
    dup = replace(man.nodes[1], node=man.nodes[0].node)
    hacked = [man.nodes[0], dup] + list(man.nodes[2:])
    bad = _validate(
        replace_nodes(man, hacked), st.model, {n.id: n for n in st.nodes},
        st.union_experts, st.placements, st.cfg, res.w_cap,
        n_fronts=len(res.fronts_final),
        back_counts={u: len(v) for u, v in res.backs.items()},
    )
    assert any(v.startswith("[排他]") for v in bad), bad


def test_validator_catches_incomplete_matrix(built) -> None:
    from p2pmoe.planner.manifest import _validate

    st, res = built
    man = res.manifest
    trimmed = replace_pairings(man, man.pairings[:-1])
    bad = _validate(
        trimmed, st.model, {n.id: n for n in st.nodes},
        st.union_experts, st.placements, st.cfg, res.w_cap,
        n_fronts=len(res.fronts_final),
        back_counts={u: len(v) for u, v in res.backs.items()},
    )
    assert any(v.startswith("[组合矩阵]") for v in bad), bad


def replace_nodes(man, nodes):
    import copy

    m = copy.copy(man)
    m.nodes = nodes
    return m


def replace_pairings(man, pairings):
    import copy

    m = copy.copy(man)
    m.pairings = pairings
    return m
