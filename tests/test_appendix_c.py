"""附录 C 算例的回归基线 + 池群鲁棒性扫描。

这里的断言分两类：
  * 与文档**数值一致**的部分（内存形状、L₀、配额、公平比）—— 强断言；
  * 规划结果本身 —— 只断言结构性质（跳数不超下界太多、终审自洽），不冻结
    具体节点分配，因为那取决于模拟网络的随机种子。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.capacity import estimate_capacity_by_tier
from p2pmoe.planner.memory import choose_l0, hops_min, make_back_spec, make_front_spec
from p2pmoe.planner.pipeline import PlanningError, plan
from p2pmoe.sim.scenario import UNION_EXPERTS, appendix_c


# --------------------------------------------------------------------------- #
# 与文档数值一致的部分
# --------------------------------------------------------------------------- #
def test_c0_memory_shapes_match_document() -> None:
    st = appendix_c()
    m = st.model
    f = make_front_spec(m, UNION_EXPERTS, 5)
    assert f.total_gb() == pytest.approx(28.0, abs=0.05)          # 文档：27.7 + 0.34
    got = {t.name: make_back_spec(m, t, 5).total_gb() for t in st.tasks}
    assert got["X"] == pytest.approx(63.6, abs=0.1)
    assert got["Y"] == pytest.approx(49.1, abs=0.1)
    assert got["Z"] == pytest.approx(56.4, abs=0.1)
    weighted = sum(t.lam * got[t.name] for t in st.tasks)
    assert weighted == pytest.approx(57.8, abs=0.2)               # 文档：≈ 57GB


def test_c1_l0_choice_matches_document() -> None:
    """文档 C.1 定 L₀=5（p=0.91，前段仍装进单张 V100-32、仍零跳）。"""
    st = appendix_c()
    best, table = choose_l0(st.model, st.tasks, UNION_EXPERTS, st.nodes, st.p_curve, p_min=0.85)
    assert best.l0 == 5
    assert best.hops_front == 0
    assert best.front_single_node_count == 14      # 4× A40 + 10× V100-32
    by = {c.l0: c for c in table}
    assert by[6].front_single_node_count == 4      # L₀=6 起前段只能上 A40
    assert by[6].n_channels < by[5].n_channels     # 通道数因此塌掉


def test_c1_tier_zeroing_diverges_from_document() -> None:
    """文档 C.1 断言 16GB 档「整档归零」、废料率 40%。本实现算出的结论不同 ——
    这是一处**文档偏严**的地方，测试把它固定下来以免被无声改回去。

    文档的理由是「后段两节点形态的任一位 ✗（最小位 > 23GB）」，隐含了「两个
    节点平分层区间」。但切分不必平分：Y 后段 49.1GB 可以由 A40 承 25 层
    （45.4GB）+ 一台 15GB 节点承剩下 2 层（3.6GB）。故 16GB 档能承担 Y/Z 后段
    形态里的小位，不该整档归零。
    """
    st = appendix_c()
    f = make_front_spec(st.model, UNION_EXPERTS, 5)
    bs = {t.name: make_back_spec(st.model, t, 5) for t in st.tasks}
    cap = estimate_capacity_by_tier(st.nodes, f, bs, st.tasks)
    small = [r for r in cap.reports if r.tier.usable_gb == pytest.approx(15.0)]
    assert small, "应有 15GB 可用的档"
    for r in small:
        assert not r.zeroed, "16GB 档能承担 Y/Z 后段的小位，不应整档归零"
        assert r.serves["front"] is False        # 但确实装不下前段
        assert r.serves["back:X"] is False       # 也装不下 X 后段的任一位
        assert r.serves["back:Y"] is True


def test_c1_quota_and_fairness_match_document() -> None:
    st = appendix_c(seed=7)
    res = plan(st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve)
    assert res.n_work == 4
    assert res.quota == {"X": 2, "Y": 1, "Z": 1}
    assert res.fair["X"] == pytest.approx(1.00)
    assert res.fair["Y"] == pytest.approx(0.833, rel=1e-2)
    assert res.fair["Z"] == pytest.approx(1.25)
    assert min(res.fair, key=res.fair.get) == "Y"


# --------------------------------------------------------------------------- #
# 端到端结构性质
# --------------------------------------------------------------------------- #
def test_end_to_end_baseline_seed7() -> None:
    st = appendix_c(seed=7)
    res = plan(st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve)

    assert res.l0 == 5
    assert res.audit.passed, res.audit.reasons
    assert not res.audit.unserved_tasks
    assert res.audit.worst_rel_spread <= st.cfg.eta

    # 前段全部单节点（零内部跳）—— 分散环境下的最优形态
    assert all(s.hops == 0 for s in res.fronts_final)
    # 每条前段的出口都在公共带内
    assert all(s.tail in set(res.band.exits) for s in res.fronts_final)
    # 网络占绝对大头
    p = min(res.audit.pairs, key=lambda x: x.t50)
    assert (p.w_fwd + p.d_loop) / p.t50 > 0.5


@pytest.mark.parametrize("seed", range(16))
def test_no_segment_exceeds_hop_floor_by_much(seed: int) -> None:
    """质量下限：任何进入方案的段，跳数不得比 III.5.2 的整数下界超出 slack。

    这条同时守住两件事：分散环境下多一跳就是数十毫秒/token；以及规划器不能
    靠「反正建出来了」蒙混过关 —— 建不出好段就该记缺口。
    """
    st = appendix_c(seed=seed)
    try:
        res = plan(st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve)
    except PlanningError:
        return  # 建不出来是合法结论，且日志里已说明出路
    mx = max(n.usable_gb for n in st.nodes)
    for u, segs in res.backs.items():
        spec = make_back_spec(st.model, next(t for t in st.tasks if t.name == u), res.l0)
        floor = hops_min(spec.total_gb(), mx)
        for s in segs:
            assert s.hops <= floor + st.cfg.max_hops_slack + 1, (
                f"seed={seed} {s.label()} 用了 {s.hops} 跳，下界 {floor}"
            )


@pytest.mark.parametrize("seed", range(16))
def test_audit_is_self_consistent(seed: int) -> None:
    """终审报告不能自相矛盾：passed 为真 ⟺ 没有任何 reason。"""
    st = appendix_c(seed=seed)
    try:
        res = plan(st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve)
    except PlanningError as e:
        assert e.log  # 失败必须带日志
        return
    assert res.audit.passed == (not res.audit.reasons)
    # 保留的前段数不超过后段数 + 备胎
    assert len(res.fronts_final) <= res.n_back_total + st.cfg.n_standby
    # 命题 III.8.3(b)
    assert res.trim.spread_kept <= res.trim.spread_all + 1e-9


def test_planning_error_carries_log() -> None:
    """人口凑不够时抛 PlanningError，且日志里必须给出可执行的出路。"""
    st = appendix_c(seed=7)
    st.cfg.eta = 0.01          # 把 W 上限压到不可能
    with pytest.raises(PlanningError) as ei:
        plan(st.nodes, st.model, st.tasks, UNION_EXPERTS, st.net, st.cfg, st.p_curve)
    assert ei.value.log
    assert "域" in str(ei.value) or "条数" in str(ei.value)
