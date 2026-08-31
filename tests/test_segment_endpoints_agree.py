"""段的头/尾在两处说法必须一致。

「谁是这条段的尾」有**两个来源**：

    · 段记录里的 `tail` 字段        —— 协调器按它发 bind
    · 逐节点计划里的 `is_tail`      —— 节点按它决定跑不跑 `_front_tail`

构造时它们必然相等（`Segment.tail` 就是 `nodes[-1]`），但手写清单、改过的 JSON、
别的翻译路径都可能打破这个巧合。而打破之后的症状极其难查：

    bind 发给一台没跑过这条请求的节点 → 它报「没有可发的 L₀ 输出」
    真正的尾段永远等不到 bind        → 请求挂到 300 秒超时
    同池后续请求跟着排队饿死          → 看起来像拥塞

三条线索指向三个不同的方向，没有一条指向「清单自相矛盾」。所以这个检查放在
协调器启动时，让它**在第一条请求之前**就炸。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.runtime.coordinator import Coordinator


def _plan() -> dict:
    def seg(sid, role, task, nodes, a, b):
        return {sid: {"role": role, "task": task, "nodes": nodes,
                      "splits": [[a, b]], "head": nodes[0], "tail": nodes[-1],
                      "hops": len(nodes) - 1, "compute_ms": 1.0,
                      "hop_ms": 0.0, "delay_ms": 1.0}}

    def np_(node, role, sid, i, n, a, b):
        return {"node": node, "role": role, "segment": sid, "position": i,
                "is_head": i == 0, "is_tail": i == n - 1,
                "layer_range": [a, b], "weight_gb": 1.0, "kv_gb": 0.0,
                "total_gb": 1.0,
                "layers": [{"layer": l, "experts": [0], "weight_gb": .5,
                            "kv_gb": 0.0} for l in range(a, b + 1)]}

    return {"l0": 2, "model": {},
            "segments": {**seg("F0", "front", None, ["nf1", "nf2"], 1, 2),
                         **seg("B0", "back:u", "u", ["nb1"], 3, 4)},
            "nodes": [np_("nf1", "front", "F0", 0, 2, 1, 1),
                      np_("nf2", "front", "F0", 1, 2, 2, 2),
                      np_("nb1", "back:u", "B0", 0, 1, 3, 4)]}


def _coord(plan: dict) -> Coordinator:
    return Coordinator(DeploymentManifest.from_dict(plan),
                       baselines={n["node"]: 1.0 for n in plan["nodes"]},
                       priors={"u": 1.0})


def test_a_consistent_plan_starts_fine() -> None:
    _coord(_plan())


def test_bind_goes_to_the_declared_tail() -> None:
    """协调器发 bind 的那台，必须就是段记录里的 tail。"""
    c = _coord(_plan())
    assert c.seg_tail["F0"] == "nf2"


def test_a_tail_field_that_contradicts_is_tail_is_rejected(  ) -> None:
    """**这条是重点。** 段记录说 nf1 是尾，而 nf1 的计划说它不是。"""
    plan = copy.deepcopy(_plan())
    plan["segments"]["F0"]["tail"] = "nf1"
    with pytest.raises(ValueError, match="自相矛盾"):
        _coord(plan)


def test_a_head_field_that_contradicts_is_head_is_rejected() -> None:
    plan = copy.deepcopy(_plan())
    plan["segments"]["F0"]["head"] = "nf2"
    with pytest.raises(ValueError, match="自相矛盾"):
        _coord(plan)


def test_node_order_that_contradicts_the_tail_field_is_rejected() -> None:
    """`nodes` 的顺序也是一个来源 —— 有代码路径按 nodes[-1] 推断。"""
    plan = copy.deepcopy(_plan())
    plan["segments"]["F0"]["nodes"] = ["nf2", "nf1"]
    with pytest.raises(ValueError, match="自相矛盾"):
        _coord(plan)


def test_the_error_names_the_segment_and_both_claims() -> None:
    """报错要能直接改 —— 说清是哪条段、两边各说是谁。"""
    plan = copy.deepcopy(_plan())
    plan["segments"]["F0"]["tail"] = "nf1"
    with pytest.raises(ValueError) as e:
        _coord(plan)
    msg = str(e.value)
    assert "F0" in msg and "nf1" in msg and "nf2" in msg


def test_the_check_runs_before_any_request() -> None:
    """拖到第一条请求才发现的话，代价是 300 秒超时加一串误导性线索。"""
    plan = copy.deepcopy(_plan())
    plan["segments"]["F0"]["tail"] = "nf1"
    with pytest.raises(ValueError):
        _coord(plan)          # 构造期就该炸，还没提交过任何请求


def test_the_real_plan_is_consistent() -> None:
    """仓库里那份真实清单自己要过这一关。"""
    import json

    root = Path(__file__).resolve().parent.parent
    f = root / "task" / "plan_deploy.json"
    if not f.exists():
        pytest.skip("没有 plan_deploy.json")
    m = DeploymentManifest.from_json(f.read_text(encoding="utf-8"))
    for p in m.nodes:
        info = m.segments[p.segment]
        if p.is_tail:
            assert info["tail"] == p.node, f"{p.segment} 的尾对不上"
        if p.is_head:
            assert info["head"] == p.node, f"{p.segment} 的头对不上"
        assert info["nodes"][-1] == info["tail"]
        assert info["nodes"][0] == info["head"]
