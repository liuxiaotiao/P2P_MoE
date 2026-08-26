"""某个 task 一条后段都没有时，请求必须**立刻失败**，而不是排队等到超时。

真发生过：规划器给 Y 分到 0 条通道（离散配平的正常结果），于是每条被识别成
Y 的请求都排进一个空队列，300 秒后报「超时」。事件日志只说「池无空闲后段」——
读起来像拥塞，实际是这个池压根不存在。诊断跑偏比失败本身更费时间。

分清两件事：
    容量 > 0，空闲 = 0  → 都忙着，等就是了（有界等待，II.5）
    容量 = 0            → 等的是一个永远不会到的事件
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.runtime.coordinator import Coordinator


def _manifest(back_tasks: list[str], n_fronts: int = 2) -> DeploymentManifest:
    """每个 task 一条后段；back_tasks 里没出现的 task 就是容量 0。"""
    nodes, segs = [], {}
    for i in range(n_fronts):
        sid = f"F{i}"
        segs[sid] = {"role": "front", "task": None, "nodes": [f"nf{i}"],
                     "splits": [[1, 2]], "head": f"nf{i}", "tail": f"nf{i}",
                     "hops": 0, "compute_ms": 1.0, "hop_ms": 0.0, "delay_ms": 1.0}
        nodes.append({"node": f"nf{i}", "role": "front", "segment": sid,
                      "position": 0, "is_head": True, "is_tail": True,
                      "layer_range": [1, 2], "weight_gb": 1.0, "kv_gb": 0.0,
                      "total_gb": 1.0,
                      "layers": [{"layer": l, "experts": [0], "weight_gb": .5,
                                  "kv_gb": 0.0} for l in (1, 2)]})
    for i, u in enumerate(back_tasks):
        sid = f"B{u}{i}"
        segs[sid] = {"role": f"back:{u}", "task": u, "nodes": [f"nb{i}"],
                     "splits": [[3, 4]], "head": f"nb{i}", "tail": f"nb{i}",
                     "hops": 0, "compute_ms": 1.0, "hop_ms": 0.0, "delay_ms": 1.0}
        nodes.append({"node": f"nb{i}", "role": f"back:{u}", "segment": sid,
                      "position": 0, "is_head": True, "is_tail": True,
                      "layer_range": [3, 4], "weight_gb": 1.0, "kv_gb": 0.0,
                      "total_gb": 1.0,
                      "layers": [{"layer": l, "experts": [0], "weight_gb": .5,
                                  "kv_gb": 0.0} for l in (3, 4)]})
    return DeploymentManifest.from_dict(
        {"l0": 2, "model": {}, "segments": segs, "nodes": nodes})


def _coord(back_tasks: list[str]) -> Coordinator:
    return Coordinator(_manifest(back_tasks),
                       baselines={f"nf{i}": 1.0 for i in range(2)}
                                 | {f"nb{i}": 1.0 for i in range(len(back_tasks))},
                       priors={"X": .5, "Y": .3, "Z": .2})


# --------------------------------------------------------------------------- #
# 1. 容量与空闲是两回事
# --------------------------------------------------------------------------- #
def test_capacity_records_pools_that_exist() -> None:
    c = _coord(["X", "Y"])
    assert c.back_capacity == {"X": 1, "Y": 1}


def test_a_task_with_no_segment_has_no_pool_at_all() -> None:
    """Z 没建过后段 —— 它在 back_capacity 里就不该出现，更不该是「空队列」。"""
    c = _coord(["X", "Y"])
    assert c.back_capacity.get("Z", 0) == 0


def test_capacity_does_not_move_when_a_channel_is_taken() -> None:
    """容量是「建了几条」，不随占用变。这正是它能区分两种空的原因。"""
    c = _coord(["X"])
    c.free_backs["X"].popleft()
    assert not c.free_backs["X"]        # 空闲 0
    assert c.back_capacity["X"] == 1    # 容量仍是 1 —— 等是有意义的


# --------------------------------------------------------------------------- #
# 2. 容量 0 立刻失败
# --------------------------------------------------------------------------- #
def _rec(c: Coordinator, req: str = "r0"):
    from p2pmoe.runtime.coordinator import RequestRecord
    import time
    r = RequestRecord(req=req, true_task="Z", task="Z")
    r.t0 = time.perf_counter()
    c.records[req] = r
    return r


def test_binding_to_an_absent_pool_fails_immediately() -> None:
    c = _coord(["X", "Y"])
    r = _rec(c)
    c._bind(r, "Z", rebind=False)
    assert r.done.is_set(), "必须立刻结束，而不是排队"
    assert r.stop_reason == "no_channel"


def test_the_failure_says_it_is_not_congestion() -> None:
    """报错要指向真正的原因。「无空闲」会把人引向扩容/调并发，那是错的方向。"""
    c = _coord(["X", "Y"])
    r = _rec(c)
    c._bind(r, "Z", rebind=False)
    why = "\n".join(c.errors)
    assert "容量 0" in why
    assert "X" in why and "Y" in why, "要告诉人现在有哪些池"
    joined = "\n".join(r.events)
    assert "一条后段都没有" in joined


def test_it_does_not_enter_the_waiting_queue() -> None:
    """进了队列就等于把它交给一个永远不会被触发的唤醒路径。"""
    c = _coord(["X", "Y"])
    c._bind(_rec(c), "Z", rebind=False)
    assert not c._await_back.get("Z"), "不该排进 Z 的等待队列"


def test_it_does_not_hand_back_a_channel_it_never_held() -> None:
    """失败路径**不能**走归还 —— 这条请求从没拿到过通道，
    归还会凭空给池子加一条，后面的请求就会绑到一个不存在的段。"""
    c = _coord(["X", "Y"])
    before = {u: len(q) for u, q in c.free_backs.items()}
    c._bind(_rec(c), "Z", rebind=False)
    assert {u: len(q) for u, q in c.free_backs.items()} == before


def test_failing_twice_is_idempotent() -> None:
    c = _coord(["X"])
    r = _rec(c)
    c._bind(r, "Z", rebind=False)
    n = len(c.errors)
    c._bind(r, "Z", rebind=False)
    assert len(c.errors) == n, "已判死的请求不该再记一次错"


# --------------------------------------------------------------------------- #
# 3. 容量 > 0 时照旧排队 —— 别把有界等待也改坏了
# --------------------------------------------------------------------------- #
def test_a_busy_but_existing_pool_still_queues() -> None:
    c = _coord(["X"])
    c.free_backs["X"].popleft()          # 唯一一条被占
    r = _rec(c, "r1")
    c._bind(r, "X", rebind=False)
    assert not r.done.is_set(), "容量 > 0 时该等，不该判死"
    assert c._await_back["X"] and c._await_back["X"][0] is r
    assert "全忙" in "\n".join(r.events)
