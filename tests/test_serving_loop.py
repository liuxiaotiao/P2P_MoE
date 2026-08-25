"""服务循环：有界等待、自动配对、跑完回池。

直接驱动 Coordinator 的状态机（假的 PeerPool + 合成上报），不起进程 —— 这一层
要测的是**队列语义**，不是通信。端到端由 test_runtime / test_deploy 覆盖。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.runtime.coordinator import Coordinator


# --------------------------------------------------------------------------- #
@dataclass
class FakePool:
    """记录发出去的消息，不真发。"""

    sent: list[tuple[str, dict]] = field(default_factory=list)

    def send(self, node_id, header, arr=None, *, delay=True):
        self.sent.append((node_id, header))
        return 0.0

    def of_type(self, t: str) -> list[tuple[str, dict]]:
        return [(n, h) for n, h in self.sent if h.get("type") == t]


@dataclass
class FakeNodePlan:
    node: str
    role: str
    segment: str
    position: int = 0


class FakeManifest:
    """2 条前段 + (X:2, Y:1) 后段，每条段一个节点。"""

    def __init__(self, n_front=2, backs=None):
        backs = backs or {"X": 2, "Y": 1}
        self.nodes = [FakeNodePlan(f"f{i}", "front", f"F{i}") for i in range(n_front)]
        for u, n in backs.items():
            for i in range(n):
                self.nodes.append(FakeNodePlan(f"b{u}{i}", f"back:{u}", f"B{u}{i}"))
        self.segments = {
            p.segment: {"head": p.node, "nodes": [p.node]} for p in self.nodes
        }


def make_coord(**kw) -> tuple[Coordinator, FakePool]:
    man = FakeManifest(**kw)
    c = Coordinator(man, baselines={"X": 0.1, "Y": 0.1},
                    priors={"X": 0.6, "Y": 0.4}, alarm_factor=99.0)
    pool = FakePool()
    c.pool = pool                # 不调 start()：不需要真的监听
    c.max_tokens = 2
    return c, pool


def classify(c: Coordinator, req: str, task: str, conf: float = 0.9) -> None:
    c._on({"type": "classified", "req": req, "node": "f0", "task": task,
           "conf": conf, "zone": "commit", "scores": {task: conf},
           "front_stats": {}})


def token(c: Coordinator, req: str, phase: str = "decode") -> None:
    c._on({"type": "token", "req": req, "node": "b", "segment": "B",
           "phase": phase, "token": 1,
           "back_stats": {"hist": [], "ntl": 3, "miss": 0, "mass": 0.0}})


def run_to_completion(c: Coordinator, req: str) -> None:
    """喂够 token 让它走完 _finish。"""
    for _ in range(c.max_tokens + 1):
        if c.records[req].done.is_set():
            break
        token(c, req)


# --------------------------------------------------------------------------- #
def test_pool_is_the_product_of_planning() -> None:
    c, _ = make_coord()
    assert list(c.free_fronts) == ["F0", "F1"]
    assert {u: list(q) for u, q in c.free_backs.items()} == {
        "X": ["BX0", "BX1"], "Y": ["BY0"]
    }


def test_blind_bind_pops_queue_head() -> None:
    """盲绑就是 popleft，不做任何比较（推论 III.3.2）。"""
    c, pool = make_coord()
    r0 = c.submit("r0", [1, 2])
    assert r0.front == "F0"
    r1 = c.submit("r1", [3, 4])
    assert r1.front == "F1"
    assert not c.free_fronts
    assert len(pool.of_type("prefill")) == 2


def test_submit_beyond_capacity_queues_instead_of_failing() -> None:
    """池满不是错误，是有界等待（II.5）。早先的版本这里抛异常。"""
    c, pool = make_coord()
    c.submit("r0", [1])
    c.submit("r1", [2])
    r2 = c.submit("r2", [3])
    assert r2.front == "", "第 3 条应当排队而不是拿到前段"
    assert c.queue_depths()["waiting_front"] == 1
    assert len(pool.of_type("prefill")) == 2, "排队的那条不该发出 prefill"
    assert any("排队" in e for e in r2.events)


def test_queued_request_starts_when_a_front_frees() -> None:
    """前面哪条一完成，队首立刻被接走 —— 不用等整批。"""
    c, pool = make_coord()
    c.submit("r0", [1])
    c.submit("r1", [2])
    r2 = c.submit("r2", [3])

    classify(c, "r0", "X")
    run_to_completion(c, "r0")

    assert c.records["r0"].done.is_set()
    assert r2.front == "F0", "释放的前段应直接交给队首"
    assert c.queue_depths()["waiting_front"] == 0
    assert len(pool.of_type("prefill")) == 3
    assert r2.wait_front_ms >= 0


def test_back_pool_exhaustion_also_queues() -> None:
    """Y 池只有 1 条：第 2 条识别成 Y 的请求要等后段。"""
    c, pool = make_coord()
    c.submit("r0", [1])
    c.submit("r1", [2])
    classify(c, "r0", "Y")
    classify(c, "r1", "Y")

    assert c.records["r0"].back == "BY0"
    assert c.records["r1"].back == "", "第 2 条 Y 请求应当排队等后段"
    assert c.queue_depths()["waiting_back"] == {"Y": 1}
    assert len(pool.of_type("bind")) == 1

    run_to_completion(c, "r0")
    assert c.records["r1"].back == "BY0", "释放的后段应直接交给排队者"
    assert len(pool.of_type("bind")) == 2


def test_segments_return_to_pool_after_completion() -> None:
    """跑完前后段都回池，池深恢复原样。"""
    c, pool = make_coord()
    before = c.queue_depths()
    c.submit("r0", [1])
    classify(c, "r0", "X")
    assert c.queue_depths()["free_fronts"] == 1     # 用掉一条
    run_to_completion(c, "r0")
    assert c.queue_depths() == before, "完成后池子应恢复"
    # 释放消息发给了两段的所有节点
    released = {n for n, _ in pool.of_type("release")}
    assert released == {"f0", "bX0"}


def test_consecutive_requests_get_different_pairs() -> None:
    """跑完回池后，下一条请求重新配 —— 因为池是 FIFO，会轮转到别的段。"""
    c, _ = make_coord()
    seen = []
    for i in range(4):
        req = f"r{i}"
        c.submit(req, [i])
        classify(c, req, "X")
        seen.append((c.records[req].front, c.records[req].back))
        run_to_completion(c, req)
    assert len(set(seen)) > 1, f"四条请求配了同一对段: {seen}"
    # 前段轮转
    assert [f for f, _ in seen] == ["F0", "F1", "F0", "F1"]


def test_pairing_history_is_recorded() -> None:
    c, _ = make_coord()
    for i, u in enumerate(["X", "Y", "X"]):
        req = f"r{i}"
        c.submit(req, [i])
        classify(c, req, u)
        run_to_completion(c, req)
    assert [p[0] for p in c.pairings] == ["r0", "r1", "r2"]
    assert [p[3] for p in c.pairings] == ["X", "Y", "X"]
    assert all(p[1] and p[2] for p in c.pairings)


def test_heavy_burst_all_complete() -> None:
    """一次打入 10 条到只有 2 条前段的池子：全部应当依次完成，无死锁。"""
    c, _ = make_coord()
    recs = [c.submit(f"r{i}", [i]) for i in range(10)]
    assert c.queue_depths()["waiting_front"] == 8

    for _ in range(10):
        active = [r for r in recs if r.front and not r.done.is_set()]
        if not active:
            break
        for r in active:
            if not r.task:
                classify(c, r.req, "X")
            run_to_completion(c, r.req)

    assert all(r.done.is_set() for r in recs), \
        f"未完成: {[r.req for r in recs if not r.done.is_set()]}"
    assert c.queue_depths()["waiting_front"] == 0
    assert c.queue_depths()["free_fronts"] == 2
