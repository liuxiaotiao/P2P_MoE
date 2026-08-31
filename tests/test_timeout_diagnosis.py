"""请求超时了，要说清楚**卡在哪台**。

协调器这一侧的事件日志只能说到「派发出去了」为止。派发之后的沉默有好几种
完全不同的原因，而它们在日志里长得一模一样：

    · 段首根本没收到 seg_in
    · 收到了但还没装载完模型（第一条请求常撞上，真模型几十 GB）
    · 装载失败了（错误上报走另一条路，可能还没到）
    · 算到一半卡在某一跳

分辨它们只要一件事：**问节点自己**。要问 `capabilities` 而不是 `stats` ——
`stats` 在配置之前答不了（返回 error），而「压根没配置成功」恰恰是最常见的
那一种。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.control import _span, _where_is_it_stuck
from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.runtime.node import NodeServer


def _manifest() -> DeploymentManifest:
    def seg(role, task, nodes, a, b):
        return {"role": role, "task": task, "nodes": nodes,
                "splits": [[a, b]], "head": nodes[0], "tail": nodes[-1],
                "hops": len(nodes) - 1, "compute_ms": 1., "hop_ms": 0.,
                "delay_ms": 1.}

    def nd(n, role, sid, a, b, head, tail):
        return {"node": n, "role": role, "segment": sid, "position": 0,
                "is_head": head, "is_tail": tail, "layer_range": [a, b],
                "weight_gb": .1, "kv_gb": 0., "total_gb": .1,
                "layers": [{"layer": l, "experts": [0], "weight_gb": .05,
                            "kv_gb": 0.} for l in range(a, b + 1)]}

    return DeploymentManifest.from_dict({
        "l0": 2, "model": {}, "segments": {
            "F0": seg("front", None, ["nf"], 1, 2),
            "BZ0": seg("back:Z", "Z", ["nb1", "nb2"], 3, 4)},
        "nodes": [nd("nf", "front", "F0", 1, 2, True, True),
                  nd("nb1", "back:Z", "BZ0", 3, 3, True, False),
                  nd("nb2", "back:Z", "BZ0", 4, 4, False, True)]})


class _Rec:
    def __init__(self, front="F0", back="BZ0"):
        self.front, self.back, self.req = front, back, "req0"


class _Coord:
    def __init__(self, man):
        self.man = man


@pytest.fixture
def live():
    """一台真 agent —— 未配置。capabilities 答得出，stats 答不出。"""
    srv = NodeServer("nb1", host="127.0.0.1", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    yield srv
    try:
        srv.sock.close()
    except OSError:
        pass


def _dead_addr() -> tuple[str, int]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    a = s.getsockname()
    s.close()
    return a


def test_it_reports_every_node_on_the_path(live, caplog) -> None:
    man = _manifest()
    addrs = {"nf": _dead_addr(), "nb1": ("127.0.0.1", live.port),
             "nb2": _dead_addr()}
    with caplog.at_level("ERROR", logger="p2pmoe.control"):
        _where_is_it_stuck(_Rec(), _Coord(man), addrs, None)
    text = caplog.text
    for nid in ("nf", "nb1", "nb2"):
        assert nid in text, f"{nid} 没被问到"


def test_an_unconfigured_node_is_called_out(live, caplog) -> None:
    """**最常见的那一种。** 节点活着、端口在听，但清单下发没到或装载失败了 ——
    从协调器看只是「沉默」，从节点看是「我还没被配置」。"""
    man = _manifest()
    addrs = {"nb1": ("127.0.0.1", live.port)}
    with caplog.at_level("ERROR", logger="p2pmoe.control"):
        _where_is_it_stuck(_Rec(front=""), _Coord(man), addrs, None)
    assert "还没配置" in caplog.text


def test_an_unreachable_node_is_distinguished(caplog) -> None:
    """「问不到」与「答了但没干活」是两回事 —— 前者是网络，后者是逻辑。"""
    man = _manifest()
    addrs = {"nb1": _dead_addr(), "nb2": _dead_addr()}
    with caplog.at_level("ERROR", logger="p2pmoe.control"):
        _where_is_it_stuck(_Rec(front=""), _Coord(man), addrs, None)
    assert "问不到" in caplog.text


def test_a_node_missing_from_agents_is_named(caplog) -> None:
    """清单里有、`--agents` 里没有 —— 这台永远不会响应，而原因在命令行上。"""
    man = _manifest()
    with caplog.at_level("ERROR", logger="p2pmoe.control"):
        _where_is_it_stuck(_Rec(front=""), _Coord(man), {}, None)
    assert "不在 --agents 里" in caplog.text


def test_it_says_how_to_read_the_output(caplog) -> None:
    """一张表不够 —— 要告诉人从哪个数字看起。"""
    man = _manifest()
    with caplog.at_level("ERROR", logger="p2pmoe.control"):
        _where_is_it_stuck(_Rec(front=""), _Coord(man), {}, None)
    assert "算过 0ms" in caplog.text and "段首" in caplog.text


def test_an_unbound_request_says_so(caplog) -> None:
    """还没绑上任何段就超时 —— 那是排队问题，不是节点问题，别去问节点。"""
    man = _manifest()
    with caplog.at_level("ERROR", logger="p2pmoe.control"):
        _where_is_it_stuck(_Rec(front="", back=""), _Coord(man), {}, None)
    assert "还没绑上" in caplog.text


def test_diagnosis_never_masks_the_original_failure(caplog) -> None:
    """诊断自己炸了不能盖住「请求超时」这件事本身。"""
    class Boom:
        @property
        def man(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _where_is_it_stuck(_Rec(), Boom(), {}, None)

    # control.py 里这一段被 try/except 包着 —— 确认它确实包着
    src = (Path(__file__).resolve().parent.parent
           / "p2pmoe" / "deploy" / "control.py").read_text(encoding="utf-8")
    # 注意别匹配到函数定义那一行 —— 它长得一模一样
    i = src.index("                    _where_is_it_stuck(rec, coord")
    assert "try:" in src[i - 120:i], "调用点没有 try 保护"


def test_layer_span_is_readable() -> None:
    assert _span([3]) == "3"
    assert _span([3, 4, 5]) == "3–5"
    assert _span(None) == "—"
