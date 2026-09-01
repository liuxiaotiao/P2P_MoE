"""节点处理请求时抛的异常，必须**回到那条连接上**。

原来只往协调器的上报通道发一份。可下发清单的那一刻上报通道未必建好 ——
于是 traceback 彻底消失，调用方阻塞到超时，拿到的是一句
`TimeoutError: timed out`，里面什么都没有。

一次 120 秒的静默换来零信息。而真正的原因就在节点的 traceback 里。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.control import NodeError, _rpc

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def agent():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    pr = subprocess.Popen(
        [sys.executable, "-m", "p2pmoe.deploy.agent", "--id", "NX",
         "--bind", f"127.0.0.1:{port}", "--mem-mb", "4000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    for _ in range(60):
        try:
            _rpc("NX", ("127.0.0.1", port), {"type": "capabilities"}, timeout=2)
            break
        except Exception:
            time.sleep(0.25)
    yield ("127.0.0.1", port)
    pr.terminate()
    pr.wait(timeout=10)


def test_a_failing_request_raises_instead_of_hanging(agent) -> None:
    """**这条是重点。** 快速失败，而不是等满超时。"""
    t0 = time.time()
    with pytest.raises(NodeError):
        _rpc("NX", agent, {"type": "configure", "config": {"node_id": "NX"}},
             timeout=30)
    assert time.time() - t0 < 5, "还是在等超时"


def test_the_error_carries_the_node_traceback(agent) -> None:
    """堆栈是唯一能直接定位的东西 —— 没有它就只能猜。"""
    with pytest.raises(NodeError) as e:
        _rpc("NX", agent, {"type": "configure", "config": {"node_id": "NX"}},
             timeout=30)
    msg = str(e.value)
    assert "NX" in msg
    assert "Traceback" in msg
    assert "node.py" in msg


def test_a_healthy_request_still_returns_normally(agent) -> None:
    r = _rpc("NX", agent, {"type": "capabilities"}, timeout=10)
    assert r["node"] == "NX"


def test_the_agent_survives_a_failed_request(agent) -> None:
    """一次失败不该把 agent 带走 —— 后面还有 14 台要配。"""
    with pytest.raises(NodeError):
        _rpc("NX", agent, {"type": "configure", "config": {"node_id": "NX"}},
             timeout=30)
    assert _rpc("NX", agent, {"type": "capabilities"}, timeout=10)["node"] == "NX"


def test_only_request_response_types_get_an_error_reply() -> None:
    """单向消息（seg_in / hop / loop / bind）不能回 —— 往那些连接上多发一条
    会把协议搞乱，它们的错误本来就走上报通道。"""
    from p2pmoe.runtime.node import _REPLIES

    for one_way in ("seg_in", "hop", "loop", "bind", "release", "drop_kv"):
        assert one_way not in _REPLIES, f"{one_way} 不该回错误"
    for rr in ("configure", "capabilities", "check_model"):
        assert rr in _REPLIES


def test_the_error_is_also_reported_to_the_coordinator() -> None:
    """回给调用方**之外**仍要上报 —— 单向消息只有这一条路。"""
    src = (ROOT / "p2pmoe" / "runtime" / "node.py").read_text(encoding="utf-8")
    i = src.index("self._dispatch(header, arr, conn)")
    blk = src[i:i + 1400]
    assert "self._report" in blk, "不上报了"
    assert "send_msg(conn" in blk, "不回调用方"
    assert blk.index("self._report") < blk.index("send_msg(conn"), \
        "上报要在前 —— 回复失败也不能丢掉上报"
