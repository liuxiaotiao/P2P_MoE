"""中继：节点之间没有直连时的兜底。

中继要成立靠一条纪律：**握手完就只搬字节，不认识 p2pmoe 的协议**。
所以这里测的是「上层完全分辨不出中间隔了一台机器」——
同一套 `send_msg`/`recv_msg`、同一个 `NodeServer.serve_forever`。
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from p2pmoe.deploy.relay import Relay
from p2pmoe.runtime.wire import (
    Peer,
    PeerPool,
    RelayListener,
    dial_via_relay,
    recv_msg,
    rpc,
    send_msg,
)


@pytest.fixture
def relay():
    r = Relay("127.0.0.1", 0).start()
    yield r
    r.stop()


def echo_server(listener: RelayListener, stop: threading.Event) -> None:
    """一个最小的对端：收什么回什么，外加把 payload 形状报回去。"""
    listener.settimeout(0.2)
    while not stop.is_set():
        try:
            conn, _ = listener.accept()
        except (socket.timeout, OSError):
            continue
        threading.Thread(target=_serve_one, args=(conn, stop), daemon=True).start()


def _serve_one(conn: socket.socket, stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            h, arr = recv_msg(conn)
            send_msg(conn, {"type": "ack", "got": h.get("type"),
                            "shape": None if arr is None else list(arr.shape)}, arr)
    except (OSError, ConnectionError):
        pass


@pytest.fixture
def peer_n5(relay):
    """n5 挂在中继上，等人来接。"""
    lis = RelayListener((relay.host, relay.port), "n5", depth=4)
    stop = threading.Event()
    t = threading.Thread(target=echo_server, args=(lis, stop), daemon=True)
    t.start()
    time.sleep(0.3)
    yield lis
    stop.set()
    lis.close()


# --------------------------------------------------------------------------- #
# 1. 接通
# --------------------------------------------------------------------------- #
def test_a_dialer_reaches_a_parked_node(relay, peer_n5) -> None:
    s = dial_via_relay((relay.host, relay.port), "n1", "n5")
    send_msg(s, {"type": "hello"})
    h, _ = recv_msg(s)
    assert h["got"] == "hello"
    s.close()


def test_dialing_a_node_that_never_showed_up_says_so(relay) -> None:
    """报错要指向真正的原因：对方的 agent 没起，或者没加 --relay。"""
    import p2pmoe.deploy.relay as R

    R._PARK_TIMEOUT = 0.5      # 别等满 30 秒
    with pytest.raises(ConnectionError, match="没有挂在中继上"):
        dial_via_relay((relay.host, relay.port), "n1", "n404")
    R._PARK_TIMEOUT = 30.0


def test_neither_side_needs_an_inbound_port(relay, peer_n5) -> None:
    """**中继存在的全部理由。** 双方都只有出站连接。

    n5 没有 bind 任何端口 —— 它只是往中继开了几条出站连接挂着。
    """
    assert not hasattr(peer_n5, "bind")
    s = dial_via_relay((relay.host, relay.port), "n1", "n5")
    assert s.getpeername()[1] == relay.port      # 拨号方连的也是中继
    s.close()


# --------------------------------------------------------------------------- #
# 2. 上层分辨不出中间隔了一台机器
# --------------------------------------------------------------------------- #
def test_binary_payloads_survive_the_splice(relay, peer_n5) -> None:
    """hidden state 走的是 raw float32，中继必须逐字节原样搬。"""
    s = dial_via_relay((relay.host, relay.port), "n1", "n5")
    arr = np.random.default_rng(0).normal(size=(37, 64)).astype(np.float32)
    send_msg(s, {"type": "seg_in"}, arr)
    h, back = recv_msg(s)
    assert h["shape"] == [37, 64]
    assert np.array_equal(back, arr)
    s.close()


def test_a_big_payload_is_not_truncated(relay, peer_n5) -> None:
    """prefill 的 hidden state 是 MB 级 —— 会跨很多次 recv，拼错就会截断。"""
    s = dial_via_relay((relay.host, relay.port), "n1", "n5")
    arr = np.random.default_rng(1).normal(size=(4096, 512)).astype(np.float32)
    send_msg(s, {"type": "seg_in"}, arr)
    _, back = recv_msg(s)
    assert np.array_equal(back, arr)
    s.close()


def test_many_messages_on_one_connection(relay, peer_n5) -> None:
    """保温长连接：一条 socket 上连续来回，中继不能在中间插入或吞掉边界。"""
    s = dial_via_relay((relay.host, relay.port), "n1", "n5")
    for i in range(50):
        send_msg(s, {"type": f"m{i}"})
        h, _ = recv_msg(s)
        assert h["got"] == f"m{i}"
    s.close()


def test_rpc_works_through_the_relay(relay, peer_n5) -> None:
    """控制面的一问一答也走同一条路 —— 中继按节点 id 找人，不按地址。"""
    r = rpc(("0.0.0.0", 0), {"type": "capabilities"},
            relay=(relay.host, relay.port), me="__coord__", to="n5")
    assert r["got"] == "capabilities"


def test_rpc_via_relay_needs_a_node_id(relay) -> None:
    with pytest.raises(ValueError, match="对端节点 id"):
        rpc(("0.0.0.0", 0), {"type": "x"}, relay=(relay.host, relay.port))


# --------------------------------------------------------------------------- #
# 3. Peer / PeerPool 走中继
# --------------------------------------------------------------------------- #
def test_peer_dials_through_the_relay(relay, peer_n5) -> None:
    p = Peer("n5", ("0.0.0.0", 0), relay=(relay.host, relay.port), me="n1")
    p.send({"type": "ping"})
    assert p.msgs_sent == 1
    p.close()


def test_pool_switches_every_peer_at_once(relay, peer_n5) -> None:
    """一次部署要么全走中继要么全不走，没有一半一半 —— 所以是池级开关。"""
    pool = PeerPool("n1", __import__("p2pmoe.runtime.wire", fromlist=["LinkTable"]).LinkTable())
    pool.use_relay((relay.host, relay.port))
    pool.register("n5", ("0.0.0.0", 0))
    pool.send("n5", {"type": "ping"})
    assert pool.stats()["msgs"] == 1
    pool.close()


def test_an_unreachable_peer_surfaces_the_relay_reason(relay) -> None:
    import p2pmoe.deploy.relay as R

    R._PARK_TIMEOUT = 0.3
    p = Peer("n404", ("0.0.0.0", 0), relay=(relay.host, relay.port), me="n1",
             retries=2)
    with pytest.raises(ConnectionError, match="经中继"):
        p.send({"type": "ping"})
    R._PARK_TIMEOUT = 30.0


# --------------------------------------------------------------------------- #
# 4. 挂起池的补充与回收
# --------------------------------------------------------------------------- #
def test_the_park_pool_refills_after_being_consumed(relay, peer_n5) -> None:
    """一条被接走就补一条 —— 否则接够 depth 次之后就再也拨不通了。"""
    for _ in range(peer_n5.depth + 3):
        s = dial_via_relay((relay.host, relay.port), "n1", "n5")
        send_msg(s, {"type": "ping"})
        recv_msg(s)
        s.close()
    time.sleep(0.3)
    with relay._cv:
        assert len(relay._parked.get("n5", [])) > 0


def test_relay_counts_what_it_moved(relay, peer_n5) -> None:
    """接通次数与字节数是「数据面确实经过它」的直接证据。"""
    s = dial_via_relay((relay.host, relay.port), "n1", "n5")
    send_msg(s, {"type": "x"}, np.zeros((100, 100), dtype=np.float32))
    recv_msg(s)
    s.close()
    time.sleep(0.3)
    assert relay.n_spliced >= 1
    assert relay.bytes_relayed >= 100 * 100 * 4


def test_status_handshake_reports_the_pool(relay, peer_n5) -> None:
    c = socket.create_connection((relay.host, relay.port), timeout=5)
    send_msg(c, {"type": "status"})
    h, _ = recv_msg(c)
    c.close()
    assert h["parked"]["n5"] > 0


def test_an_unknown_handshake_is_refused(relay) -> None:
    c = socket.create_connection((relay.host, relay.port), timeout=5)
    send_msg(c, {"type": "nonsense"})
    h, _ = recv_msg(c)
    c.close()
    assert h["type"] == "error"
