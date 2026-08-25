"""通信层：消息编解码、保温长连接、延迟抖动注入。

三件事：

1. **消息格式**：长度前缀的 JSON 头 + 可选的二进制 payload（hidden state 直接走
   raw float32，不进 JSON）。prefill 传整段 hidden state（真实系统里是 MB 级），
   decode 每 token 传一个 hidden state（KB 级）—— 两者在这里走同一条通路，
   只是 payload 大小差两个数量级，正好对应文档 I.1.1 的「带宽 vs 延迟」之分。

2. **保温长连接**（II.6）：每条有向边一个常驻 socket，带锁复用，断了自动重连。
   分散环境下每次重建 TCP 要付一个 RTT（数十毫秒），而 decode 每 token 都要走
   两个接口 —— 不保温的话光握手就把预算吃光了。

3. **延迟抖动注入**：本地 socket 的真实延迟是微秒级，跑不出分散环境的行为。
   发送前按规划期实测到的 (p50, jitter) 采一个样本并 sleep，让在线时序与
   规划时的假设一致。这样才谈得上「验证协议在分散环境下成立」。
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from math import log
from typing import Mapping

import numpy as np

__all__ = ["LinkTable", "send_msg", "recv_msg", "rpc", "Peer", "PeerPool", "Addr"]

_HDR = struct.Struct("!II")
_LN2, _LN20 = log(2.0), log(20.0)

Addr = tuple[str, int]


# --------------------------------------------------------------------------- #
@dataclass
class LinkTable:
    """逐有向对的 (p50, jitter)，来自规划期的实测。"""

    p50: dict[tuple[str, str], float] = field(default_factory=dict)
    jitter: dict[tuple[str, str], float] = field(default_factory=dict)
    scale: float = 1.0
    """整体缩放。demo 里常设 <1 让 24 个进程跑得快些；设 1.0 即真实量级。"""

    def sample_ms(self, a: str, b: str, rng: np.random.Generator) -> float:
        """采一次单程延迟。分布与 sim/network.py 一致：floor + Exp(scale)，
        参数由 (p50, p95−p50) 反解，故中位与尾部都对得上。"""
        if a == b:
            return 0.0
        m = self.p50.get((a, b), 0.0)
        j = self.jitter.get((a, b), 0.0)
        if m <= 0:
            return 0.0
        s = max(j / (_LN20 - _LN2), 1e-6)
        return max(0.0, (m - s * _LN2 + rng.exponential(s))) * self.scale

    def to_dict(self) -> dict:
        return {
            "p50": {f"{a}>{b}": v for (a, b), v in self.p50.items()},
            "jitter": {f"{a}>{b}": v for (a, b), v in self.jitter.items()},
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "LinkTable":
        def un(m):
            out = {}
            for k, v in m.items():
                a, b = k.split(">")
                out[(a, b)] = float(v)
            return out

        return cls(p50=un(d["p50"]), jitter=un(d["jitter"]), scale=float(d.get("scale", 1.0)))


# --------------------------------------------------------------------------- #
def send_msg(sock: socket.socket, header: dict, arr: np.ndarray | None = None) -> None:
    if arr is not None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        header = {**header, "_shape": list(arr.shape)}
        payload = arr.tobytes()
    else:
        payload = b""
    hb = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(_HDR.pack(len(hb), len(payload)) + hb + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("对端关闭")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> tuple[dict, np.ndarray | None]:
    hl, pl = _HDR.unpack(_recv_exact(sock, _HDR.size))
    header = json.loads(_recv_exact(sock, hl))
    arr = None
    if pl:
        raw = _recv_exact(sock, pl)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(header["_shape"]).copy()
    return header, arr


# --------------------------------------------------------------------------- #
def rpc(addr: Addr, header: dict, *, timeout: float = 30.0) -> dict:
    """一次性请求-响应。用于控制面：采集能力、下发清单、驱动探测。

    数据面是 fire-and-forget 的（环自己转），只有控制面需要等回包。
    """
    s = socket.create_connection(addr, timeout=timeout)
    try:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_msg(s, header)
        reply, _ = recv_msg(s)
        return reply
    finally:
        try:
            s.close()
        except OSError:
            pass


class Peer:
    """到某个对端的保温长连接。线程安全，断开自动重连。"""

    def __init__(self, node_id: str, addr: Addr, *, retries: int = 40):
        self.node_id = node_id
        self.addr = addr
        self.retries = retries
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.bytes_sent = 0
        self.msgs_sent = 0

    def _connect(self) -> socket.socket:
        last: Exception | None = None
        for _ in range(self.retries):
            try:
                s = socket.create_connection(self.addr, timeout=10.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError as e:  # 对端还没起来
                last = e
                time.sleep(0.05)
        raise ConnectionError(f"连不上 {self.node_id}@{self.addr}: {last}")

    def send(self, header: dict, arr: np.ndarray | None = None) -> None:
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._sock is None:
                        self._sock = self._connect()
                    send_msg(self._sock, header, arr)
                    self.msgs_sent += 1
                    self.bytes_sent += 0 if arr is None else arr.nbytes
                    return
                except (OSError, ConnectionError):
                    try:
                        if self._sock:
                            self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
                    if attempt:
                        raise

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None


class PeerPool:
    """本节点持有的全部保温连接 + 发送前的延迟注入。"""

    def __init__(
        self,
        me: str,
        links: LinkTable,
        seed: int = 0,
        egress_ms: float = 0.0,
        egress_jitter_ms: float = 0.0,
    ):
        self.me = me
        self.links = links
        self.egress_ms = egress_ms
        self.egress_jitter_ms = egress_jitter_ms
        self._peers: dict[str, Peer] = {}
        self._addr: dict[str, Addr] = {}
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(seed)
        self.wait_ms = 0.0
        """累计注入的延迟 —— 用来核对在线时序是否与规划一致。"""

    def register(self, node_id: str, addr: Addr) -> None:
        with self._lock:
            self._addr[node_id] = tuple(addr)

    def warm(self, node_ids) -> None:
        """预热：提前把连接建起来（II.4 Step 6 的「保温网格」）。"""
        for n in node_ids:
            self._peer(n).send({"type": "ping"})

    def _peer(self, node_id: str) -> Peer:
        with self._lock:
            p = self._peers.get(node_id)
            if p is None:
                if node_id not in self._addr:
                    raise KeyError(f"未注册的对端 {node_id}")
                p = Peer(node_id, self._addr[node_id])
                self._peers[node_id] = p
            return p

    def egress_delay(self) -> float:
        """本节点的**出口接入段**延迟（II.3.1 的加法结构里那一项）。

        真机部署不用它（网络自己会给）。它存在是为了让「在一台机器上演练多机
        部署」有意义：本地 socket 是几十微秒，跑不出分散环境的行为。给每个
        agent 配一个自己的出口延迟后，逐对 RTT ≈ access(a) + access(b)，
        与文档的加法结构一致 —— 而且**探测量到的就是它**，不是另开一套账。
        """
        if self.egress_ms <= 0:
            return 0.0
        s = max(self.egress_jitter_ms / 2.303, 1e-6)
        ms = max(0.0, self.egress_ms - s * _LN2 + self._rng.exponential(s))
        time.sleep(ms / 1000.0)
        return ms

    def send(
        self, node_id: str, header: dict, arr: np.ndarray | None = None, *, delay: bool = True
    ) -> float:
        """发一条消息。delay=False 用于控制面（与协调器的通信不占数据面预算）。"""
        ms = self.egress_delay()
        ms += self.links.sample_ms(self.me, node_id, self._rng) if delay else 0.0
        if ms > 0:
            time.sleep(ms / 1000.0)
            self.wait_ms += ms
        self._peer(node_id).send({**header, "_from": self.me, "_link_ms": round(ms, 3)}, arr)
        return ms

    def close(self) -> None:
        for p in self._peers.values():
            p.close()
        self._peers.clear()

    def stats(self) -> dict:
        return {
            "peers": len(self._peers),
            "msgs": sum(p.msgs_sent for p in self._peers.values()),
            "payload_bytes": sum(p.bytes_sent for p in self._peers.values()),
            "wait_ms": round(self.wait_ms, 1),
        }
