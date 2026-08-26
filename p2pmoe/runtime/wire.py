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
import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from math import log
from typing import Mapping

import numpy as np

__all__ = ["RelayListener", "dial_via_relay", "LinkTable", "send_msg", "recv_msg", "rpc", "Peer", "PeerPool", "Addr"]

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
def _as_wire_array(arr) -> np.ndarray:
    """把 hidden state 归一成 float32 的连续 numpy 数组。

    线上的 payload **恒为 float32**（第〇部分：接口只传字节，不传框架对象）——
    所以 torch 后端的 bf16 张量在这里降到 float32，对端再升回去。这不是精度
    妥协的疏忽，是接口定义：两端可以是不同 dtype、不同框架、不同设备。

    这里用鸭子类型认 torch 张量（`.detach`）而不是 `import torch` —— wire 是
    通信层，「装了 torch 才能通信」说不过去。`.float()` 已经给出 float32，
    不需要引用 torch 的 dtype 常量，于是这条边界一行代码都不用破。
    """
    if hasattr(arr, "detach"):          # torch.Tensor：bf16/fp16/GPU 都要先落地
        arr = arr.detach().to("cpu").float().numpy()
    return np.ascontiguousarray(arr, dtype=np.float32)


def send_msg(sock: socket.socket, header: dict, arr: np.ndarray | None = None) -> None:
    if arr is not None:
        arr = _as_wire_array(arr)
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
def dial_via_relay(relay: Addr, me: str, to: str, *, timeout: float = 35.0) -> socket.socket:
    """经中继要一条到 `to` 的连接。返回的就是一条普通 socket。

    握手完就是裸流 —— 中继不解析消息（见 deploy/relay.py）。所以调用方拿到它
    之后一切照旧：`send_msg` / `recv_msg` 不知道中间隔了一台机器。
    """
    s = socket.create_connection(relay, timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        send_msg(s, {"type": "dial", "from": me, "to": to})
        h, _ = recv_msg(s)
        if not h.get("ok"):
            raise ConnectionError(f"中继接不通 {to}：{h.get('why', h)}")
        s.settimeout(None)
        return s
    except Exception:
        s.close()
        raise


def rpc(addr: Addr, header: dict, *, timeout: float = 30.0,
        relay: Addr | None = None, me: str = "__coord__", to: str | None = None) -> dict:
    """一次性请求-响应。用于控制面：采集能力、下发清单、驱动探测。

    数据面是 fire-and-forget 的（环自己转），只有控制面需要等回包。

    `relay` 给了就经中继拨号，这时 `to`（对端节点 id）必填 —— 中继按 id 找人，
    不按地址。
    """
    if relay is not None:
        if not to:
            raise ValueError("经中继要给对端节点 id（to=）")
        s = dial_via_relay(relay, me, to, timeout=timeout)
        s.settimeout(timeout)
        try:
            send_msg(s, header)
            reply, _ = recv_msg(s)
            return reply
        finally:
            try:
                s.close()
            except OSError:
                pass
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


class RelayListener:
    """替代监听 socket：预先在中继上挂几条连接，等别人来接。

    **接口刻意与 `socket` 对齐**（`accept` / `close` / `settimeout`），这样
    `NodeServer.serve_forever` 那个循环一行都不用改 —— 节点不需要知道自己是在
    监听端口还是挂在中继上。

    预挂多条是为了省一个往返：一条被接走立刻补一条，拨号方来的时候总有现成的。
    """

    def __init__(self, relay: Addr, me: str, *, depth: int = 8):
        self.relay = tuple(relay)
        self.me = me
        self.depth = depth
        self._ready: "queue.Queue[socket.socket]" = queue.Queue()
        self._stop = threading.Event()
        self._timeout: float | None = None
        self._threads: list[threading.Thread] = []
        for _ in range(depth):
            self._spawn()

    def _spawn(self) -> None:
        t = threading.Thread(target=self._park_one, daemon=True)
        t.start()
        self._threads.append(t)

    def _park_one(self) -> None:
        """挂一条连接，等它被接通；接通后交给 accept()，再补一条。"""
        while not self._stop.is_set():
            try:
                s = socket.create_connection(self.relay, timeout=15.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                send_msg(s, {"type": "listen", "node": self.me})
                s.settimeout(None)
                h, _ = recv_msg(s)          # 阻塞到有人拨过来
            except (OSError, ConnectionError):
                if self._stop.is_set():
                    return
                time.sleep(0.5)             # 中继还没起来 / 网络抖 —— 重挂
                continue
            if h.get("type") != "accepted":
                s.close()
                continue
            self._ready.put(s)
            return                          # 这条已交出，由 accept() 补新的

    def accept(self) -> tuple[socket.socket, Addr]:
        try:
            s = self._ready.get(timeout=self._timeout if self._timeout else None)
        except queue.Empty:
            raise socket.timeout("no pending relay connection")
        self._spawn()                       # 补上被接走的那条
        return s, self.relay

    def settimeout(self, t: float | None) -> None:
        self._timeout = t

    def close(self) -> None:
        self._stop.set()
        while not self._ready.empty():
            try:
                self._ready.get_nowait().close()
            except (queue.Empty, OSError):
                break

    def getsockname(self) -> Addr:
        """没有真实监听地址 —— 报中继的，日志里好读。"""
        return self.relay


class Peer:
    """到某个对端的保温长连接。线程安全，断开自动重连。

    `relay` 给了就经中继拨号（节点之间没有直连时），否则直连 `addr`。
    两条路返回的都是一条普通 socket，上层无从分辨 —— 这是有意的。
    """

    def __init__(self, node_id: str, addr: Addr, *, retries: int = 40,
                 relay: Addr | None = None, me: str = "?"):
        self.node_id = node_id
        self.addr = addr
        self.retries = retries
        self.relay = tuple(relay) if relay else None
        self.me = me
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.bytes_sent = 0
        self.msgs_sent = 0

    def _connect(self) -> socket.socket:
        last: Exception | None = None
        for _ in range(self.retries):
            try:
                if self.relay:
                    return dial_via_relay(self.relay, self.me, self.node_id)
                s = socket.create_connection(self.addr, timeout=10.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except (OSError, ConnectionError) as e:  # 对端还没起来
                last = e
                time.sleep(0.05)
        where = f"经中继 {self.relay}" if self.relay else f"@{self.addr}"
        raise ConnectionError(f"连不上 {self.node_id} {where}: {last}")

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
        self.relay: Addr | None = None
        """给了就所有对端都经中继 —— 节点之间没有直连时（deploy/relay.py）。"""
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

    def use_relay(self, relay: Addr | None) -> None:
        """改走中继。地址表仍然要注册 —— 中继模式下地址只是个占位。"""
        self.relay = tuple(relay) if relay else None

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
                p = Peer(node_id, self._addr[node_id], relay=self.relay, me=self.me)
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
