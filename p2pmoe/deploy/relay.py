#!/usr/bin/env python3
"""中继：节点之间**没有**直连时的兜底。

    # 一台有公网入站的机器上（控制机通常就是）
    python3 -m p2pmoe.deploy.relay --bind 0.0.0.0:9200

    # 每个节点：不再监听，改成挂到中继上
    python3 -m p2pmoe.deploy.agent --id n3 --relay relay.example.com:9200

为什么需要它
------------
数据面是节点**直接互发** hidden state 的 —— 段内转发、跨接口出段、绕环 decode，
全是节点到节点。这要求 15 台两两可达。而分散环境里最常见的情形恰恰相反：
消费级显卡挂在家庭宽带后面，有出站没入站，两台之间根本连不上。

中继让每台只需要**一条出站连接**：所有节点连到同一台有公网入站的机器上，
由它把两条连接接起来。

它是接线员，不是代理
--------------------
握手完成之后中继**只搬字节**，不认识 p2pmoe 的消息格式：

    n1 → 中继：{"type":"dial","to":"n5"}         想找 n5
    中继 → n5：{"type":"accepted","from":"n1"}   用 n5 事先挂着的一条连接
    中继 → n1：{"type":"dialed","ok":true}
    然后两条 socket 对接，双向 splice 到任意一端断开

所以 `wire.py` 的消息层、`node.py` 的处理逻辑一行都不用改 —— 对它们来说这就是
一条普通的 TCP 流。这是刻意的：中继是**部署形态**的选择，不该渗进协议里。

代价，说清楚
------------
* **每一跳变成两段**。正向接口、回环、段内转发全要经中继绕一圈，逐 token 延迟
  大致翻倍。方案的均匀性机制照常工作（它优化的是放置），但绝对延迟回不来；
* **中继是带宽瓶颈与单点**。所有 hidden state 都过它。15 台的规模还好，
  再大就要多开几台按段分流；
* **探测量到的是「经中继的延迟」**，不是两台之间的真实延迟。规划器据此做的
  放置仍然自洽（它优化的就是实际付出的延迟），但别拿这些数字去推断链路质量。

**生产环境更该用 VPN overlay**（WireGuard / Tailscale）：给每台一个虚拟 IP，
节点之间恢复真正的直连，框架这边什么都不用改（`--bind` 到虚拟网卡即可）。
中继是零配置的兜底，不是最优解。
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from ..runtime.wire import recv_msg, send_msg

log = logging.getLogger("p2pmoe.relay")

__all__ = ["Relay"]

_PARK_TIMEOUT = 30.0     # dial 时等对方挂起连接的上限


@dataclass
class _Parked:
    """一个节点事先挂在中继上、等着被接的连接。"""

    conn: socket.socket
    at: float


class Relay:
    """接线员。挂起（listen）与拨号（dial）两种客户端，接上之后纯搬字节。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 9200):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(512)
        self.host, self.port = self.sock.getsockname()[:2]
        self._parked: dict[str, deque[_Parked]] = {}
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.n_spliced = 0
        self.bytes_relayed = 0
        self._lock = threading.Lock()

    # -- 生命周期 ---------------------------------------------------------- #
    def start(self) -> "Relay":
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass
        with self._cv:
            for q in self._parked.values():
                for p in q:
                    try:
                        p.conn.close()
                    except OSError:
                        pass
            self._parked.clear()
            self._cv.notify_all()

    def __enter__(self) -> "Relay":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- 接入 -------------------------------------------------------------- #
    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._greet, args=(conn, addr),
                             daemon=True).start()

    def _greet(self, conn: socket.socket, addr) -> None:
        try:
            h, _ = recv_msg(conn)
        except (OSError, ConnectionError):
            conn.close()
            return
        t = h.get("type")
        if t == "listen":
            self._park(str(h["node"]), conn)
        elif t == "dial":
            self._dial(str(h.get("from", "?")), str(h["to"]), conn)
        elif t == "status":
            with self._cv:
                depth = {n: len(q) for n, q in self._parked.items() if q}
            try:
                send_msg(conn, {"type": "status_ack", "parked": depth,
                                "spliced": self.n_spliced,
                                "bytes": self.bytes_relayed})
            except OSError:
                pass
            conn.close()
        else:
            try:
                send_msg(conn, {"type": "error", "why": f"未知的握手 {t!r}"})
            except OSError:
                pass
            conn.close()

    def _park(self, node: str, conn: socket.socket) -> None:
        """把一条连接挂起，等别人来拨。

        节点会同时挂好几条 —— 一条被接走就少一条，不预挂的话每次建链都要先
        通知节点再等它连过来，多一个往返。
        """
        with self._cv:
            self._parked.setdefault(node, deque()).append(_Parked(conn, time.time()))
            self._cv.notify_all()
        log.debug("挂起 %s（现有 %d 条）", node, len(self._parked[node]))

    def _dial(self, src: str, dst: str, dialer: socket.socket) -> None:
        deadline = time.time() + _PARK_TIMEOUT
        target: socket.socket | None = None
        with self._cv:
            while not self._stop.is_set():
                q = self._parked.get(dst)
                while q:
                    cand = q.popleft()
                    # 挂太久的连接可能已经死了，先探一下再用
                    if _alive(cand.conn):
                        target = cand.conn
                        break
                    cand.conn.close()
                if target is not None or time.time() >= deadline:
                    break
                self._cv.wait(timeout=0.2)

        if target is None:
            _try_send(dialer, {"type": "dialed", "ok": False,
                               "why": f"{dst} 没有挂在中继上（它的 agent 起了吗？"
                                      f"是不是没加 --relay？）"})
            dialer.close()
            return

        if not _try_send(target, {"type": "accepted", "from": src}):
            # 挂起的那条刚好断了 —— 递归重试一次，队列里通常还有别的
            target.close()
            self._dial(src, dst, dialer)
            return
        if not _try_send(dialer, {"type": "dialed", "ok": True}):
            target.close()
            dialer.close()
            return

        with self._lock:
            self.n_spliced += 1
        log.debug("接通 %s → %s", src, dst)
        self._splice(dialer, target)

    # -- 搬字节 ------------------------------------------------------------ #
    def _splice(self, a: socket.socket, b: socket.socket) -> None:
        """双向对拷，任意一端断开就收摊。

        这里**完全不解析消息**。中继不认识 p2pmoe 的协议，也不该认识 ——
        协议要是渗进中继，以后改一次消息格式就要同步改两处。
        """
        done = threading.Event()

        def pump(src: socket.socket, dst: socket.socket) -> None:
            n = 0
            try:
                while not self._stop.is_set():
                    buf = src.recv(1 << 16)
                    if not buf:
                        break
                    dst.sendall(buf)
                    n += len(buf)
            except OSError:
                pass
            finally:
                with self._lock:
                    self.bytes_relayed += n
                done.set()
                for s in (src, dst):
                    try:
                        s.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        t = threading.Thread(target=pump, args=(a, b), daemon=True)
        t.start()
        pump(b, a)
        done.wait(timeout=5.0)
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass


def _alive(s: socket.socket) -> bool:
    """对端还在吗 —— 非阻塞窥一眼，收到 EOF 就是没了。"""
    try:
        s.setblocking(False)
        try:
            return s.recv(1, socket.MSG_PEEK) != b""
        except BlockingIOError:
            return True          # 没数据可读 = 连接还开着
        except OSError:
            return False
    finally:
        try:
            s.setblocking(True)
        except OSError:
            pass


def _try_send(s: socket.socket, header: dict) -> bool:
    try:
        send_msg(s, header)
        return True
    except (OSError, ConnectionError):
        return False


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2pmoe-relay",
                                 description="节点之间没有直连时的中继")
    ap.add_argument("--bind", default="0.0.0.0:9200")
    ap.add_argument("--stats-every", type=float, default=30.0,
                    help="每隔多少秒打一次统计。0 = 不打")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    host, _, port = args.bind.rpartition(":")
    r = Relay(host or "0.0.0.0", int(port)).start()
    log.info("中继在 %s:%d —— 各节点加 --relay <这台的地址>:%d", r.host, r.port, r.port)
    log.info("注意：所有 hidden state 都会过这台机器，逐 token 延迟大致翻倍。"
             "生产环境更该用 WireGuard/Tailscale 之类的 VPN overlay 恢复直连")
    try:
        while True:
            time.sleep(max(args.stats_every, 1.0))
            if args.stats_every > 0:
                with r._cv:
                    parked = {n: len(q) for n, q in r._parked.items() if q}
                log.info("挂起 %s；累计接通 %d 次，搬运 %.1fMB",
                         parked or "无", r.n_spliced, r.bytes_relayed / 1e6)
    except KeyboardInterrupt:
        pass
    r.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
