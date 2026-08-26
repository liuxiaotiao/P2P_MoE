"""真实网络探测 —— 把 SimNetwork 换成真机。

规划器只通过 `NetworkOracle.probe(a, b, k) -> Probe(p50, p95)` 这一个方法看网络，
所以从模拟切到真机，改的只有这一个类。规划器本身一行不动。

**唯一的结构性差别：探测动作必须由节点自己发起。**

逐对代价 h(v,v′) 里最大的一项是两端的接入段（II.3.1：延迟 ≈ 出口接入 + 骨干 +
入口接入，接入段主导）。从控制器去 ping 两台机器，量到的是「控制器→A」和
「控制器→B」，这两个数里都含控制器自己的接入段，而不含 A 与 B 之间的那条路。
所以流程是：控制器下发探测指令 → 节点 A 执行 → A 把结果回传。

这也是为什么真机部署绕不开 agent：没有 agent，就没有「从 A 出发」的测量点。
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..planner.network import Probe
from ..runtime.wire import Addr, rpc

__all__ = ["RemoteNetworkOracle"]

log = logging.getLogger("p2pmoe.probe")


@dataclass
class RemoteNetworkOracle:
    """通过 agent 打真实探测的 NetworkOracle 实现。

    Parameters
    ----------
    addrs : 节点 id → (host, port)。用控制器**拨号用的**地址，不用 agent 自报的 ——
        agent 绑 0.0.0.0 时自报的 host 是 "0.0.0.0"，对别人没意义。
    symmetric : 把 (a,b) 与 (b,a) 视为同一次测量。

        默认开启，原因是诚实的：应用层只能量到 RTT，取 RTT/2 作单向估计；
        **没有时钟同步就观测不到方向不对称**。文档把正向接口 ŵ(t,v) 与回环
        d_loop(v,t) 当作两个独立的有向量，真机上这个区分是量不出来的 —— 只能
        靠 II.6 的在线仪表（decode 每 token 实付的 w 与 d_loop）事后拆开，
        那才是真正逐向的观测。关掉它会让探测次数翻倍，但拿到的仍是同一个数。
    """

    addrs: Mapping[str, Addr]
    k_default: int = 8
    symmetric: bool = True
    timeout: float = 30.0
    relay: Addr | None = None
    """给了就经中继下发探测指令。注意这时量到的是**经中继的**往返，不是两台
    之间的真实延迟 —— 规划据此做的放置仍然自洽（它优化的就是实付延迟），
    但别拿这些数字去推断链路质量。"""
    _cache: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    n_rpc: int = 0
    failures: list[str] = field(default_factory=list)

    # -- NetworkOracle ----------------------------------------------------- #
    def probe(self, a: str, b: str, k: int) -> Probe:
        if a == b:
            return Probe(p50=0.0, p95=0.0, k=k)
        key = (min(a, b), max(a, b), k) if self.symmetric else (a, b, k)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit

        src, dst = (key[0], key[1]) if self.symmetric else (a, b)
        pr = self._ask(src, dst, k)
        with self._lock:
            self._cache[key] = pr
        return pr

    def _ask(self, src: str, dst: str, k: int) -> Probe:
        """让 src 去量 dst。"""
        try:
            reply = rpc(
                self.addrs[src],
                {"type": "probe", "peer": dst, "addr": list(self.addrs[dst]), "k": k},
                timeout=self.timeout, relay=self.relay, to=src,
            )
            self.n_rpc += 1
        except Exception as e:  # 控制器连不上该 agent
            self.failures.append(f"{src}→{dst}: 控制器连不上 {src}: {e}")
            return Probe(p50=float("inf"), p95=float("inf"), k=k)

        if not reply.get("ok"):
            # A 连不上 B —— 这是**真实的拓扑事实**（NAT / 防火墙 / 单向可达），
            # 不是错误。返回 inf 让规划器把这条链路自然排除。
            self.failures.append(f"{src}→{dst}: {reply.get('error', '不可达')}")
            return Probe(p50=float("inf"), p95=float("inf"), k=k)
        return Probe(p50=float(reply["p50"]), p95=float(reply["p95"]), k=k)

    # -- 预热 -------------------------------------------------------------- #
    def warm_all(self, nodes: Sequence[str], k: int | None = None, workers: int = 8) -> int:
        """并发预热全对矩阵。

        规划期会按需探测，但那是串行的；|V|=24 时全对是 276 次，一次几十毫秒 ×
        k 次采样，串行要几分钟。这里并发跑一遍，后面规划直接命中缓存。
        """
        k = k or self.k_default
        pairs = [
            (a, b)
            for i, a in enumerate(nodes)
            for b in (nodes[i + 1:] if self.symmetric else nodes)
            if a != b
        ]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda ab: self.probe(ab[0], ab[1], k), pairs))
        if self.failures:
            log.warning("探测中有 %d 条链路不可达", len(self.failures))
        return len(pairs)

    def reachability(self, nodes: Sequence[str]) -> dict[str, int]:
        """每个节点能达到多少对端 —— 不可达是分散环境的常态，值得单独看。"""
        out: dict[str, int] = {}
        for a in nodes:
            n = 0
            for b in nodes:
                if a == b:
                    continue
                key = (min(a, b), max(a, b), self.k_default) if self.symmetric else (a, b, self.k_default)
                pr = self._cache.get(key)
                if pr is not None and pr.p50 != float("inf"):
                    n += 1
            out[a] = n
        return out
