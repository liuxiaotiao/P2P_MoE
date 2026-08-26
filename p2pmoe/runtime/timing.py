"""逐请求的时序报告：总时延怎么分掉的，各节点算力用了多少。

回答两个问题
------------
**总时延去哪了？** 排队 / prefill / decode 三段，再往下拆成「各节点实际算了多久」
与「其余」。其余里包含网络往返、协议开销、线程调度 —— 分不开，也不假装分得开。

**算力使用率多少？** 一个节点在这条请求存续期间，有多少比例的时间在算。
分散环境下这个数会很低（文档 I.2.1 说网络占九成），本模块就是用来把「九成」
这个说法变成你自己池子里的实测数字。

为什么网络时间只能反推
----------------------
15 台机器**没有时钟同步**。跨机的绝对时刻拼不到一条时间轴上，所以
「n1 发出 → n2 收到」这段单向时延**测不了**（这也是 `deploy/probe.py` 里
只能量 RTT 取一半的同一个原因）。

能测的只有两样：
* **各节点的计算时长** —— 本地单调时钟量时长，本地可信；
* **端到端总时延** —— 协调器一台机器上的两个时刻之差，同一个时钟。

于是：`网络 + 开销 = 总时延 − Σ 各节点计算 − 排队`。这是个**上界**（把所有
说不清的都算进去了），但它是诚实的：没有把测不出来的东西假装成测出来的。

单请求口径
----------
一条通道同时只服务一条请求（I.2.4 排他独占），所以这里的每个数都是那一条请求
独占整条路径时的表现，没有批内干扰。**低使用率是设计的已知代价**，不是 bug ——
方案主动用算力利用率换确定性延迟，这里量的就是这笔交易的价码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = ["NodeTiming", "RequestTiming", "summarise_request"]


@dataclass
class NodeTiming:
    node: str
    role: str
    segment: str
    layers: tuple[int, ...]
    n_experts: int
    compute_ms: float
    n_forward: int
    bytes_out: int

    @property
    def layer_span(self) -> str:
        if not self.layers:
            return "—"
        return (f"{self.layers[0]}–{self.layers[-1]}" if len(self.layers) > 1
                else str(self.layers[0]))

    @property
    def ms_per_forward(self) -> float:
        return self.compute_ms / self.n_forward if self.n_forward else 0.0


@dataclass
class RequestTiming:
    req: str
    front: str
    back: str
    n_prompt: int
    n_generated: int
    total_ms: float
    queue_ms: float
    prefill_ms: float
    """到首 token 为止（含排队）。"""
    decode_ms: float
    token_ms: list[float] = field(default_factory=list)
    nodes: list[NodeTiming] = field(default_factory=list)
    missing_traces: list[str] = field(default_factory=list)

    # -- 汇总 -------------------------------------------------------------- #
    @property
    def compute_ms(self) -> float:
        return sum(n.compute_ms for n in self.nodes)

    @property
    def other_ms(self) -> float:
        """网络 + 协议开销 + 调度。**反推出来的上界**，不是测量值。"""
        return max(0.0, self.total_ms - self.compute_ms - self.queue_ms)

    @property
    def utilisation(self) -> float:
        """整条路径的算力使用率 = Σ 计算 / 总时延。

        注意分母是**墙钟**而不是「节点数 × 墙钟」：这条路上的节点是**流水线**，
        同一时刻只有一个在算（段内逐跳、跨段绕环）。所以这个比值的含义是
        「这条请求的生命周期里，有多少时间真的在做矩阵乘」。
        """
        return self.compute_ms / self.total_ms if self.total_ms > 0 else 0.0

    def node_utilisation(self, n: NodeTiming) -> float:
        return n.compute_ms / self.total_ms if self.total_ms > 0 else 0.0

    @property
    def per_token_ms(self) -> float:
        return self.decode_ms / self.n_generated if self.n_generated else 0.0

    # -- 打印 -------------------------------------------------------------- #
    def render(self, *, width: int = 78) -> str:
        L: list[str] = []
        line = "─" * width

        def pct(x: float) -> str:
            return f"{x / self.total_ms:6.1%}" if self.total_ms > 0 else "     —"

        L.append(f"请求 {self.req}：prompt {self.n_prompt} token，"
                 f"生成 {self.n_generated} token")
        L.append(f"路径 {self.front} → {self.back}"
                 f"（{len(self.nodes)} 台机器，流水线）")
        L.append(line)
        L.append(f"总时延 {self.total_ms:9.0f}ms")
        L.append(f"  ├ 排队      {self.queue_ms:9.0f}ms {pct(self.queue_ms)}")
        L.append(f"  ├ prefill  {self.prefill_ms:9.0f}ms {pct(self.prefill_ms)}"
                 f"   （到首 token）")
        L.append(f"  └ decode   {self.decode_ms:9.0f}ms {pct(self.decode_ms)}"
                 f"   {self.n_generated} 步，逐步均值 {self.per_token_ms:.0f}ms")
        if self.token_ms:
            srt = sorted(self.token_ms)
            L.append(f"      逐 token  p50 {srt[len(srt)//2]:.0f}ms   "
                     f"p95 {srt[min(len(srt)-1, int(len(srt)*0.95))]:.0f}ms   "
                     f"max {srt[-1]:.0f}ms")

        L.append("")
        L.append(f"  {'节点':<8}{'角色':<14}{'层':<9}{'专家':>6}"
                 f"{'计算':>10}{'占总时延':>10}{'每次前向':>10}")
        for n in self.nodes:
            L.append(f"  {n.node:<8}{n.role:<14}{n.layer_span:<9}{n.n_experts:>6}"
                     f"{n.compute_ms:>9.0f}ms{self.node_utilisation(n):>10.1%}"
                     f"{n.ms_per_forward:>9.1f}ms")
        if self.missing_traces:
            L.append(f"  ⚠ 没收到埋点的节点：{self.missing_traces}"
                     f"（下面的计算合计因此偏小）")

        L.append(line)
        L.append(f"  计算合计   {self.compute_ms:9.0f}ms   "
                 f"**算力使用率 {self.utilisation:.1%}**")
        L.append(f"  其余       {self.other_ms:9.0f}ms   {pct(self.other_ms)}"
                 f"   网络往返 + 协议开销 + 调度")
        L.append("")
        L.append("  「其余」是**反推**的上界，不是测量值 —— 15 台机器没有时钟同步，")
        L.append("  跨机的单向时延测不了，只能用总时延减掉能测的部分。")
        L.append(f"  使用率低是设计的已知代价：一条通道同时只服务一条请求"
                 f"（I.2.4 排他独占），")
        L.append("  用算力利用率换确定性延迟。这个数就是那笔交易的价码。")
        return "\n".join(L)


def summarise_request(rec, coord) -> RequestTiming:
    """把 `RequestRecord` + 各节点埋点，拼成一份可读的时序报告。"""
    total = (rec._last - rec.t0) * 1000 if rec._last else 0.0
    queue = rec.wait_front_ms
    prefill = (rec.t_first - rec.t0) * 1000 if rec.t_first else 0.0
    decode = max(0.0, total - prefill)

    order = coord.expected_trace_nodes(rec)
    # 按「段内位置」排序，读起来就是数据流的顺序
    pos = {}
    for sid in (rec.front, rec.back):
        for i, v in enumerate(coord.seg_nodes.get(sid, [])):
            pos[v] = (0 if sid == rec.front else 1, i)

    nodes = [
        NodeTiming(
            node=t["node"], role=t.get("role", "?"), segment=t.get("segment", "?"),
            layers=tuple(t.get("layers", ())), n_experts=int(t.get("n_experts", 0)),
            compute_ms=float(t.get("compute_ms", 0.0)),
            n_forward=int(t.get("n_forward", 0)),
            bytes_out=int(t.get("bytes_out", 0)),
        )
        for t in rec.traces.values()
    ]
    nodes.sort(key=lambda n: pos.get(n.node, (9, 9)))

    return RequestTiming(
        req=rec.req, front=rec.front, back=rec.back,
        n_prompt=len(rec.ids), n_generated=len(rec.tokens),
        total_ms=total, queue_ms=queue, prefill_ms=prefill, decode_ms=decode,
        token_ms=list(rec.token_ms), nodes=nodes,
        missing_traces=[v for v in order if v not in rec.traces],
    )
