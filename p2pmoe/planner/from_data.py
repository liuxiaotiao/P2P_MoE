"""把**真实测量数据**接进规划器：激活画像 CSV + 网络拓扑 JSON。

规划器本身不依赖任何推理框架，也不该依赖任何特定的数据格式 —— 这个模块是
那道翻译层，把外部量得的东西翻成 `ActivationProfile` / `Node` / `NetworkOracle`。

三处口径必须说清楚，因为它们决定了后面所有数字
------------------------------------------------
**1. 「激活度」= count × mean_weight。** CSV 给的是计数与平均门控权重，两者
相乘才是**总门控质量**。用计数会高估那些「常被选中但权重很低」的专家 ——
而 top-k 里排第 10 的那个权重可能只有 0.02，丢了几乎无害。文档 III.7.3 的 q
定义用的就是质量占比，不是频次。

**2. 用 decode 相，不用 prefill。** 一条请求的生命周期由 decode 步主导
（prefill 一次、decode 几十到几百次），而通道二的滑窗看到的也是 decode。
两相的分布并不相同（实测 prefill 更集中），混在一起会让基线标定失准。

**3. 逐对延迟 = 传播时延 + 载荷/带宽。** 文档的 h(v,v′) 是单向时延。decode 每
token 只传一个 hidden state（d_model × 2 字节，几 KB），传输时间在 10Gbps 上是
微秒级，**延迟由传播时延主导**；prefill 传整段（T × 几 KB），带宽才开始起作用。
所以两者都算进去，但按 decode 的载荷定标 —— 那才是逐 token 预算要付的。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .experts import ActivationProfile
from .network import Probe
from .types import Node

__all__ = ["load_activation_csv", "load_topology", "FabricOracle", "TopoInfo"]


# --------------------------------------------------------------------------- #
def load_activation_csv(
    path: str | Path, *, task: str, phase: str = "decode",
    n_experts: int | None = None, n_layers: int | None = None,
) -> tuple[ActivationProfile, dict]:
    """逐层逐专家的激活 CSV → `ActivationProfile`。

    期望列：`layer, expert, prefill_count, decode_count,
    prefill_mean_weight, decode_mean_weight`（层与专家都是 0-based）。

    返回 (画像, 诊断)。诊断里会点名**权重为 NaN 但计数非零**的格子 ——
    那是采集端的 bug（除零得到的 NaN 会伴随零计数，计数非零却是 NaN 说明
    权重累加器坏了），不是「没被路由到」。悄悄按 0 处理会让那些专家永远
    进不了驻留集，而它们可能恰恰是热的。
    """
    rows = list(csv.DictReader(open(Path(path), newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{path} 是空的")
    L = n_layers or max(int(r["layer"]) for r in rows) + 1
    E = n_experts or max(int(r["expert"]) for r in rows) + 1
    cnt_col, w_col = f"{phase}_count", f"{phase}_mean_weight"
    if cnt_col not in rows[0]:
        raise ValueError(f"{path} 没有 {cnt_col} 列；有的是 {list(rows[0])}")

    mass = np.zeros((L, E), dtype=np.float64)
    suspect: list[tuple[int, int, float]] = []
    for r in rows:
        l, e = int(r["layer"]), int(r["expert"])
        c, w = float(r[cnt_col]), float(r[w_col])
        if math.isnan(w) or math.isinf(w):
            if c > 0:
                suspect.append((l, e, c))       # 计数非零却没有权重 → 采集 bug
            continue
        mass[l, e] = c * w

    empty = [l for l in range(L) if mass[l].sum() <= 0]
    if empty:
        raise ValueError(f"{path} 的第 {empty[:5]} 层激活质量全为 0 —— 数据不完整")
    prof = ActivationProfile(
        task=task, n_layers=L, n_experts=E,
        mass=tuple(tuple(mass[l] / mass[l].sum()) for l in range(L)),
    )
    lost = defaultdict(float)
    for l, e, c in suspect:
        lost[l] += c
    diag = {
        "rows": len(rows), "n_layers": L, "n_experts": E, "phase": phase,
        "n_suspect": len(suspect),
        "suspect_layers": sorted(lost),
        "suspect_experts": sorted({e for _, e, _ in suspect}),
        "suspect_count_share": {
            l: lost[l] / (lost[l] + mass[l].sum()) for l in sorted(lost)
        },
    }
    return prof, diag


# --------------------------------------------------------------------------- #
@dataclass
class TopoInfo:
    """拓扑 JSON 读出来的原始事实。"""

    nodes: list[Node]
    p50_ms: dict[tuple[str, str], float]
    bandwidth_mbps: dict[tuple[str, str], float]
    prop_ms: dict[tuple[str, str], float]
    raw: dict = field(default_factory=dict)

    @property
    def reachable_pairs(self) -> int:
        return len(self.p50_ms)


def load_topology(
    path: str | Path, *, d_model: int = 2048, dtype_bytes: int = 2,
    reserve_gb: float = 1.0, avail: float = 0.95,
    ref_capacity: float | None = None, ms_per_layer_ref: float = 0.35,
) -> TopoInfo:
    """拓扑 JSON → 规划器的节点表 + 逐对单向时延。

    期望格式::

        {"nodes": [{"id","compute_capacity","memory_capacity","role"}, …],
         "edges": [{"u","v","bandwidth","propagation_delay"}, …]}

    `bandwidth` 单位 Mbps，`propagation_delay` 单位 ms。

    **算力 → ms_per_layer 是相对的。** 拓扑给的是抽象算力值，不是「每层多少毫秒」。
    这里以最快的那档为基准，按反比缩放 —— 规划器只用它做相对比较（谁算得快
    就多担几层），绝对值不影响放置，只影响打印出来的估算延迟。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    caps = {n["id"]: float(n["compute_capacity"]) for n in raw["nodes"]}
    ref = ref_capacity or max(caps.values())
    nodes = [
        Node(
            id=n["id"],
            tier=str(n.get("role", f"{n['memory_capacity']:.0f}GB")),
            mem_gb=float(n["memory_capacity"]),
            ms_per_layer=round(ms_per_layer_ref * ref / caps[n["id"]], 4),
            reserve_gb=reserve_gb,
            avail=avail,
        )
        for n in raw["nodes"]
    ]

    # decode 每 token 的载荷：一个 hidden state
    payload_bits = d_model * dtype_bytes * 8
    p50: dict[tuple[str, str], float] = {}
    bw: dict[tuple[str, str], float] = {}
    pr: dict[tuple[str, str], float] = {}
    for e in raw["edges"]:
        u, v = e["u"], e["v"]
        mbps = float(e["bandwidth"])
        prop = float(e["propagation_delay"])
        tx_ms = payload_bits / (mbps * 1e6) * 1e3        # 传输时间（ms）
        for a, b in ((u, v), (v, u)):                    # 无向边 → 两个方向
            p50[(a, b)] = prop + tx_ms
            bw[(a, b)] = mbps
            pr[(a, b)] = prop
    return TopoInfo(nodes=nodes, p50_ms=p50, bandwidth_mbps=bw, prop_ms=pr, raw=raw)


# --------------------------------------------------------------------------- #
class FabricOracle:
    """把测量好的逐对时延喂给规划器（实现 `NetworkOracle` 的 probe 接口）。

    **这份拓扑没有抖动维度。** 真实探测会给出 (p50, p95) 两个分位，而这里只有
    一个确定的传播时延 —— 于是 `jitter = p95 − p50 = 0`，文档里那两道闸
    （尾闸 β、抖动闸 J_cap）**恒定通过**，形同虚设。

    这不是实现偷懒，是数据里没有那一维。后果要说清楚：
    * 间隙检测（II.2.1）与回环的抖动闸不会淘汰任何链路；
    * 均匀性目标 A1″ 退化成「把 p50 的极差压小」，而 p50 极差本来就比抖动小 ——
      **公共中值域会比真实分散环境宽松得多**，规划出来的通道数偏乐观。

    想让这两道闸真正起作用，拓扑里得带上抖动（或 p95）。这里用
    `jitter_frac` 可以人为注入一个与 p50 成比例的抖动做敏感性分析，
    默认 0 —— 宁可让人看见「闸没起作用」，也不要偷偷编一个数。
    """

    def __init__(self, topo: TopoInfo, *, jitter_frac: float = 0.0):
        self.topo = topo
        self.jitter_frac = jitter_frac
        self.n_probe = 0

    def probe(self, a: str, b: str, k: int) -> Probe:
        self.n_probe += 1
        if a == b:
            return Probe(p50=0.0, p95=0.0, k=k)
        m = self.topo.p50_ms.get((a, b))
        if m is None:
            return Probe(p50=float("inf"), p95=float("inf"), k=k)   # 不可达
        return Probe(p50=m, p95=m * (1.0 + self.jitter_frac), k=k)
