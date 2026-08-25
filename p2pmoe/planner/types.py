"""核心数据结构 —— 对应方案文档第〇部分（记号与术语）与 I.2（形式化）。

单位约定
--------
* 延迟：毫秒 (ms)
* 内存：十进制 GB (1e9 字节)。附录 C 的算例即按此口径，例如
  ctx=4096 时每层 KV = 4096 × 2 × 4096 × 2 B = 67.1 MB = 0.067 GB。
* 配对代价：一律为 k 次采样的分位数量 (p50 / p95)，见第〇部分「分位数记号」。

段 (Segment) 的定义见第〇部分：承载连续层区间的有序节点链，允许单节点成段
(s_min = 1)。段有入口 head(π) 与出口 tail(π)；单节点段两者同一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:  # 只用于标注，避免与 experts.py 形成顶层循环依赖
    from .experts import ExpertPlacement

__all__ = [
    "Node",
    "ModelSpec",
    "TaskProfile",
    "SegmentSpec",
    "Segment",
    "Objective",
    "PlannerConfig",
]


# --------------------------------------------------------------------------- #
# 节点与模型
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Node:
    """一台分散 GPU 节点。

    分散环境下节点之间没有机房/共享上联的结构，唯一的结构性差异来自
    (a) 内存档位 M_v，(b) 接入质量（由 NetworkOracle 实测体现），
    (c) 可用率 r_v（churn 频繁，进求解器评分一等项，见 II.1）。
    """

    id: str
    tier: str
    mem_gb: float
    ms_per_layer: float
    """每层每 token 的计算耗时。等价于算力 H_v 的倒数形式；
    注意 c_l = attention + top-k 专家 FFN 与 task 无关（I.1.1 关键澄清）。"""
    reserve_gb: float = 1.0
    avail: float = 0.99
    """r_v，可用率。churn 越频繁越低。"""

    @property
    def usable_gb(self) -> float:
        """M_v − 预留。所有内存判据都以此为准（I.2.2）。"""
        return self.mem_gb - self.reserve_gb


@dataclass(frozen=True)
class ModelSpec:
    """MoE 模型的静态形状。"""

    n_layers: int
    d_model: int
    n_experts: int
    top_k: int
    base_gb_per_layer: float
    """每层非专家部分（attention + norm + router）的权重量。"""
    expert_gb: float
    """单个专家的权重量。"""
    ctx_max: int
    kv_bytes_per_elem: int = 2
    """fp16 = 2 bytes。KV 存 K/V 各一份，公式里另有一个 ×2。"""
    kv_dim: int | None = None
    """K/V 投影的输出维度。None 表示等于 d_model（多头注意力 MHA）。

    **真实模型几乎都用 GQA，这一项不能省。** 文档 I.2.2 的公式
    kv = |S| × ctx × 2 × d_model × 2 bytes 隐含了 MHA（KV 维 = d_model）。
    Qwen3-30B-A3B 是 4 个 KV 头 × head_dim 128 = 512，而 d_model 是 2048 ——
    按 d_model 算会把 KV 高估 4 倍，直接影响可行性判断与跳数下界。
    """

    @property
    def kv_gb_per_layer(self) -> float:
        """kv(S, ctx) / |S| —— 每层的 KV 上限占用（I.2.2，按 GQA 修正）。"""
        d = self.kv_dim if self.kv_dim else self.d_model
        return self.ctx_max * 2 * d * self.kv_bytes_per_elem / 1e9

    def kv_gb(self, n_layers: int) -> float:
        return n_layers * self.kv_gb_per_layer

    def weight_gb_per_layer(self, n_resident_experts: int) -> float:
        """驻留 n 个专家时每层的权重量。

        n 对前段取全 task 并集，对后段取 task 专用集 n_{u,l}。
        """
        return self.base_gb_per_layer + n_resident_experts * self.expert_gb


@dataclass(frozen=True)
class TaskProfile:
    """一类 task 的画像。

    唯一给定的流量信息是占比 λ_u —— 不给到达率、不给平均占用时长
    （I.1.1、II.7 开头）。
    """

    name: str
    lam: float
    experts_per_layer: int | Sequence[int]
    """n_{u,l}，**基数**。给标量表示逐层同质；给序列表示逐层异构（索引 0 对应 layer 1）。

    只给基数足够算内存，但不足以做可检性（III.7.3）、池合并（III.7.4）、
    前段并集与在线 miss 检出 —— 那些需要 placement 里的专家**身份**。
    """
    placement: "ExpertPlacement | None" = None
    """逐层驻留专家集（身份）。给了它，experts_per_layer 就只是缓存，一切以它为准。"""

    def n_experts_at(self, layer: int) -> int:
        if self.placement is not None:
            return self.placement.size_at(layer)
        if isinstance(self.experts_per_layer, int):
            return self.experts_per_layer
        return self.experts_per_layer[layer - 1]

    def experts_at(self, layer: int) -> frozenset[int]:
        """该层要驻留哪些专家（身份）。没有 placement 时抛错 —— 与其返回一个
        编出来的集合，不如让调用方知道这份画像里没有身份信息。"""
        if self.placement is None:
            raise ValueError(
                f"task {self.name} 只有专家基数没有身份；"
                f"请用 experts.build_placement() 从回放画像构造 placement"
            )
        return self.placement.at(layer)

    def resident_experts(self, lo: int, hi: int) -> int:
        """层区间 [lo, hi] 上的平均驻留专家数（用于粗估）。"""
        vals = [self.n_experts_at(l) for l in range(lo, hi + 1)]
        return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------- #
# 段的规格与结果
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SegmentSpec:
    """deploy_path 的输入规格。前后段共用求解器，差异全部收进本结构（II.1 表）。"""

    kind: str
    """"front" | "back" """
    task: str | None
    layer_lo: int
    layer_hi: int
    gb_per_layer: Callable[[int], float]
    """逐层权重形状 mem_F(l) 或 mem_u(l)。"""
    kv_gb_per_layer: float
    ms_per_layer_scale: float = 1.0
    tail_allowed: frozenset[str] | None = None
    """Q_F^out。None 表示不设资格约束（后段主线，Step 3 自由构造）。"""
    head_allowed: frozenset[str] | None = None
    """Q_F^in 主线不设约束，改由回环裁剪择优（II.4 Step 6）。"""

    @property
    def n_layers(self) -> int:
        return self.layer_hi - self.layer_lo + 1

    def resident_gb(self, lo: int, hi: int) -> float:
        """层区间 [lo, hi] 在一个节点上的总驻留量（权重 + KV）。

        KV 显式入判据 —— 16GB 档节点上 KV 常与权重同量级，遗漏会使可行性
        判断系统性偏乐观（I.2.2）。
        """
        w = sum(self.gb_per_layer(l) for l in range(lo, hi + 1))
        return w + (hi - lo + 1) * self.kv_gb_per_layer

    def total_gb(self) -> float:
        return self.resident_gb(self.layer_lo, self.layer_hi)

    def tail_ok(self, node_id: str) -> bool:
        return self.tail_allowed is None or node_id in self.tail_allowed

    def head_ok(self, node_id: str) -> bool:
        return self.head_allowed is None or node_id in self.head_allowed


@dataclass(frozen=True)
class Segment:
    """一条建成的段。"""

    kind: str
    task: str | None
    nodes: tuple[str, ...]
    splits: tuple[tuple[int, int], ...]
    """与 nodes 一一对应的层区间 (lo, hi)，闭区间、1-based。"""
    compute_ms: float
    hop_ms: float
    """段内跳的 p50 代价之和。单节点段为 0。"""
    jitter_ms: float = 0.0
    """段内跳的 (p95 − p50) 之和，仅用于诊断。"""

    @property
    def delay_ms(self) -> float:
        """T_π = Σ_层 c_l/H_v + Σ_相邻跳 h(v,v′)  （I.2.1）"""
        return self.compute_ms + self.hop_ms

    @property
    def hops(self) -> int:
        """hops(π) = |nodes(π)| − 1（第〇部分）。"""
        return len(self.nodes) - 1

    @property
    def head(self) -> str:
        return self.nodes[0]

    @property
    def tail(self) -> str:
        return self.nodes[-1]

    def label(self) -> str:
        who = self.task or "F"
        return f"{who}[{'+'.join(self.nodes)}]"


# --------------------------------------------------------------------------- #
# 求解目标与全局配置
# --------------------------------------------------------------------------- #
@dataclass
class Objective:
    """beam 求解器的评分函数（II.1 表「评分」行）。

    主项在两种模式间切换：
    * target_ms is None  → 纯求快（Step 3/5 的初建、以及 tighten 的「拆慢者」）
    * target_ms 给定     → 求靠近目标（tighten 的「拆快者」，见 II.2.2）

    stability 项 μ·Σ(−log r_v) 在分散环境下升为一等（II.1 三点差异之三）：
    churn 频繁时，可用率低的节点即使快也不值。
    """

    mu_ms: float = 40.0
    """稳定性权重。量纲是「每 nat 不可用率折算多少 ms」，默认取一跳量级。"""
    jitter_w: float = 0.2
    """抖动软罚系数（硬屏蔽由 J_cap 在测量层完成）。"""
    rent_nu_ms: float = 0.0
    """影子租金 ν：后段占用前段高潜节点 R_F 的每节点罚（II.4「影子租金」）。"""
    rent_nodes: frozenset[str] = frozenset()
    good_gamma_ms: float = 0.0
    """γ：tighten 拆快者时，鼓励其交出大内存好节点。"""
    good_nodes: frozenset[str] = frozenset()
    target_ms: float | None = None

    head_cost: Mapping[str, float] | None = None
    """δ̂_in(v)：若 v 做本段入口，正向接口 w 里由 v 贡献的那一份（实测代理）。"""
    tail_cost: Mapping[str, float] | None = None
    """δ̂_out(v)：若 v 做本段出口，回环 d_loop 里由 v 贡献的那一份（实测代理）。"""
    endpoint_w: float = 0.0
    """端点项权重。见下方 [端点项] 说明；置 0 即退回文档的原始目标函数。

    [端点项 —— 对文档目标函数的一处补正]

    I.2.1 的延迟模型是 T = T_F(f) + w(f,b) + T_B(b) + d_loop(b,f)。文档的
    Step 3 只优化 T_B，把 w 与 d_loop 完全留给后面的步骤。但 w 依赖 head(b)、
    d_loop 依赖 tail(b) —— Step 3 在自由构造后段时，其实已经把这两项的一半
    定死了，却没有把它们计入代价。

    后果在实测中很直接：Step 3 挑出的入口若落在接入质量差的节点上，它的
    in-access 会作为一个公共加项抬高**所有**候选出口对该入口的 ŵ，于是
    公共带被整体压缩，Step 4 无论怎么滑窗都凑不出人口。文档给的补救是
    Step 4→3 的异类入口诊断，但那一次只能换掉一个入口，且要求跃升足够显著。

    权重取 1.0 不是调参：head 的 in-access 被每个组合的每个 token 各付一次，
    它进总延迟的系数就是 1。这一项把「后段自由构造」从对接口完全无知，变成
    对接口的可测部分负责，同时不引入任何对网络结构的假设 —— δ̂ 是实测中位数，
    与 II.3.1(a) 的「纯实测驱动」一致。
    """

    def score(
        self,
        compute_ms: float,
        hop_ms: float,
        jitter_ms: float,
        nodes: Sequence[str],
        node_map: Mapping[str, Node],
        *,
        partial: bool = False,
    ) -> float:
        """partial=True 时用于 beam 前缀剪枝：对 target 模式取可容许的下界
        max(0, delay − target)，避免前缀过早被判为「偏离目标」。"""
        import math

        delay = compute_ms + hop_ms
        if self.target_ms is None:
            main = delay
        elif partial:
            main = max(0.0, delay - self.target_ms)
        else:
            main = abs(delay - self.target_ms)

        stab = self.mu_ms * sum(
            -math.log(max(node_map[v].avail, 1e-6)) for v in nodes
        )
        rent = self.rent_nu_ms * len(set(nodes) & self.rent_nodes)
        good = self.good_gamma_ms * len(set(nodes) & self.good_nodes)

        endpoint = 0.0
        if self.endpoint_w and nodes:
            if self.head_cost is not None:
                endpoint += self.head_cost.get(nodes[0], 0.0)
            if self.tail_cost is not None and not partial:
                # 出口只有在段放完时才确定，前缀阶段不计（否则不是下界）
                endpoint += self.tail_cost.get(nodes[-1], 0.0)
            endpoint *= self.endpoint_w

        return main + stab + self.jitter_w * jitter_ms + rent + good + endpoint


@dataclass
class PlannerConfig:
    """全流程超参。带 [启发式] 标记的项在文档 §III.9「诚实定位」中被明确列为
    无理论依据、需按实际校准的经验值。"""

    # --- 目标与闸 (I.2.3) ---
    eta: float = 0.12
    """相对均匀性目标 η。建议 10%~15%。"""
    beta: float = 1.25
    """目标区间对中位/下界的倍率。文档称其为「全局唯一需人工拍板的延迟政策」。"""
    j_cap_ms: float = 25.0
    """抖动上限 J_cap（p95 − p50）。超过者链路硬屏蔽。"""
    w_cap_ms: float | None = None
    """正向接口 p50 闸。None 表示由比例锚 ρ_w × 典型跳 p50 自动推定。"""
    rho_w: float = 1.5
    """w_cap 的比例锚倍率（分散环境建议 1~1.5）。"""
    w_cap95_ms: float | None = None
    t_cap95_ms: float | None = None

    # --- 采样 ---
    k_probe: int = 8
    """常规探测采样次数，文档要求 k ≥ 8。"""
    k_audit: int = 16
    """终审网格加密实测，文档要求 k ≥ 16。"""
    k_gate: int = 32
    """闸门（尾闸 / 抖动闸）判定用的采样数。

    **不能沿用 k_probe。** 文档统一写「k ≥ 8」，但 p50 与 p95 对样本量的需求
    差一个量级：中位数在 k=8 时已经相当稳，而 p95 在 k=8 时几乎就是「8 个样本
    的最大值」—— 对指数尾它的期望是 scale·H₈ ≈ 2.72·scale，而真值是
    scale·ln20 ≈ 3.00·scale，**系统性偏低**，且方差极大。

    后果是可复现的：用 k=8 过闸放行的链路，到终审用 k=16 复测就超 J_cap。
    实测中这是终审失败的首要原因，且它是测量方法的问题，不是放置的问题。
    取 k_gate = 32 让闸门判定站得住；p50 仍用 k_probe 采，探测预算只在闸门上多花。
    """

    # --- 求解器 ---
    beam_width: int = 12
    beam_width_front: int = 16
    prune_topk: int = 10
    """预剪枝 N*(v)：按 (M_v, 接入偏移 δ̂_v, 可用率 r_v) 取前 K。"""

    # --- 容量 (II.7) [启发式] ---
    theta: float = 0.8
    """折扣系数。§III.9 明确标注：0.8 是初值、无理论依据，经验区间 0.7–0.85。"""
    kappa_over: float = 0.3
    """前段超建系数 κ_over。"""
    n_standby: int = 1
    """备胎条数（Step 6 落选者转备胎）。"""

    # --- 两轮筛 (II.2) [启发式] ---
    max_hops_slack: int = 1
    """段的跳数相对 III.5.2 的整数下界最多可超出多少。

    这是一条**质量下限**，不是调参：分散环境下每多一跳即数十毫秒/token，一条
    比下界多 4 跳的段其延迟是主体的三倍，把它放进池子会直接毁掉均匀性目标。
    文档 II.2.1 对慢尾的处置是「救不回则剔除该节点、该段淘汰并上报缺口」——
    但间隙检测需要 ≥3 条同池段才跑得起来，单条池的段没人管。这条下限补上了
    那个缺口：建段时就地判定，超限即当作建不出，记入公平比缺口。

    """
    gap_theta1: float = 0.35
    gap_k: float = 1.5
    tighten_max_steps: int = 8
    tighten_eps_ms: float = 0.5

    # --- 求解器评分 [启发式] ---
    mu_ms: float = 40.0
    jitter_w: float = 0.2
    rent_nu_ms: float = 15.0
    good_gamma_ms: float = 25.0
    endpoint_w: float = 1.0
    """后段端点项的权重（Objective.endpoint_w）。置 0 复现文档的原始目标函数。"""

    seed: int = 0
