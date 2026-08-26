"""从 HF `config.json` 得到规划器的 ModelSpec，并判断模型是否适合本方案。

**接真实模型之前先跑这一步。** 不是所有 MoE 都适合双段专家驻留 —— 关键是
**细粒度**：`n_experts / top_k` 这个比值决定了「一个 task 只用一小撮专家」这件事
是否成立。

    比值小（如 Mixtral 8 专家 / top-2 = 4）：每个 token 就用掉 1/4 的专家，
      一个 task 跑几十条语料下来几乎会碰到全部 8 个 —— 按覆盖率取的驻留集
      ≈ 全集，「只驻留子集」省不下内存，整套方案失去前提。

    比值大（如 Qwen3-30B-A3B 128 专家 / top-8 = 16）：单 token 只碰 6%，
      task 层面的聚集才有意义，驻留集能压到全集的 10–20%。

这一层不 import torch，也不下载权重 —— 只读 config.json 里的几个整数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .types import ModelSpec

__all__ = ["HFModelInfo", "model_spec_from_hf", "PRESETS", "granularity_verdict"]


@dataclass
class HFModelInfo:
    """从 config.json 读出来的原始事实 + 派生量。"""

    name: str
    n_layers: int
    n_experts: int
    top_k: int
    d_model: int
    moe_intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    dtype_bytes: int = 2

    # ---- 混合架构（Qwen3-Next）。非混合模型这些全是默认值，行为不变 ----
    layer_types: tuple[str, ...] = ()
    """逐层是 "linear_attention" 还是 "full_attention"。空 = 每层都是标准注意力。"""
    shared_intermediate: int = 0
    """共享专家的中间维。>0 表示每层多一个**恒定激活**的专家 —— 它不参与
    驻留集裁剪，承载该层的节点必须装它。"""
    lin_v_heads: int = 0
    lin_k_heads: int = 0
    lin_v_dim: int = 0
    lin_k_dim: int = 0
    conv_kernel: int = 0

    # -- 逐层拆解（GB） -------------------------------------------------- #
    @property
    def expert_gb(self) -> float:
        """一个专家的权重量。SwiGLU 三个矩阵：w1 / w3 (d→f) + w2 (f→d)。"""
        return 3 * self.d_model * self.moe_intermediate * self.dtype_bytes / 1e9

    @property
    def hybrid(self) -> bool:
        return bool(self.layer_types) and "linear_attention" in self.layer_types

    @property
    def shared_gb(self) -> float:
        """共享专家的权重量。没有共享专家时为 0。"""
        if not self.shared_intermediate:
            return 0.0
        return (3 * self.d_model * self.shared_intermediate + self.d_model) \
            * self.dtype_bytes / 1e9

    @property
    def attn_gb(self) -> float:
        """每层 attention 的权重量（含 GQA 的非对称 q/kv）。

        Qwen3-Next 的 q_proj **同时产出 query 与输出门**，宽度是 2 倍 ——
        这里按 `layer_types` 判断要不要算那一倍。
        """
        gate_mult = 2 if self.hybrid else 1
        qo = self.d_model * (self.n_heads * self.head_dim) * (1 + gate_mult)
        kv = 2 * self.d_model * (self.n_kv_heads * self.head_dim)
        return (qo + kv) * self.dtype_bytes / 1e9

    @property
    def linear_attn_gb(self) -> float:
        """一层 Gated DeltaNet 的权重量。非混合模型为 0。"""
        if not self.lin_v_heads:
            return 0.0
        key_dim = self.lin_k_dim * self.lin_k_heads
        val_dim = self.lin_v_dim * self.lin_v_heads
        n = (self.d_model * (2 * key_dim + 2 * val_dim)      # in_proj_qkvz
             + self.d_model * 2 * self.lin_v_heads           # in_proj_ba
             + (2 * key_dim + val_dim) * self.conv_kernel    # conv1d（深度可分离）
             + 2 * self.lin_v_heads                          # A_log + dt_bias
             + self.lin_v_dim                                # norm
             + val_dim * self.d_model)                       # out_proj
        return n * self.dtype_bytes / 1e9

    def mixer_gb(self, layer: int) -> float:
        """第 layer 层（1-based）的 token mixer 权重量。"""
        if not self.hybrid:
            return self.attn_gb
        return (self.linear_attn_gb
                if self.layer_types[layer - 1] == "linear_attention" else self.attn_gb)

    def kv_gb_at(self, layer: int, ctx: int) -> float:
        """第 layer 层的 KV/状态占用。

        **DeltaNet 层的状态是定长的** —— 一个 [v_heads, k_dim, v_dim] 的矩阵加一个
        conv 窗口，与上下文长度无关。3/4 的层是这种层，所以按「每层 KV 随 ctx
        线性增长」估 Qwen3-Next 会**系统性高估**，实际能放下更多层。
        """
        if self.hybrid and self.layer_types[layer - 1] == "linear_attention":
            state = self.lin_v_heads * self.lin_k_dim * self.lin_v_dim
            conv = (2 * self.lin_k_dim * self.lin_k_heads
                    + self.lin_v_dim * self.lin_v_heads) * self.conv_kernel
            return (state + conv) * 4 / 1e9        # 状态用 fp32 存
        return ctx * 2 * self.kv_dim * self.dtype_bytes / 1e9

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def layer_full_gb(self) -> float:
        """一层全装的权重量。混合模型取逐层平均（各层不同，见 `mixer_gb`）。"""
        moe = self.n_experts * self.expert_gb + self.shared_gb
        if not self.hybrid:
            return self.attn_gb + moe
        mix = sum(self.mixer_gb(l) for l in range(1, self.n_layers + 1)) / self.n_layers
        return mix + moe

    @property
    def total_gb(self) -> float:
        embed = 2 * self.vocab * self.d_model * self.dtype_bytes / 1e9
        moe = self.n_experts * self.expert_gb + self.shared_gb
        mixers = sum(self.mixer_gb(l) for l in range(1, self.n_layers + 1))
        return mixers + self.n_layers * moe + embed

    @property
    def granularity(self) -> float:
        """n_experts / top_k —— 本方案是否适用的第一判据。"""
        return self.n_experts / self.top_k

    def layer_gb(self, n_resident: int) -> float:
        """只驻留 n_resident 个专家时的每层权重量。"""
        return self.attn_gb + n_resident * self.expert_gb

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_layers} 层 × {self.n_experts} 专家 top-{self.top_k}"
            f"，d_model {self.d_model}，专家中间维 {self.moe_intermediate}"
            f"，GQA {self.n_heads}/{self.n_kv_heads} 头 × {self.head_dim}"
        )


# --------------------------------------------------------------------------- #
def _get(cfg: Mapping, *names, default=None):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    if default is None:
        raise KeyError(f"config.json 里找不到 {names}")
    return default


def _layer_types(cfg) -> list[str]:
    """逐层类型。非混合模型返回空列表（下游据此走原来的路径）。"""
    if cfg.get("layer_types"):
        return list(cfg["layer_types"])
    iv = cfg.get("full_attention_interval")
    if not iv:
        return []
    n = int(_get(cfg, "num_hidden_layers"))
    return ["full_attention" if (i + 1) % int(iv) == 0 else "linear_attention"
            for i in range(n)]


def model_spec_from_hf(
    cfg: Mapping | str | Path,
    *,
    name: str = "model",
    ctx_max: int = 4096,
    dtype_bytes: int = 2,
) -> tuple[ModelSpec, HFModelInfo]:
    """读 HF config.json → (规划器用的 ModelSpec, 原始信息)。

    字段名在各模型家族之间不统一（Mixtral 用 num_local_experts，Qwen3 用
    num_experts，OLMoE 用 num_experts），这里做了别名兼容。
    """
    if isinstance(cfg, (str, Path)):
        cfg = json.loads(Path(cfg).read_text(encoding="utf-8"))

    d_model = int(_get(cfg, "hidden_size"))
    n_heads = int(_get(cfg, "num_attention_heads"))
    head_dim = int(_get(cfg, "head_dim", default=d_model // n_heads))
    info = HFModelInfo(
        name=name,
        n_layers=int(_get(cfg, "num_hidden_layers")),
        n_experts=int(_get(cfg, "num_experts", "num_local_experts", "n_routed_experts")),
        top_k=int(_get(cfg, "num_experts_per_tok", "moe_top_k")),
        d_model=d_model,
        moe_intermediate=int(_get(cfg, "moe_intermediate_size", "intermediate_size")),
        n_heads=n_heads,
        n_kv_heads=int(_get(cfg, "num_key_value_heads", default=n_heads)),
        head_dim=head_dim,
        vocab=int(_get(cfg, "vocab_size")),
        dtype_bytes=dtype_bytes,
        layer_types=tuple(_layer_types(cfg)),
        shared_intermediate=int(_get(cfg, "shared_expert_intermediate_size", default=0)),
        lin_v_heads=int(_get(cfg, "linear_num_value_heads", default=0)),
        lin_k_heads=int(_get(cfg, "linear_num_key_heads", default=0)),
        lin_v_dim=int(_get(cfg, "linear_value_head_dim", default=0)),
        lin_k_dim=int(_get(cfg, "linear_key_head_dim", default=0)),
        conv_kernel=int(_get(cfg, "linear_conv_kernel_dim", default=0)),
    )
    # 混合模型的 KV 逐层不同，规划器的 ModelSpec 只吃一个标量 ——
    # 取**加权平均**而不是最大值：取最大会把 3/4 的定长状态层也按 full attention
    # 估，系统性高估到没法用；取平均则整段的总量是对的，逐层分配的误差在段内互相
    # 抵消（段是连续层区间，总会横跨两种层）。
    kv_per_layer = (sum(info.kv_gb_at(l, ctx_max) for l in range(1, info.n_layers + 1))
                    / info.n_layers) if info.hybrid else None
    spec = ModelSpec(
        n_layers=info.n_layers,
        d_model=info.d_model,
        n_experts=info.n_experts,
        top_k=info.top_k,
        # 混合模型：base 取逐层平均的 mixer（各层不同），并把共享专家算进去 ——
        # 共享专家每层都有、且**不参与裁剪**，漏了它内存判据会系统性偏乐观
        base_gb_per_layer=(
            sum(info.mixer_gb(l) for l in range(1, info.n_layers + 1)) / info.n_layers
            + info.shared_gb) if info.hybrid else info.attn_gb,
        expert_gb=info.expert_gb,
        ctx_max=ctx_max,
        kv_bytes_per_elem=dtype_bytes,
        kv_dim=info.kv_dim,
    )
    if kv_per_layer is not None:
        # ModelSpec 的 KV 是按 (ctx × kv_dim × bytes) 算的；混合模型的定长状态层
        # 塞不进那个公式，所以反解一个等效 kv_dim 让总量对上。
        object.__setattr__(spec, "kv_dim",
                           max(1, int(round(kv_per_layer * 1e9
                                            / (ctx_max * 2 * dtype_bytes)))))
    return spec, info


# --------------------------------------------------------------------------- #
def granularity_verdict(info: HFModelInfo) -> tuple[bool, str]:
    """本方案是否适用于这个模型。

    判据是 n_experts / top_k。这不是拍的：一个 task 的驻留集由「该 task 的激活
    质量集中在多少个专家上」决定，而单 token 就要碰 top_k 个。若 top_k 已经占了
    全集的相当比例，跑几十条语料下来的并集就接近全集。
    """
    g = info.granularity
    if g < 6:
        return False, (
            f"细粒度比 {g:.0f}（{info.n_experts} 专家 / top-{info.top_k}）**偏低**。"
            f"单 token 就用掉 {100/g:.0f}% 的专家，一个 task 的驻留集会接近全集，"
            f"「只驻留子集」省不下内存 —— 本方案的前提不成立。"
            f"该换细粒度 MoE（专家多、每个小）。"
        )
    if g < 12:
        return True, (
            f"细粒度比 {g:.0f}，勉强可用。驻留集大概能压到全集的 30–50%，"
            f"收益有限但方向对。"
        )
    return True, (
        f"细粒度比 {g:.0f}（单 token 只碰 {100/g:.1f}%），适合本方案 —— "
        f"驻留集通常能压到全集的 10–20%。"
    )


# --------------------------------------------------------------------------- #
# 常见 MoE 的 config 关键字段（数值取自各模型的 config.json）
PRESETS: dict[str, dict] = {
    "mixtral-8x7b": {
        "num_hidden_layers": 32, "num_local_experts": 8, "num_experts_per_tok": 2,
        "hidden_size": 4096, "intermediate_size": 14336,
        "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32000,
    },
    "olmoe-1b-7b": {
        "num_hidden_layers": 16, "num_experts": 64, "num_experts_per_tok": 8,
        "hidden_size": 2048, "intermediate_size": 1024,
        "num_attention_heads": 16, "num_key_value_heads": 16, "vocab_size": 50304,
    },
    "qwen3-30b-a3b": {
        "num_hidden_layers": 48, "num_experts": 128, "num_experts_per_tok": 8,
        "hidden_size": 2048, "moe_intermediate_size": 768,
        "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
        "vocab_size": 151936,
    },
}
