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

    # -- 逐层拆解（GB） -------------------------------------------------- #
    @property
    def expert_gb(self) -> float:
        """一个专家的权重量。SwiGLU 三个矩阵：w1 / w3 (d→f) + w2 (f→d)。"""
        return 3 * self.d_model * self.moe_intermediate * self.dtype_bytes / 1e9

    @property
    def attn_gb(self) -> float:
        """每层 attention 的权重量（含 GQA 的非对称 q/kv）。"""
        qo = 2 * self.d_model * (self.n_heads * self.head_dim)
        kv = 2 * self.d_model * (self.n_kv_heads * self.head_dim)
        return (qo + kv) * self.dtype_bytes / 1e9

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def layer_full_gb(self) -> float:
        return self.attn_gb + self.n_experts * self.expert_gb

    @property
    def total_gb(self) -> float:
        embed = 2 * self.vocab * self.d_model * self.dtype_bytes / 1e9
        return self.n_layers * self.layer_full_gb + embed

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
    )
    spec = ModelSpec(
        n_layers=info.n_layers,
        d_model=info.d_model,
        n_experts=info.n_experts,
        top_k=info.top_k,
        base_gb_per_layer=info.attn_gb,
        expert_gb=info.expert_gb,
        ctx_max=ctx_max,
        kv_bytes_per_elem=dtype_bytes,
        kv_dim=info.kv_dim,
    )
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
