"""真实 MoE 的执行层 —— `PartialExpertMoEBlock` 的 torch 版。

与 numpy 的 toy 版（`runtime/model.py`）**保持同一套契约**：

    PartialExpertMoEBlock(cfg, layer, resident, weights).forward(x, cache) -> (y, MoEStats)
    SegmentModel(cfg, layer_experts).forward(req, x) -> (y, MoEStats)
    SegmentModel.drop_kv(req) / has_kv(req) / kv_tokens(req)

契约不变的意义：`tests/test_runtime.py` 里那批语义测试（只驻留子集、drop-expert
重归一、换绑时前段 KV 不动）原样适用于这一版 —— 换的是实现，不是行为。

为什么不能直接用 HF 的 MoE block
--------------------------------
主流实现把每层专家打成 fused 张量（`[E, d, f]`），加载器按完整 E 维走，
前向也是一个融合的 gating→dispatch→combine。这带来两个致命问题：

1. **加载器不认识专家子集** —— 我们要的是「只装第 20 层的 3、7、12 号专家」，
   fused 张量没有这个概念；
2. **拿不到 routing 决策** —— miss 检出（II.5 通道二）与 drop-expert 近似都需要
   知道「这个 token 选了哪几个专家、哪些不在本地」，融合 op 里看不到。

所以专家按 `ModuleDict` 逐个存，前向自己写。代价是比 fused kernel 慢；但方案自己
的账本里计算只占约 13%（分散环境下网络占九成），这个代价换来的是可行性。

当前支持 Qwen3-MoE 系（key 命名与 QK-RMSNorm 取自其 checkpoint 索引）。
换模型家族时只需加一个 adapter：key 命名 + 层内结构，其余不动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .model import MoEStats

__all__ = ["TorchModelConfig", "TorchPartialExpertMoEBlock", "TorchSegmentModel"]


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TorchModelConfig:
    n_layers: int
    n_experts: int
    top_k: int
    d_model: int
    moe_intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"

    @classmethod
    def from_hf(cls, cfg: Mapping) -> "TorchModelConfig":
        d = int(cfg["hidden_size"])
        h = int(cfg["num_attention_heads"])
        return cls(
            n_layers=int(cfg["num_hidden_layers"]),
            n_experts=int(cfg.get("num_experts", cfg.get("num_local_experts"))),
            top_k=int(cfg.get("num_experts_per_tok", cfg.get("moe_top_k", 2))),
            d_model=d,
            moe_intermediate=int(cfg.get("moe_intermediate_size", cfg.get("intermediate_size"))),
            n_heads=h,
            n_kv_heads=int(cfg.get("num_key_value_heads", h)),
            head_dim=int(cfg.get("head_dim", d // h)),
            vocab=int(cfg["vocab_size"]),
            rms_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            dtype=str(cfg.get("torch_dtype", "bfloat16")),
        )

    @property
    def torch_dtype(self):
        import torch

        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.dtype]


# --------------------------------------------------------------------------- #
def _rms_norm(x, weight, eps: float):
    import torch

    dt = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dt) * weight)


def _rotate_half(x):
    import torch

    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _rope(q, k, pos, head_dim: int, theta: float):
    """标准 RoPE。pos 是**绝对位置**，decode 时由 KV 长度给出偏移。"""
    import torch

    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device,
                                        dtype=torch.float32) / head_dim))
    fr = pos.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# --------------------------------------------------------------------------- #
class TorchPartialExpertMoEBlock:
    """一层 Qwen3-MoE。**只有 `resident` 里的专家会被 materialize。**

    weights 是这一层需要的张量（由 `weights.SelectiveLoader` 按 key 取来）。
    构造函数不做任何「先建全量再裁剪」的事 —— 不在 resident 里的专家，
    它的张量从头到尾没进过这个进程。
    """

    def __init__(
        self,
        cfg: TorchModelConfig,
        layer: int,
        resident: Iterable[int],
        weights: Mapping[str, "object"],
        *,
        device: str = "cpu",
    ):
        import torch

        self.cfg = cfg
        self.layer = int(layer)
        self.resident = frozenset(int(e) for e in resident)
        self.device = device
        if not self.resident:
            raise ValueError(f"layer {layer} 的驻留专家集为空")

        p = f"model.layers.{self.layer - 1}"   # 规划器 1-based → checkpoint 0-based

        def w(name: str):
            k = f"{p}.{name}"
            if k not in weights:
                raise KeyError(f"缺少权重 {k}")
            return weights[k]

        self.wq, self.wk, self.wv, self.wo = (
            w("self_attn.q_proj.weight"), w("self_attn.k_proj.weight"),
            w("self_attn.v_proj.weight"), w("self_attn.o_proj.weight"),
        )
        self.q_norm, self.k_norm = w("self_attn.q_norm.weight"), w("self_attn.k_norm.weight")
        self.ln_in, self.ln_post = (
            w("input_layernorm.weight"), w("post_attention_layernorm.weight"),
        )
        self.router = w("mlp.gate.weight")

        self._gate, self._up, self._down = {}, {}, {}
        for e in sorted(self.resident):
            self._gate[e] = w(f"mlp.experts.{e}.gate_proj.weight")
            self._up[e] = w(f"mlp.experts.{e}.up_proj.weight")
            self._down[e] = w(f"mlp.experts.{e}.down_proj.weight")

        self._resident_idx = torch.tensor(sorted(self.resident), device=device,
                                          dtype=torch.long)

    # -- 计量 -------------------------------------------------------------- #
    @property
    def resident_bytes(self) -> int:
        base = sum(t.numel() * t.element_size() for t in
                   (self.wq, self.wk, self.wv, self.wo, self.q_norm, self.k_norm,
                    self.ln_in, self.ln_post, self.router))
        exp = sum(t.numel() * t.element_size()
                  for d in (self._gate, self._up, self._down) for t in d.values())
        return base + exp

    @property
    def full_bytes(self) -> int:
        per = sum(t.numel() * t.element_size()
                  for t in (self._gate[min(self.resident)], self._up[min(self.resident)],
                            self._down[min(self.resident)]))
        base = sum(t.numel() * t.element_size() for t in
                   (self.wq, self.wk, self.wv, self.wo, self.q_norm, self.k_norm,
                    self.ln_in, self.ln_post, self.router))
        return base + per * self.cfg.n_experts

    # -- attention --------------------------------------------------------- #
    def _attend(self, x, cache: dict):
        import torch

        c = self.cfg
        T = x.shape[0]
        q = (x @ self.wq.T).view(T, c.n_heads, c.head_dim)
        k = (x @ self.wk.T).view(T, c.n_kv_heads, c.head_dim)
        v = (x @ self.wv.T).view(T, c.n_kv_heads, c.head_dim)

        # Qwen3 特有：q/k 各自过一次 RMSNorm（按 head_dim）再进 RoPE
        q = _rms_norm(q, self.q_norm, c.rms_eps)
        k = _rms_norm(k, self.k_norm, c.rms_eps)

        past = cache.get("k")
        offset = 0 if past is None else past.shape[0]
        pos = torch.arange(offset, offset + T, device=x.device)
        q, k = _rope(q, k, pos, c.head_dim, c.rope_theta)

        if past is not None:
            k = torch.cat([past, k], dim=0)
            v = torch.cat([cache["v"], v], dim=0)
        cache["k"], cache["v"] = k, v
        T_all = k.shape[0]

        rep = c.n_heads // c.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # [H, T, D]
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = (qh @ kh.transpose(1, 2)) / math.sqrt(c.head_dim)
        # 因果 mask：第 i 个新 token 只能看到前 offset+i 个
        idx = (torch.arange(T_all, device=x.device)[None, :]
               > (offset + torch.arange(T, device=x.device))[:, None])
        scores = scores.masked_fill(idx[None, :, :], torch.finfo(scores.dtype).min)
        out = (torch.softmax(scores.float(), dim=-1).to(scores.dtype) @ vh)
        return out.transpose(0, 1).reshape(T, c.n_heads * c.head_dim) @ self.wo.T

    # -- 前向 -------------------------------------------------------------- #
    def forward(self, x, cache: dict):
        """x: [T, d_model]。返回 (y, 本层统计)。"""
        import torch

        c = self.cfg
        h = x + self._attend(_rms_norm(x, self.ln_in, c.rms_eps), cache)
        z = _rms_norm(h, self.ln_post, c.rms_eps)
        T = z.shape[0]

        logits = z @ self.router.T                       # [T, E]
        probs = torch.softmax(logits.float(), dim=-1)
        topw, topi = torch.topk(probs, c.top_k, dim=-1)  # [T, k]
        if c.norm_topk_prob:
            topw = topw / topw.sum(-1, keepdim=True)

        here = torch.isin(topi, self._resident_idx)       # [T, k]
        kept = topw * here
        denom = kept.sum(-1, keepdim=True)
        alive = denom.squeeze(-1) > 0

        # ---- drop-expert 近似（II.5）：跳过缺失专家、门控重归一 ----
        # 文档标注它「运维近似，非无损」。miss_mass 记的就是被丢掉的门控质量。
        renorm = torch.where(denom > 0, kept / denom.clamp_min(1e-9),
                             torch.zeros_like(kept))

        out = torch.zeros_like(z)
        for e in sorted(self.resident):
            sel = (topi == e) & here
            if not sel.any():
                continue
            tok, slot = sel.nonzero(as_tuple=True)
            w = renorm[tok, slot].to(z.dtype)
            xi = z[tok]
            hi = torch.nn.functional.silu(xi @ self._gate[e].T) * (xi @ self._up[e].T)
            out.index_add_(0, tok, (hi @ self._down[e].T) * w[:, None])

        miss_tok = int((~here).any(-1).sum().item())
        miss_mass = float((topw * ~here).sum().item())
        hist = torch.zeros(c.n_experts, dtype=torch.float64, device=z.device)
        hist.index_add_(0, topi.reshape(-1), topw.reshape(-1).double())

        stats = MoEStats(
            hist=hist.cpu().numpy(),
            n_token_layer=T,
            miss_token_layer=miss_tok,
            miss_mass=miss_mass,
        )
        # top-k 全缺的 token：本层只走 attention 残差，不凭空造 FFN 输出
        out = out * alive[:, None].to(out.dtype)
        return h + out, stats


# --------------------------------------------------------------------------- #
class TorchSegmentModel:
    """一台节点上承载的层区间。KV 按 (req, layer) 分桶。

    换绑时后段整体 `drop_kv`、前段原地不动 —— 命题 III.7.1 的实现依据不变：
    前段的计算是对输入的精确 forward，识别是旁路统计、不进计算图。
    """

    def __init__(
        self,
        cfg: TorchModelConfig,
        layer_experts: Mapping[int, Sequence[int]],
        weights: Mapping[str, "object"],
        *,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.device = device
        self.layers = sorted(int(l) for l in layer_experts)
        self.blocks = {
            int(l): TorchPartialExpertMoEBlock(cfg, int(l), es, weights, device=device)
            for l, es in layer_experts.items()
        }
        self.embed = weights.get("model.embed_tokens.weight")
        self.final_norm = weights.get("model.norm.weight")
        self.lm_head = weights.get(
            "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        )
        self._kv: dict[str, dict[int, dict]] = {}

    # -- 计量 -------------------------------------------------------------- #
    @property
    def resident_bytes(self) -> int:
        return sum(b.resident_bytes for b in self.blocks.values())

    @property
    def full_bytes(self) -> int:
        return sum(b.full_bytes for b in self.blocks.values())

    # -- 前向 -------------------------------------------------------------- #
    def embed_tokens(self, ids: Sequence[int]):
        import torch

        if self.embed is None:
            raise RuntimeError("本节点没有词嵌入（只有前段的 head 需要）")
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=self.device)
        return self.embed[idx]

    def forward(self, req: str, x):
        kv = self._kv.setdefault(req, {})
        total = MoEStats.zeros(self.cfg.n_experts)
        h = x
        for l in self.layers:
            h, st = self.blocks[l].forward(h, kv.setdefault(l, {}))
            total = total.merge(st)
        return h, total

    def logits(self, h):
        """只有后段的 tail 会调 —— 最终 norm + 输出头。"""
        if self.final_norm is None or self.lm_head is None:
            raise RuntimeError("本节点没有输出头（只有后段的 tail 需要）")
        return _rms_norm(h, self.final_norm, self.cfg.rms_eps) @ self.lm_head.T

    # -- KV 生命周期 -------------------------------------------------------- #
    def drop_kv(self, req: str) -> bool:
        return self._kv.pop(req, None) is not None

    def has_kv(self, req: str) -> bool:
        return req in self._kv

    def kv_tokens(self, req: str) -> int:
        kv = self._kv.get(req)
        if not kv:
            return 0
        k = kv.get(self.layers[0], {}).get("k")
        return 0 if k is None else int(k.shape[0])

    def active_requests(self) -> list[str]:
        return sorted(self._kv)
