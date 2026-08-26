"""Qwen3-Next 的执行层 —— 混合注意力 + 超稀疏 MoE + 共享专家。

与 `torch_model.py`（Qwen3-MoE）**保持同一套契约**：

    SegmentModel(cfg, layer_experts, weights).forward(req, x) -> (y, MoEStats)
    .drop_kv(req) / .has_kv(req) / .embed_tokens(ids) / .logits(h) / .sample(h)

契约不变的意义：段内流水、跨段绕环、只驻留子集、drop-expert、画像采集、
时序埋点 —— 上面那一整套一行都不用改。换的只是「一层里面算什么」。

与 Qwen3-MoE 的三处结构性差异
-----------------------------
**1. 层是混合的。** `full_attention_interval=4` —— 每 4 层里 3 层是 Gated DeltaNet
（线性注意力），1 层是标准 attention。所以逐层的权重 key、参数量、KV 形态**都不同**，
「每层长一样」这个假设在这里不成立（规划器那边见 `hf_config.py` 的逐层记账）。

**2. 有共享专家。** 每层除了 top-k 路由的专家，还有一个**恒定激活**的
`shared_expert`，输出按 `sigmoid(shared_expert_gate(x))` 加权后叠加。
它对本方案的含义很直接：**共享专家必须每个后段节点都装**，不能参与裁剪 ——
它不属于「某个 task 的驻留集」，而是那一层的固定成分。好在它只有
`shared_expert_intermediate_size` 那么大（512，与单个专家同量级）。

**3. DeltaNet 的「KV」是固定大小的循环状态。** 不随上下文增长 —— 一个
`[num_v_heads, head_k_dim, head_v_dim]` 的矩阵加一个 conv 窗口，就这么大。
3/4 的层是这种层，所以本模型的 KV 内存**远小于**同规模的标准 MoE，
规划器按「每层 KV 随 ctx 线性增长」估会系统性高估。

细粒度比 51（512 专家 / top-10）
--------------------------------
单 token 只碰 2% 的专家 —— 比 Qwen3-30B-A3B 的 16 好得多。「后段只驻留子集」
在这个模型上的收益比在 Qwen3-30B 上大。这是本方案最想要的那种模型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .model import MoEStats


__all__ = ["NextModelConfig", "TorchNextSegmentModel", "layer_types_of"]


# --------------------------------------------------------------------------- #
def layer_types_of(cfg: Mapping) -> list[str]:
    """逐层是 linear_attention 还是 full_attention。

    优先用 config 里显式的 `layer_types`；没有就按 `full_attention_interval`
    推 —— HF 的规则是「第 i 层当 (i+1) % interval == 0 时是 full attention」，
    即 interval=4 时第 3、7、11… 层（0-based）是标准注意力。
    """
    n = int(cfg["num_hidden_layers"])
    if cfg.get("layer_types"):
        return list(cfg["layer_types"])
    iv = int(cfg.get("full_attention_interval", 4))
    return ["full_attention" if (i + 1) % iv == 0 else "linear_attention"
            for i in range(n)]


@dataclass(frozen=True)
class NextModelConfig:
    n_layers: int
    n_experts: int
    top_k: int
    d_model: int
    moe_intermediate: int
    shared_intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    layer_types: tuple[str, ...]
    # ---- DeltaNet ----
    lin_v_heads: int
    lin_k_heads: int
    lin_v_dim: int
    lin_k_dim: int
    conv_kernel: int
    rms_eps: float = 1e-6
    rope_theta: float = 10_000.0
    partial_rotary: float = 0.25
    """**只有前 head_dim × 这个比例的维度参与旋转**，其余原样透传。

    Qwen3-Next 是 0.25 —— head_dim 256 里只有 64 维带位置信息。照 Qwen3-MoE
    的写法（全维旋转）接过来不会报错，只是所有位置编码都错。"""
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    dtype: str = "bfloat16"

    @classmethod
    def from_hf(cls, cfg: Mapping) -> "NextModelConfig":
        d = int(cfg["hidden_size"])
        h = int(cfg["num_attention_heads"])
        return cls(
            n_layers=int(cfg["num_hidden_layers"]),
            n_experts=int(cfg["num_experts"]),
            top_k=int(cfg["num_experts_per_tok"]),
            d_model=d,
            moe_intermediate=int(cfg["moe_intermediate_size"]),
            shared_intermediate=int(cfg.get("shared_expert_intermediate_size", 0)),
            n_heads=h,
            n_kv_heads=int(cfg.get("num_key_value_heads", h)),
            head_dim=int(cfg.get("head_dim", d // h)),
            vocab=int(cfg["vocab_size"]),
            layer_types=tuple(layer_types_of(cfg)),
            lin_v_heads=int(cfg["linear_num_value_heads"]),
            lin_k_heads=int(cfg["linear_num_key_heads"]),
            lin_v_dim=int(cfg["linear_value_head_dim"]),
            lin_k_dim=int(cfg["linear_key_head_dim"]),
            conv_kernel=int(cfg.get("linear_conv_kernel_dim", 4)),
            rms_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            # rope_theta 可能在顶层，也可能藏在 rope_scaling / rope_parameters 里 ——
            # transformers 读的是后者，两处不一致时以后者为准
            rope_theta=float(
                (cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}).get(
                    "rope_theta", cfg.get("rope_theta", 10_000.0))),
            partial_rotary=float(
                (cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}).get(
                    "partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0))),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            dtype=str(cfg.get("torch_dtype", "bfloat16")),
        )

    @property
    def torch_dtype(self):
        import torch

        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.dtype]

    def kind(self, layer: int) -> str:
        """1-based 层号 → 该层类型。"""
        return self.layer_types[layer - 1]

    @property
    def key_dim(self) -> int:
        return self.lin_k_dim * self.lin_k_heads

    @property
    def value_dim(self) -> int:
        return self.lin_v_dim * self.lin_v_heads


# --------------------------------------------------------------------------- #
def _rms_norm(x, weight, eps: float):
    """**零中心 RMSNorm** —— Qwen3-Next 的 `Qwen3NextRMSNorm`。

    与 Qwen3-MoE 的差别只有一处，但足以毁掉一切：这里乘的是 `(1 + weight)`，
    而不是 `weight`。对应地，它的权重初始化为 **0** 而非 1 —— 也就是说
    「什么都不做」在这里写作 weight=0。

    照 Qwen3-MoE 的写法（`* weight`）接过来不会报错、形状全对，只是每一个
    归一化都算错。这是本文件里最该被逐元素比对钉住的一行。
    """
    import torch

    dt = x.dtype
    x32 = x.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * (1.0 + weight.float())).to(dt)


def _rotate_half(x):
    import torch

    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _partial_rope(q, k, pos, head_dim: int, theta: float, factor: float):
    """部分旋转的 RoPE：只旋转前 `int(head_dim * factor)` 维，其余原样接回去。

    Qwen3-Next 的 factor 是 0.25。全维旋转（factor=1）就退化成 Qwen3-MoE 的写法，
    所以这一个函数覆盖两种模型 —— 但**默认值必须是模型自己的**，
    默认 1.0 再让 Qwen3-Next 忘了传，就会静默地全维旋转。
    """
    import torch

    dim = int(head_dim * factor)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device,
                                        dtype=torch.float32) / dim))
    fr = pos.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos()[:, None, :].to(q.dtype), emb.sin()[:, None, :].to(q.dtype)
    qr, qp = q[..., :dim], q[..., dim:]
    kr, kp = k[..., :dim], k[..., dim:]
    q_out = torch.cat([qr * cos + _rotate_half(qr) * sin, qp], dim=-1)
    k_out = torch.cat([kr * cos + _rotate_half(kr) * sin, kp], dim=-1)
    return q_out, k_out


def _l2norm(x, eps: float = 1e-6):
    import torch

    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def _rms_norm_gated(x, weight, gate, eps: float):
    """先归一化再乘 silu(gate) —— Qwen3-Next 的 `RMSNormGated`。

    注意顺序：**norm 在 gate 之前**。反过来写出来的东西形状一样、不报错、
    数值全错，正是那种只能靠逐元素比对抓住的 bug。
    """
    import torch

    dt = x.dtype
    x32 = x.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    # `RMSNormGated` 用的是 `weight *`（**不是** 1+weight）—— 与上面那个不同。
    # 同一个模型里两种 norm 的约定不一样，这种地方只能照抄，不能推理。
    x32 = weight * x32.to(dt)
    return (x32 * torch.nn.functional.silu(gate.to(torch.float32))).to(dt)


# --------------------------------------------------------------------------- #
class TorchGatedDeltaNet:
    """线性注意力层。状态是**固定大小**的，不随上下文增长。

    递推逐字对齐 transformers 的 `torch_recurrent_gated_delta_rule`：

        S ← S · exp(g_t)                       遗忘门
        Δ ← (v_t − Σ_k S·k_t) · β_t            delta 规则：只写「新信息」
        S ← S + k_t ⊗ Δ
        out_t ← Σ_k S · q_t

    HF 另有一个分块版（`torch_chunk_gated_delta_rule`）做加速，两者等价；
    这里用逐步版，因为它就是定义本身，读得懂也验得了。
    """

    def __init__(self, cfg: NextModelConfig, layer: int, w, *, device: str = "cpu"):
        self.cfg = cfg
        self.layer = layer
        self.device = device
        self.conv1d = w("linear_attn.conv1d.weight")
        self.A_log = w("linear_attn.A_log")
        self.dt_bias = w("linear_attn.dt_bias")
        self.in_qkvz = w("linear_attn.in_proj_qkvz.weight")
        self.in_ba = w("linear_attn.in_proj_ba.weight")
        self.norm_w = w("linear_attn.norm.weight")
        self.out_proj = w("linear_attn.out_proj.weight")

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.conv1d, self.A_log, self.dt_bias, self.in_qkvz,
                    self.in_ba, self.norm_w, self.out_proj))

    def _split(self, qkvz, ba):
        """把打包的投影拆成 q/k/v/z/b/a —— 布局取自 HF 的 fix_query_key_value_ordering。"""
        import torch

        c = self.cfg
        T = qkvz.shape[0]
        ratio = c.lin_v_heads // c.lin_k_heads
        qkvz = qkvz.view(T, c.lin_k_heads, 2 * c.lin_k_dim + 2 * ratio * c.lin_v_dim)
        ba = ba.view(T, c.lin_k_heads, 2 * ratio)
        q, k, v, z = torch.split(
            qkvz, [c.lin_k_dim, c.lin_k_dim, ratio * c.lin_v_dim, ratio * c.lin_v_dim], dim=2)
        b, a = torch.split(ba, [ratio, ratio], dim=2)
        v = v.reshape(T, -1, c.lin_v_dim)
        z = z.reshape(T, -1, c.lin_v_dim)
        return q, k, v, z, b.reshape(T, c.lin_v_heads), a.reshape(T, c.lin_v_heads)

    def forward(self, x, cache: dict):
        import torch

        c = self.cfg
        T = x.shape[0]
        q, k, v, z, b, a = self._split(x @ self.in_qkvz.T, x @ self.in_ba.T)
        q = q.reshape(T, -1)
        k = k.reshape(T, -1)
        v = v.reshape(T, -1)

        # ---- 因果深度可分离卷积（窗口 = conv_kernel）----
        mixed = torch.cat((q, k, v), dim=-1)                     # [T, conv_dim]
        pad = cache.get("conv")
        if pad is None:
            pad = torch.zeros(c.conv_kernel - 1, mixed.shape[1],
                              dtype=mixed.dtype, device=mixed.device)
        seq = torch.cat([pad, mixed], dim=0)                     # [K-1+T, C]
        cache["conv"] = seq[-(c.conv_kernel - 1):] if c.conv_kernel > 1 else pad
        wgt = self.conv1d.squeeze(1)                             # [C, K]
        # cols[t, ch, j] = seq[t+j, ch] —— 因果窗口展开成显式的一维
        cols = torch.stack([seq[i:i + T] for i in range(c.conv_kernel)], dim=-1)
        mixed = torch.nn.functional.silu((cols * wgt[None, :, :]).sum(-1))

        q, k, v = torch.split(mixed, [c.key_dim, c.key_dim, c.value_dim], dim=-1)
        q = q.reshape(T, -1, c.lin_k_dim)
        k = k.reshape(T, -1, c.lin_k_dim)
        v = v.reshape(T, -1, c.lin_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(
            a.float() + self.dt_bias)
        ratio = c.lin_v_heads // c.lin_k_heads
        if ratio > 1:
            q = q.repeat_interleave(ratio, dim=1)
            k = k.repeat_interleave(ratio, dim=1)

        # ---- gated delta 递推（逐位对齐 HF 的参考实现）----
        q32, k32 = _l2norm(q.float()), _l2norm(k.float())
        v32, beta32, g32 = v.float(), beta.float(), g.float()
        q32 = q32 * (c.lin_k_dim ** -0.5)
        S = cache.get("state")
        if S is None:
            S = torch.zeros(c.lin_v_heads, c.lin_k_dim, c.lin_v_dim,
                            dtype=torch.float32, device=x.device)
        out = torch.zeros(T, c.lin_v_heads, c.lin_v_dim,
                          dtype=torch.float32, device=x.device)
        for t in range(T):
            S = S * g32[t].exp().unsqueeze(-1).unsqueeze(-1)
            kv = (S * k32[t].unsqueeze(-1)).sum(dim=-2)
            delta = (v32[t] - kv) * beta32[t].unsqueeze(-1)
            S = S + k32[t].unsqueeze(-1) * delta.unsqueeze(-2)
            out[t] = (S * q32[t].unsqueeze(-1)).sum(dim=-2)
        cache["state"] = S

        y = _rms_norm_gated(out.to(x.dtype).reshape(-1, c.lin_v_dim),
                            self.norm_w, z.reshape(-1, c.lin_v_dim), c.rms_eps)
        return y.reshape(T, c.value_dim) @ self.out_proj.T


# --------------------------------------------------------------------------- #
class TorchNextAttention:
    """标准注意力层。与 Qwen3-MoE 的差别：**q_proj 同时产出 query 与输出门**。

    `q_proj` 的输出宽度是 `n_heads * head_dim * 2`，前一半是 query、后一半是
    gate；attention 算完之后按 `sigmoid(gate)` 逐元素缩放再过 o_proj。
    照 Qwen3-MoE 的写法接会把 query 读成两倍宽 —— 形状就对不上，会当场报错，
    这算是幸运的那种错。
    """

    def __init__(self, cfg: NextModelConfig, layer: int, w, *, device: str = "cpu"):
        self.cfg = cfg
        self.layer = layer
        self.device = device
        self.wq = w("self_attn.q_proj.weight")
        self.wk = w("self_attn.k_proj.weight")
        self.wv = w("self_attn.v_proj.weight")
        self.wo = w("self_attn.o_proj.weight")
        self.q_norm = w("self_attn.q_norm.weight")
        self.k_norm = w("self_attn.k_norm.weight")

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.wq, self.wk, self.wv, self.wo, self.q_norm, self.k_norm))

    def forward(self, x, cache: dict):
        import torch

        c = self.cfg
        T = x.shape[0]
        qg = (x @ self.wq.T).view(T, -1, c.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = _rms_norm(q.reshape(T, c.n_heads, c.head_dim), self.q_norm, c.rms_eps)
        k = _rms_norm((x @ self.wk.T).view(T, c.n_kv_heads, c.head_dim),
                      self.k_norm, c.rms_eps)
        v = (x @ self.wv.T).view(T, c.n_kv_heads, c.head_dim)

        past = cache.get("k")
        offset = 0 if past is None else past.shape[0]
        pos = torch.arange(offset, offset + T, device=x.device)
        q, k = _partial_rope(q, k, pos, c.head_dim, c.rope_theta, c.partial_rotary)
        if past is not None:
            k = torch.cat([past, k], dim=0)
            v = torch.cat([cache["v"], v], dim=0)
        cache["k"], cache["v"] = k, v
        T_all = k.shape[0]

        rep = c.n_heads // c.n_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = (qh @ kh.transpose(1, 2)) / math.sqrt(c.head_dim)
        idx = (torch.arange(T_all, device=x.device)[None, :]
               > (offset + torch.arange(T, device=x.device))[:, None])
        scores = scores.masked_fill(idx[None, :, :], torch.finfo(scores.dtype).min)
        o = (torch.softmax(scores.float(), dim=-1).to(scores.dtype) @ vh)
        o = o.transpose(0, 1).reshape(T, c.n_heads * c.head_dim)
        return (o * torch.sigmoid(gate)) @ self.wo.T


# --------------------------------------------------------------------------- #
class TorchNextMoE:
    """超稀疏 MoE + 共享专家。**只有 `resident` 里的路由专家会被 materialize。**

    共享专家不参与裁剪 —— 它对每个 token 都激活，是这一层的固定成分而不是
    某个 task 的驻留集。所以每个承载该层的节点都得装它。
    """

    def __init__(self, cfg: NextModelConfig, layer: int, resident: Iterable[int],
                 w, *, device: str = "cpu", miss_policy: str = "drop"):
        import torch

        self.cfg = cfg
        self.layer = layer
        self.device = device
        self.resident = frozenset(int(e) for e in resident)
        if not self.resident:
            raise ValueError(f"layer {layer} 的驻留专家集为空")
        if miss_policy not in ("drop", "drop_noscale", "local_topk"):
            raise ValueError(f"未知的 miss_policy {miss_policy!r}")
        self.miss_policy = miss_policy

        self.router = w("mlp.gate.weight")
        self._gate, self._up, self._down = {}, {}, {}
        for e in sorted(self.resident):
            self._gate[e] = w(f"mlp.experts.{e}.gate_proj.weight")
            self._up[e] = w(f"mlp.experts.{e}.up_proj.weight")
            self._down[e] = w(f"mlp.experts.{e}.down_proj.weight")

        self.has_shared = cfg.shared_intermediate > 0
        if self.has_shared:
            self.sh_gate = w("mlp.shared_expert.gate_proj.weight")
            self.sh_up = w("mlp.shared_expert.up_proj.weight")
            self.sh_down = w("mlp.shared_expert.down_proj.weight")
            self.sh_w = w("mlp.shared_expert_gate.weight")
        self._resident_idx = torch.tensor(sorted(self.resident), device=device,
                                          dtype=torch.long)

    @property
    def bytes(self) -> int:
        n = self.router.numel() * self.router.element_size()
        n += sum(t.numel() * t.element_size()
                 for d in (self._gate, self._up, self._down) for t in d.values())
        if self.has_shared:
            n += sum(t.numel() * t.element_size() for t in
                     (self.sh_gate, self.sh_up, self.sh_down, self.sh_w))
        return n

    @property
    def full_bytes(self) -> int:
        e0 = min(self.resident)
        per = sum(t.numel() * t.element_size()
                  for t in (self._gate[e0], self._up[e0], self._down[e0]))
        return self.bytes - len(self.resident) * per + self.cfg.n_experts * per

    def forward(self, z):
        import torch

        c = self.cfg
        T = z.shape[0]
        probs = torch.softmax((z @ self.router.T).float(), dim=-1)

        # 统计口径恒按**全量**路由 —— 与 torch_model.py 同一条纪律
        topw, topi = torch.topk(probs, c.top_k, dim=-1)
        if c.norm_topk_prob:
            topw = topw / topw.sum(-1, keepdim=True)
        here = torch.isin(topi, self._resident_idx)
        miss_tok = int((~here).any(-1).sum().item())
        miss_mass = float((topw * ~here).sum().item())
        hist = torch.zeros(c.n_experts, dtype=torch.float64, device=z.device)
        hist.index_add_(0, topi.reshape(-1), topw.reshape(-1).double())

        if self.miss_policy == "local_topk":
            neg = torch.finfo(probs.dtype).min
            mask = torch.zeros(c.n_experts, dtype=torch.bool, device=z.device)
            mask[self._resident_idx] = True
            local = probs.masked_fill(~mask[None, :], neg)
            kk = min(c.top_k, len(self.resident))
            use_w, use_i = torch.topk(local, kk, dim=-1)
            use_w = use_w / use_w.sum(-1, keepdim=True).clamp_min(1e-9)
            use_here = torch.ones_like(use_i, dtype=torch.bool)
            alive = torch.ones(T, dtype=torch.bool, device=z.device)
        elif self.miss_policy == "drop_noscale":
            use_w = topw * here
            alive = use_w.sum(-1) > 0
            use_i, use_here = topi, here
        else:
            kept = topw * here
            den = kept.sum(-1, keepdim=True)
            alive = den.squeeze(-1) > 0
            use_w = torch.where(den > 0, kept / den.clamp_min(1e-9),
                                torch.zeros_like(kept))
            use_i, use_here = topi, here

        out = torch.zeros_like(z)
        for e in sorted(self.resident):
            sel = (use_i == e) & use_here
            if not sel.any():
                continue
            tok, slot = sel.nonzero(as_tuple=True)
            wgt = use_w[tok, slot].to(z.dtype)
            xi = z[tok]
            hi = torch.nn.functional.silu(xi @ self._gate[e].T) * (xi @ self._up[e].T)
            out.index_add_(0, tok, (hi @ self._down[e].T) * wgt[:, None])
        out = out * alive[:, None].to(out.dtype)

        # 共享专家：恒定激活，不受驻留集与补救策略影响
        if self.has_shared:
            sh = torch.nn.functional.silu(z @ self.sh_gate.T) * (z @ self.sh_up.T)
            sh = sh @ self.sh_down.T
            out = out + torch.sigmoid(z @ self.sh_w.T) * sh

        stats = MoEStats(hist=hist.cpu().numpy(), n_token_layer=T,
                         miss_token_layer=miss_tok, miss_mass=miss_mass)
        return out, stats


# --------------------------------------------------------------------------- #
class TorchNextBlock:
    """一层 = token mixer（DeltaNet 或 attention）+ MoE。"""

    def __init__(self, cfg: NextModelConfig, layer: int, resident: Iterable[int],
                 weights: Mapping[str, "object"], *, device: str = "cpu",
                 miss_policy: str = "drop"):
        self.cfg = cfg
        self.layer = int(layer)
        self.kind = cfg.kind(self.layer)
        p = f"model.layers.{self.layer - 1}"     # 规划器 1-based → checkpoint 0-based

        def w(name: str):
            k = f"{p}.{name}"
            if k not in weights:
                raise KeyError(f"缺少权重 {k}")
            return weights[k]

        self.ln_in = w("input_layernorm.weight")
        self.ln_post = w("post_attention_layernorm.weight")
        self.mixer = (TorchGatedDeltaNet(cfg, self.layer, w, device=device)
                      if self.kind == "linear_attention"
                      else TorchNextAttention(cfg, self.layer, w, device=device))
        self.moe = TorchNextMoE(cfg, self.layer, resident, w, device=device,
                                miss_policy=miss_policy)
        self.resident = self.moe.resident

    @property
    def resident_bytes(self) -> int:
        return (self.mixer.bytes + self.moe.bytes
                + sum(t.numel() * t.element_size()
                      for t in (self.ln_in, self.ln_post)))

    @property
    def full_bytes(self) -> int:
        return (self.mixer.bytes + self.moe.full_bytes
                + sum(t.numel() * t.element_size()
                      for t in (self.ln_in, self.ln_post)))

    def forward(self, x, cache: dict):
        c = self.cfg
        h = x + self.mixer.forward(_rms_norm(x, self.ln_in, c.rms_eps), cache)
        y, st = self.moe.forward(_rms_norm(h, self.ln_post, c.rms_eps))
        return h + y, st


# --------------------------------------------------------------------------- #
class TorchNextSegmentModel:
    """一台节点上承载的层区间。契约与 `TorchSegmentModel` 完全相同。

    `_kv` 这个名字沿用了，但对 DeltaNet 层它装的是**固定大小的循环状态 +
    conv 窗口**，不随上下文增长。「换绑时后段丢 KV、前段不动」这条语义不变 ——
    丢的是同一个按 req 分桶的字典。
    """

    def __init__(self, cfg: NextModelConfig, layer_experts: Mapping[int, Sequence[int]],
                 weights: Mapping[str, "object"], *, device: str = "cpu",
                 miss_policy: str = "drop"):
        self.cfg = cfg
        self.device = device
        self.miss_policy = miss_policy
        self.layers = sorted(int(l) for l in layer_experts)
        self.blocks = {
            int(l): TorchNextBlock(cfg, int(l), es, weights, device=device,
                                   miss_policy=miss_policy)
            for l, es in layer_experts.items()
        }
        self.embed = weights.get("model.embed_tokens.weight")
        self.final_norm = weights.get("model.norm.weight")
        self.lm_head = weights.get(
            "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight")
        self._kv: dict[str, dict[int, dict]] = {}
        self.profiler = None

    def enable_profiling(self) -> None:
        from .profile import LayerProfiler

        self.profiler = LayerProfiler(self.cfg.n_experts)

    # -- 计量 -------------------------------------------------------------- #
    @property
    def resident_bytes(self) -> int:
        return sum(b.resident_bytes for b in self.blocks.values())

    @property
    def full_bytes(self) -> int:
        return sum(b.full_bytes for b in self.blocks.values())

    # -- 前向 -------------------------------------------------------------- #
    def _coerce(self, x):
        import torch

        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.cfg.torch_dtype)
        arr = np.array(x, dtype=np.float32, copy=True, order="C")
        return torch.from_numpy(arr).to(device=self.device, dtype=self.cfg.torch_dtype)

    def embed_tokens(self, ids: Sequence[int]):
        import torch

        if self.embed is None:
            raise RuntimeError("本节点没有词嵌入（只有前段的 head 需要）")
        return self.embed[torch.as_tensor(list(ids), dtype=torch.long,
                                          device=self.device)]

    def forward(self, req: str, x):
        x = self._coerce(x)
        kv = self._kv.setdefault(req, {})
        total = MoEStats.zeros(self.cfg.n_experts)
        h = x
        for l in self.layers:
            h, st = self.blocks[l].forward(h, kv.setdefault(l, {}))
            if self.profiler is not None:
                self.profiler.record(l, st.hist, st.n_token_layer)
            total = total.merge(st)
        return h, total

    def logits(self, h):
        if self.final_norm is None or self.lm_head is None:
            raise RuntimeError("本节点没有输出头（只有后段的 tail 需要）")
        return _rms_norm(h, self.final_norm, self.cfg.rms_eps) @ self.lm_head.T

    def sample(self, h, *, temperature: float = 0.0, seed: int = 0) -> int:
        import torch

        lg = self.logits(h[-1:])[0].float()
        if temperature <= 0:
            return int(torch.argmax(lg).item())
        g = torch.Generator(device="cpu").manual_seed(seed)
        return int(torch.multinomial(torch.softmax(lg.cpu() / temperature, -1), 1,
                                     generator=g).item())

    # -- KV 生命周期 -------------------------------------------------------- #
    def drop_kv(self, req: str) -> bool:
        return self._kv.pop(req, None) is not None

    def has_kv(self, req: str) -> bool:
        return req in self._kv

    def kv_tokens(self, req: str) -> int:
        kv = self._kv.get(req)
        if not kv:
            return 0
        for l in self.layers:
            k = kv.get(l, {}).get("k")
            if k is not None:
                return int(k.shape[0])
        return 0                       # 全是 DeltaNet 层：状态是定长的，无「token 数」

    def active_requests(self) -> list[str]:
        return sorted(self._kv)
