"""safetensors 选择性加载 —— 整套方案里最关键的一个工程原语。

方案成立的前提是「后段每层只装 n_{u,l} 个专家」。这不是优化，是可行性问题：
Qwen3-30B-A3B 每层 128 个专家共 1.2GB，48 层 61GB，一台 16GB 的机器装不下；
但只装 20% 的专家时每层 283MB，35 层的后段只要 10.2GB —— 放得下。

**为什么必须是 safetensors**：它是逐张量的，头部有一张 {key: (dtype, shape, 偏移)}
的表，可以只 mmap 需要的那几段。`model.layers.20.mlp.experts.7.gate_proj.weight`
是一个独立的 key，读它不需要碰其它 127 个专家。PyTorch 的 `.bin`（pickle）做不到
这件事 —— 那是一整个序列化流，要么全反序列化要么不读。

**为什么不用 accelerate**：`load_checkpoint_and_dispatch` 是整模块加载，它不认识
「只加载第 20 层的 3、7、12 号专家」；`device_map="auto"` 的自动分层更是本方案要
替换掉的东西 —— 放置由规划器决定，写在 `NodePlan` 里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = ["qwen3_next_keys", "KeyPlan", "qwen_moe_keys", "WeightIndex", "SelectiveLoader", "LoadReport"]


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KeyPlan:
    """本节点需要哪些张量。直接由 `NodePlan.layers` 翻译而来。"""

    layer_experts: Mapping[int, Sequence[int]]
    """层号（1-based，与规划器一致）→ 该层要驻留的专家 id。"""
    with_embed: bool = False
    """是否需要词嵌入（前段的 head 需要）。"""
    with_lm_head: bool = False
    """是否需要输出头与最终 norm（后段的 tail 需要）。"""


def cuda_state() -> dict:
    """CUDA 现在能不能用，以及不能用时**是哪一种**不能用。

    `torch.cuda.is_available()` 为真也可能在真正分配时才炸（驱动刚被卸载、
    GPU 被别人独占、ECC 复位中）—— 所以光问不够，要真摸一下。

    「CUDA unknown error」最常见的成因是驱动反复加载卸载：没开持久化模式时，
    GPU 一空闲就掉驱动，下一个进程再初始化就撞上竞态。
    """
    import torch

    if not torch.cuda.is_available():
        return {"ok": False, "why": "torch.cuda.is_available() 为假"}
    n = torch.cuda.device_count()
    if n == 0:
        return {"ok": False, "why": "可见 GPU 数为 0"}
    try:
        torch.zeros(8, device="cuda:0")     # 光问不够，真分配一次
    except Exception as e:
        return {"ok": False, "why": f"分配失败 {type(e).__name__}: {e}"[:200]}
    return {"ok": True, "n": n, "name": torch.cuda.get_device_name(0)}


def qwen_moe_keys(plan: KeyPlan, *, tie_word_embeddings: bool = False) -> set[str]:
    """Qwen3-MoE 的 key 命名 → 本节点需要的 key 集合。

    命名取自 `model.safetensors.index.json`（Qwen/Qwen3-30B-A3B）：

        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.self_attn.{q,k}_norm.weight     ← Qwen3 特有的 QK-RMSNorm
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.mlp.gate.weight                 ← 路由器
        model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight
        model.embed_tokens.weight / model.norm.weight / lm_head.weight

    注意规划器的层号是 1-based（文档第〇部分的层区间口径），checkpoint 是 0-based，
    这里做转换。搞错会静默加载错层 —— 所以只在这一处转，别处不碰。
    """
    keys: set[str] = set()
    for layer, experts in plan.layer_experts.items():
        i = int(layer) - 1                      # 1-based → 0-based
        p = f"model.layers.{i}"
        keys |= {
            f"{p}.self_attn.q_proj.weight",
            f"{p}.self_attn.k_proj.weight",
            f"{p}.self_attn.v_proj.weight",
            f"{p}.self_attn.o_proj.weight",
            f"{p}.self_attn.q_norm.weight",
            f"{p}.self_attn.k_norm.weight",
            f"{p}.input_layernorm.weight",
            f"{p}.post_attention_layernorm.weight",
            f"{p}.mlp.gate.weight",
        }
        for e in experts:
            keys |= {
                f"{p}.mlp.experts.{int(e)}.gate_proj.weight",
                f"{p}.mlp.experts.{int(e)}.up_proj.weight",
                f"{p}.mlp.experts.{int(e)}.down_proj.weight",
            }
    if plan.with_embed:
        keys.add("model.embed_tokens.weight")
    if plan.with_lm_head:
        keys.add("model.norm.weight")
        keys.add("model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight")
    return keys


# --------------------------------------------------------------------------- #
@dataclass
class LoadReport:
    n_keys: int
    bytes_loaded: int
    bytes_total: int
    shards_opened: int
    shards_total: int
    missing: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.bytes_loaded / self.bytes_total if self.bytes_total else 0.0

    def __str__(self) -> str:
        return (
            f"加载 {self.n_keys} 个张量 / {self.bytes_loaded/1e9:.2f}GB"
            f"（全量 {self.bytes_total/1e9:.2f}GB，{self.fraction:.1%}）；"
            f"打开分片 {self.shards_opened}/{self.shards_total}"
            + (f"；缺 {len(self.missing)} 个 key" if self.missing else "")
        )


_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


class WeightIndex:
    """读 checkpoint 目录的分片索引，回答「哪个 key 在哪个文件、多大」。

    支持两种布局：分片（有 model.safetensors.index.json）与单文件
    （只有 model.safetensors）。真实的大模型都是前者。
    """

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map: dict[str, str] = json.loads(
                idx.read_text(encoding="utf-8")
            )["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"{self.dir} 下既没有 model.safetensors.index.json 也没有 "
                    f"model.safetensors —— 权重必须是 safetensors 格式，"
                    f"pickle 的 .bin 不支持逐张量选择性加载"
                )
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}

        self.shards = sorted(set(self.weight_map.values()))
        self._sizes: dict[str, int] | None = None

    # -- 尺寸（从每个分片的头部读，不加载数据） ---------------------------- #
    def sizes(self) -> dict[str, int]:
        """每个 key 的字节数。只读 safetensors 头部，不碰张量数据。"""
        if self._sizes is not None:
            return self._sizes
        out: dict[str, int] = {}
        for shard in self.shards:
            path = self.dir / shard
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n))
            for k, meta in header.items():
                if k == "__metadata__":
                    continue
                s, e = meta["data_offsets"]
                out[k] = e - s
        self._sizes = out
        return out

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes().values())

    def shards_for(self, keys: Iterable[str]) -> dict[str, list[str]]:
        """把要的 key 按分片分组 —— 每个分片只打开一次。"""
        by: dict[str, list[str]] = {}
        for k in keys:
            shard = self.weight_map.get(k)
            if shard is not None:
                by.setdefault(shard, []).append(k)
        return by


# --------------------------------------------------------------------------- #
class SelectiveLoader:
    """只把点名的张量读进来。

    用法：
        idx = WeightIndex("/models/Qwen3-30B-A3B")
        loader = SelectiveLoader(idx)
        tensors, report = loader.load(qwen_moe_keys(plan), device="cuda", dtype=torch.bfloat16)
    """

    def __init__(self, index: WeightIndex):
        self.index = index

    def load(self, keys: Iterable[str], *, device: str = "cpu", dtype=None):
        import torch
        from safetensors import safe_open

        want = set(keys)
        by_shard = self.index.shards_for(want)
        sizes = self.index.sizes()

        tensors: dict[str, "torch.Tensor"] = {}
        loaded = 0
        for shard, ks in by_shard.items():
            # framework="pt" + device 让 safetensors 直接落到目标设备，
            # 不经过一次完整的 CPU 拷贝
            with safe_open(str(self.index.dir / shard), framework="pt", device=device) as f:
                for k in ks:
                    t = f.get_tensor(k)
                    if dtype is not None and t.dtype != dtype:
                        t = t.to(dtype)
                    tensors[k] = t
                    loaded += sizes.get(k, t.numel() * t.element_size())

        report = LoadReport(
            n_keys=len(tensors),
            bytes_loaded=loaded,
            bytes_total=self.index.total_bytes,
            shards_opened=len(by_shard),
            shards_total=len(self.index.shards),
            missing=sorted(want - set(tensors)),
        )
        return tensors, report


# --------------------------------------------------------------------------- #
def qwen3_next_keys(plan: KeyPlan, *, layer_types: Sequence[str],
                    shared_expert: bool = True,
                    tie_word_embeddings: bool = False) -> set[str]:
    """Qwen3-Next 的 key 命名 —— **逐层不同**，这是它与 Qwen3-MoE 最大的差别。

    每 4 层里 3 层是 Gated DeltaNet、1 层是标准 attention（`full_attention_interval`），
    两种层的 key 集合完全不重叠::

        linear_attention  model.layers.{i}.linear_attn.{A_log, dt_bias, conv1d.weight,
                                                        in_proj_qkvz, in_proj_ba,
                                                        norm, out_proj}.weight
        full_attention    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
                          model.layers.{i}.self_attn.{q,k}_norm.weight

    两种层都有 MoE，且都带一个**共享专家**（`mlp.shared_expert.*` 与
    `mlp.shared_expert_gate.weight`）。共享专家对每个 token 都激活，
    **不参与驻留集裁剪** —— 承载该层的节点必须装它。

    `layer_types` 用 1-based 的层号索引（`layer_types[l-1]`），与规划器口径一致。
    """
    keys: set[str] = set()
    for layer, experts in plan.layer_experts.items():
        i = int(layer) - 1                       # 1-based → 0-based
        p = f"model.layers.{i}"
        kind = layer_types[i]
        keys |= {f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
                 f"{p}.mlp.gate.weight"}
        if kind == "linear_attention":
            keys |= {
                f"{p}.linear_attn.A_log", f"{p}.linear_attn.dt_bias",
                f"{p}.linear_attn.conv1d.weight",
                f"{p}.linear_attn.in_proj_qkvz.weight",
                f"{p}.linear_attn.in_proj_ba.weight",
                f"{p}.linear_attn.norm.weight",
                f"{p}.linear_attn.out_proj.weight",
            }
        else:
            keys |= {
                f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight", f"{p}.self_attn.o_proj.weight",
                f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight",
            }
        if shared_expert:
            keys |= {
                f"{p}.mlp.shared_expert.gate_proj.weight",
                f"{p}.mlp.shared_expert.up_proj.weight",
                f"{p}.mlp.shared_expert.down_proj.weight",
                f"{p}.mlp.shared_expert_gate.weight",
            }
        for e in experts:
            keys |= {
                f"{p}.mlp.experts.{int(e)}.gate_proj.weight",
                f"{p}.mlp.experts.{int(e)}.up_proj.weight",
                f"{p}.mlp.experts.{int(e)}.down_proj.weight",
            }
    if plan.with_embed:
        keys.add("model.embed_tokens.weight")
    if plan.with_lm_head:
        keys.add("model.norm.weight")
        keys.add("model.embed_tokens.weight" if tie_word_embeddings else "lm_head.weight")
    return keys
