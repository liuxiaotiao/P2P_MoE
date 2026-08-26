"""Toy MoE 与 PartialExpertMoEBlock —— 方案里最关键的工程原语。

为什么是「只驻留子集」而不是「全部加载后只用一部分」
------------------------------------------------------
整套方案成立的前提是后段每层只装 n_{u,l} 个专家（算例 C：8/64）。这不是优化，
是可行性问题 —— 装全部 64 个专家的话，一层就是 17.4GB，32 层根本放不下。所以
`PartialExpertMoEBlock` 的构造函数只为 `resident` 里的专家分配权重，其余专家的
张量在这个进程里**从来不存在**。

由此带来三件必须处理的事，文档 II.5 都点到了，这里是它们的实现：

1. **miss 检出**：router 的 top-k 可能选中不在本地的专家。这是误绑的直接症状
   （通道二），也是 drop-expert 的触发条件。
2. **drop-expert 近似**：单发 miss 时跳过缺失专家、把剩下的门控权重重归一。
   文档标注为「运维近似，非无损」—— 这里如实实现并统计影响。
3. **激活直方图**：前段逐节点把本地路由质量累加，捎带在 hidden state 后传，
   到 tail(f) 聚齐后用于识别 task（II.5 的识别通道）。

权重从种子确定性派生
--------------------
每个进程按 (seed, layer, expert) 生成权重，所以 24 个进程不用传一个字节的权重
就能拿到一致的模型。这同时也精确模拟了目标形态：节点只 materialize 自己那份，
对应真实系统里 safetensors 的逐张量 mmap + 挑 key。

实现用 numpy 而非 torch：这一层的目的是验证**协议**（KV 语义、miss 检出、换绑），
不是 kernel 性能；而方案自己的账本里计算只占 13%。换成 torch/HF 时保持
`PartialExpertMoEBlock` 的接口不变即可。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ToyMoEConfig",
    "MoEStats",
    "PartialExpertMoEBlock",
    "SegmentModel",
    "embed_tokens",
    "embedding_table",
    "token_cluster",
    "lm_head",
]


# --------------------------------------------------------------------------- #
def _rng(seed: int, *parts: object) -> np.random.Generator:
    """按 (seed, parts...) 确定性地取一个随机流 —— 跨进程一致。"""
    key = "|".join(str(p) for p in parts)
    h = hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


@dataclass(frozen=True)
class ToyMoEConfig:
    n_layers: int = 8
    d_model: int = 96
    n_experts: int = 32
    top_k: int = 2
    d_ff: int = 128
    vocab: int = 256
    n_token_clusters: int = 8
    """词表的簇数。

    真实 MoE 里「某个 task 只用 8/64 个专家」不是巧合 —— 它成立的前提是该 task
    的输入占据表示空间的一小块区域，于是逐层路由稳定地落在同一批专家上。
    如果词嵌入是完全随机的，不同 token 会路由到不同专家，按覆盖率取驻留集就要
    27/32 个，「只驻留子集」这件事在语料层面就不成立了。

    所以词表按簇构造：同簇 token 的嵌入靠近同一个中心，task 的词表池由若干簇
    组成。cluster_w 控制「簇中心 vs 个体噪声」的比例，也就是 task 的可分辨程度。
    """
    cluster_w: float = 0.90
    sample_temp: float = 0.7
    """输出采样温度。"""
    router_temp: float = 12.0
    """router 的 logit 温度。

    随机初始化的 router 做不出真实 MoE 那种尖锐的路由分布 —— 不加温度时
    softmax 近乎均匀，覆盖 97% 的质量需要 27/32 个专家，「只驻留子集」这件事
    就失去意义了。真实模型的路由是训练出来的、高度集中的，这里用温度 + 方向
    归一化来复现那个定性特征：logit = cos(h, e_j) × temp。
    """
    seed: int = 20260824

    @property
    def expert_params(self) -> int:
        return 2 * self.d_model * self.d_ff

    @property
    def base_params(self) -> int:
        """每层非专家部分：q/k/v/o + router。"""
        return 4 * self.d_model * self.d_model + self.d_model * self.n_experts


# --------------------------------------------------------------------------- #
@dataclass
class MoEStats:
    """一次前向在一段上累计的路由统计。"""

    hist: np.ndarray
    """[n_experts] 路由质量直方图 —— 捎带在 hidden state 后传，用于识别 task。"""
    n_token_layer: int = 0
    """(token × 层) 计数，miss 率的分母。"""
    miss_token_layer: int = 0
    """至少有一个 top-k 专家不在本地的 (token, 层) 数。"""
    miss_mass: float = 0.0
    """缺失掉的门控质量之和 —— drop-expert 重归一影响的大小。"""

    @property
    def miss_rate(self) -> float:
        return self.miss_token_layer / self.n_token_layer if self.n_token_layer else 0.0

    def merge(self, other: "MoEStats") -> "MoEStats":
        return MoEStats(
            hist=self.hist + other.hist,
            n_token_layer=self.n_token_layer + other.n_token_layer,
            miss_token_layer=self.miss_token_layer + other.miss_token_layer,
            miss_mass=self.miss_mass + other.miss_mass,
        )

    @classmethod
    def zeros(cls, n_experts: int) -> "MoEStats":
        return cls(hist=np.zeros(n_experts, dtype=np.float64))

    def to_wire(self) -> dict:
        return {
            "hist": [round(float(x), 6) for x in self.hist],
            "ntl": self.n_token_layer,
            "miss": self.miss_token_layer,
            "mass": round(self.miss_mass, 6),
        }

    @classmethod
    def from_wire(cls, d: Mapping) -> "MoEStats":
        return cls(
            hist=np.asarray(d["hist"], dtype=np.float64),
            n_token_layer=int(d["ntl"]),
            miss_token_layer=int(d["miss"]),
            miss_mass=float(d["mass"]),
        )


# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def token_cluster(cfg: ToyMoEConfig) -> np.ndarray:
    """每个 token 属于哪个簇。[vocab]"""
    return _rng(cfg.seed, "tokclust").integers(0, cfg.n_token_clusters, size=cfg.vocab)


def embedding_table(cfg: ToyMoEConfig) -> np.ndarray:
    """按簇构造的词嵌入表（见 ToyMoEConfig.n_token_clusters）。"""
    r = _rng(cfg.seed, "embed")
    centers = r.standard_normal((cfg.n_token_clusters, cfg.d_model))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    noise = r.standard_normal((cfg.vocab, cfg.d_model))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    c = token_cluster(cfg)
    w = cfg.cluster_w
    tbl = w * centers[c] + (1.0 - w) * noise
    return tbl / np.linalg.norm(tbl, axis=1, keepdims=True)


def embed_tokens(cfg: ToyMoEConfig, ids: Sequence[int]) -> np.ndarray:
    """共享词嵌入。真实系统里它和 layer 1 一起放在 head(f)。"""
    return embedding_table(cfg)[np.asarray(ids, dtype=int) % cfg.vocab]


def lm_head(cfg: ToyMoEConfig, h: np.ndarray) -> np.ndarray:
    """输出头，与词嵌入**权重绑定**（weight tying）。

    这不是为了省参数，是为了让生成的 token 留在输入分布里。用一个随机的输出头
    时，argmax 会挑出词表里任意一个 token —— 它的嵌入可能落在完全不同的词簇，
    于是 decode 到第二个 token 时隐状态就漂出了该 task 的表示区域，路由跟着漂，
    在**绑对池**的情况下 miss 率也能冲到 30%+，通道二的检出信号就此失效。

    这是一个真实系统里不会出现、但 toy 模型里必然出现的失真：训练过的模型生成
    的是符合上下文的 token，而随机初始化的输出头不是。权重绑定让 argmax 选出
    「嵌入与当前隐状态最接近的 token」，自然地把生成留在同一个词簇里。
    """
    return h @ embedding_table(cfg).T


# --------------------------------------------------------------------------- #
class PartialExpertMoEBlock:
    """一层：因果 attention（带 KV cache）+ router + top-k 专家 FFN。

    **只有 `resident` 里的专家会被 materialize。** 这是与所有主流推理框架
    最本质的差别 —— 它们把每层专家打成一个 [E, ...] 的 fused 张量，加载器按
    完整 E 维走，没有「只装子集」这个概念。
    """

    def __init__(self, cfg: ToyMoEConfig, layer: int, resident: Iterable[int]):
        self.cfg = cfg
        self.layer = layer
        self.resident = frozenset(int(e) for e in resident)
        if not self.resident:
            raise ValueError(f"layer {layer} 的驻留专家集为空")

        r = _rng(cfg.seed, "attn", layer)
        d = cfg.d_model
        s = 1.0 / np.sqrt(d)
        self.wq = r.standard_normal((d, d)) * s
        self.wk = r.standard_normal((d, d)) * s
        self.wv = r.standard_normal((d, d)) * s
        self.wo = r.standard_normal((d, d)) * s
        # router 的每个专家向量取单位长度，配合输入方向归一化 ⇒ logit 就是余弦
        wr = _rng(cfg.seed, "router", layer).standard_normal((d, cfg.n_experts))
        self.wr = wr / np.linalg.norm(wr, axis=0, keepdims=True)

        # 只为驻留专家分配权重 —— 其余专家的张量在本进程里不存在
        self._w1: dict[int, np.ndarray] = {}
        self._w2: dict[int, np.ndarray] = {}
        for e in sorted(self.resident):
            re = _rng(cfg.seed, "expert", layer, e)
            self._w1[e] = re.standard_normal((d, cfg.d_ff)) * s
            self._w2[e] = re.standard_normal((cfg.d_ff, d)) * s

    # -- 计量 -------------------------------------------------------------- #
    @property
    def resident_bytes(self) -> int:
        base = (self.wq.nbytes + self.wk.nbytes + self.wv.nbytes
                + self.wo.nbytes + self.wr.nbytes)
        exp = sum(a.nbytes for a in self._w1.values()) + sum(
            a.nbytes for a in self._w2.values()
        )
        return base + exp

    @property
    def full_bytes(self) -> int:
        """若装全部专家会占多少 —— 用来量化「选择性加载」省了多少。"""
        per = self.cfg.expert_params * 8
        base = (self.wq.nbytes + self.wk.nbytes + self.wv.nbytes
                + self.wo.nbytes + self.wr.nbytes)
        return base + per * self.cfg.n_experts

    # -- 前向 -------------------------------------------------------------- #
    def _attend(self, x: np.ndarray, cache: dict) -> np.ndarray:
        """因果 attention。cache 里存本层的 K/V —— 换绑时按段丢弃的就是它。"""
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        ck, cv = cache.get("k"), cache.get("v")
        k = np.concatenate([ck, k], axis=0) if ck is not None else k
        v = np.concatenate([cv, v], axis=0) if cv is not None else v
        cache["k"], cache["v"] = k, v

        t_new, t_all = q.shape[0], k.shape[0]
        scores = q @ k.T / np.sqrt(self.cfg.d_model)
        # 因果 mask：第 i 个新 token 只能看到前 (t_all - t_new + i) 个
        offset = t_all - t_new
        idx = np.arange(t_all)[None, :] > (offset + np.arange(t_new))[:, None]
        scores = np.where(idx, -1e30, scores)
        return _softmax(scores) @ v @ self.wo

    def forward(self, x: np.ndarray, cache: dict) -> tuple[np.ndarray, MoEStats]:
        """x: [T, d]（prefill 时 T>1，decode 时 T=1）。返回 (y, 本层统计)。"""
        h = x + self._attend(x, cache)

        hn = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9)
        logits = (hn @ self.wr) * self.cfg.router_temp
        probs = _softmax(logits)
        k = self.cfg.top_k
        topk = np.argpartition(-probs, kth=k - 1, axis=-1)[:, :k]

        stats = MoEStats.zeros(self.cfg.n_experts)
        out = np.zeros_like(h)

        for t in range(h.shape[0]):
            picks = topk[t]
            gates = probs[t, picks]
            here = np.array([int(e) in self.resident for e in picks])

            stats.hist[picks] += gates            # 激活质量直方图（捎带用）
            stats.n_token_layer += 1

            if not here.all():
                # ---- drop-expert 近似（II.5）：跳过缺失专家、门控重归一 ----
                # 文档明确标注这是「运维近似，非无损」。这里如实统计它的影响：
                # miss_mass 就是被丢掉的那部分门控质量。
                stats.miss_token_layer += 1
                stats.miss_mass += float(gates[~here].sum())
                if not here.any():
                    continue  # top-k 全缺，本层对该 token 只走 attention 残差
                picks, gates = picks[here], gates[here]

            gates = gates / gates.sum()
            acc = np.zeros(self.cfg.d_model)
            for e, g in zip(picks, gates):
                e = int(e)
                acc += g * (np.maximum(h[t] @ self._w1[e], 0.0) @ self._w2[e])
            out[t] = acc

        return h + out, stats


# --------------------------------------------------------------------------- #
class SegmentModel:
    """一台节点上承载的层区间。

    KV cache 按 (req_id, layer) 组织 —— 换绑时后段整体 drop、前段原地不动，
    这正是命题 III.7.1 的实现依据：前段的计算是对输入的精确 forward，
    与事后的 task 判断无关，所以它的 KV 对任何结论都有效。
    """

    def __init__(self, cfg: ToyMoEConfig, layer_experts: Mapping[int, Iterable[int]]):
        self.cfg = cfg
        self.layers = sorted(int(l) for l in layer_experts)
        self.blocks = {
            int(l): PartialExpertMoEBlock(cfg, int(l), es)
            for l, es in layer_experts.items()
        }
        self._kv: dict[str, dict[int, dict]] = {}
        self.profiler = None
        """逐层激活画像的累加器。默认 None —— 见 runtime/profile.py。"""

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
    def forward(self, req: str, x: np.ndarray) -> tuple[np.ndarray, MoEStats]:
        kv = self._kv.setdefault(req, {})
        total = MoEStats.zeros(self.cfg.n_experts)
        h = x
        for l in self.layers:
            h, st = self.blocks[l].forward(h, kv.setdefault(l, {}))
            if self.profiler is not None:
                # 逐层记，不是合并后再记 —— 驻留集是逐层决定的（n_{u,l} 异构）
                self.profiler.record(l, st.hist, st.n_token_layer)
            total = total.merge(st)
        return h, total

    # -- KV 生命周期 -------------------------------------------------------- #
    def drop_kv(self, req: str) -> bool:
        """换绑时后段调用。前段**不调**（命题 III.7.1）。"""
        return self._kv.pop(req, None) is not None

    def has_kv(self, req: str) -> bool:
        return req in self._kv

    def kv_tokens(self, req: str) -> int:
        kv = self._kv.get(req)
        if not kv:
            return 0
        first = kv.get(self.layers[0], {})
        k = first.get("k")
        return 0 if k is None else int(k.shape[0])

    def active_requests(self) -> list[str]:
        return sorted(self._kv)
