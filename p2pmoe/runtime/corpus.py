"""回放语料与激活画像 —— 把「统计」这一步做实。

`sim/replay.py` 生成的是**合成**画像（直接编造一个质量分布），够用来验算规划器，
但它和真实模型的路由行为没有因果关系。这里做的是文档 I.1.1 描述的真实流程：

    回放语料  →  用**全专家**模型跑一遍  →  逐层统计路由质量  →  按覆盖率阈值取驻留集

这样整条链路是自洽的：驻留集来自真实路由，在线的 miss 率就真的等于
「1 − 覆盖率」，而不是靠参数凑出来的。识别 task 的直方图分类器也建立在同一份
统计上 —— 它能不能分开，是模型和语料的性质，不是我们说了算。

task 之间的区分来自输入分布：每个 task 偏好词表里不同的一段。这对应真实场景里
「不同 task 的输入分布不同 → 逐层路由分布不同 → 驻留专家集不同」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..planner.experts import ActivationProfile, ExpertPlacement
from .model import SegmentModel, ToyMoEConfig, embed_tokens, lm_head, token_cluster


def _sample(cfg: ToyMoEConfig, h: np.ndarray, rng: np.random.Generator) -> int:
    logits = lm_head(cfg, h[-1]) / max(cfg.sample_temp, 1e-6)
    pr = np.exp(logits - logits.max())
    return int(rng.choice(len(pr), p=pr / pr.sum()))

__all__ = ["TaskCorpus", "make_corpus", "profile_from_corpus", "measure_baseline_miss",
           "measure_front_refs", "sample_prompt"]


@dataclass
class TaskCorpus:
    task: str
    token_pool: np.ndarray
    """本 task 偏好的词表子集。"""
    sequences: list[list[int]]


def make_corpus(
    cfg: ToyMoEConfig,
    tasks: Sequence[str],
    *,
    n_seq: int = 24,
    seq_len: int = 16,
    clusters_per_task: int = 2,
    shared_clusters: int = 1,
    seed: int = 0,
) -> dict[str, TaskCorpus]:
    """构造逐 task 的回放语料。

    每个 task 的词表池由 `clusters_per_task` 个词簇组成，另加 `shared_clusters`
    个所有 task 共用的簇。共用簇越多，各 task 的激活分布越像 → 驻留集重叠度越高
    → q(u,û) 越低 → 越该合并池（推论 III.7.4）。这是本 demo 里控制「误绑可检性」
    的唯一旋钮，且它有明确的语义，不是拍出来的参数。
    """
    rng = np.random.default_rng(seed)
    tc = token_cluster(cfg)
    n_c = cfg.n_token_clusters
    need = shared_clusters + clusters_per_task * len(tasks)
    if need > n_c:
        raise ValueError(f"需要 {need} 个词簇，但只有 {n_c} 个（调大 n_token_clusters）")

    order = rng.permutation(n_c)
    shared = order[:shared_clusters]

    out: dict[str, TaskCorpus] = {}
    for i, u in enumerate(tasks):
        lo = shared_clusters + i * clusters_per_task
        mine = np.concatenate([shared, order[lo : lo + clusters_per_task]])
        pool = np.where(np.isin(tc, mine))[0]
        seqs = [
            rng.choice(pool, size=seq_len, replace=True).tolist() for _ in range(n_seq)
        ]
        out[u] = TaskCorpus(task=u, token_pool=pool, sequences=seqs)
    return out


def profile_from_corpus(
    cfg: ToyMoEConfig, corpus: Mapping[str, TaskCorpus]
) -> dict[str, ActivationProfile]:
    """用**全专家**模型跑回放语料，逐层统计路由质量 —— 这就是离线画像。

    注意必须用全专家模型：驻留集还没定，此时任何裁剪都会让统计有偏。真实系统
    里这一步在一台装得下全模型的机器上离线跑一次（或分层分批跑）。
    """
    full = {l: range(cfg.n_experts) for l in range(1, cfg.n_layers + 1)}
    out: dict[str, ActivationProfile] = {}

    for u, c in corpus.items():
        acc = np.zeros((cfg.n_layers, cfg.n_experts), dtype=np.float64)
        for j, ids in enumerate(c.sequences):
            # 每条序列用独立的 SegmentModel 实例？不必 —— 换 req_id 即可，
            # KV 按 req 分桶，互不干扰。
            model = SegmentModel(cfg, full) if j == 0 else model
            req = f"{u}-{j}"
            # 逐层统计：SegmentModel.forward 返回的是全段合计，这里要逐层，
            # 所以直接驱动 block
            kv: dict[int, dict] = {}
            h = embed_tokens(cfg, ids)
            for li, l in enumerate(model.layers):
                h, st = model.blocks[l].forward(h, kv.setdefault(l, {}))
                acc[l - 1] += st.hist
            model.drop_kv(req)

        acc = acc / np.maximum(acc.sum(axis=1, keepdims=True), 1e-12)
        out[u] = ActivationProfile(
            task=u,
            n_layers=cfg.n_layers,
            n_experts=cfg.n_experts,
            mass=tuple(tuple(float(x) for x in row) for row in acc),
        )
    return out


def measure_baseline_miss(
    cfg: ToyMoEConfig,
    corpus: Mapping[str, TaskCorpus],
    front: ExpertPlacement,
    backs: Mapping[str, ExpertPlacement],
    l0: int,
    *,
    n_seq: int = 8,
    n_decode: int = 16,
) -> dict[str, float]:
    """实测「绑对池」时后段的 miss 事件率 —— 通道二的告警基线（II.5）。

    [为什么不能用 1 − 覆盖率 —— 对文档的一处补正]

    文档 II.5 写「基线 = 1−覆盖率」。实测下来基线是它的 3–4 倍，两条原因叠加：

    1. **口径差一个 k**：覆盖率是**质量**口径（激活质量有多少落在驻留集内），
       而 miss 是**事件**口径（该 token 的 top-k 里有没有缺的）。top-k 给了 k 次
       落空机会，事件率大致是质量缺失率的 k 倍。

    2. **前段的 miss 会沿层放大**：前段驻留的是并集，但并集同样是按覆盖率截的，
       它自己也有 miss、也在做 drop-expert 重归一。于是送给后段的隐状态已经偏离
       了画像统计时的分布，后段的路由跟着漂，miss 率再抬一截。本 demo 实测：
       前段换成全专家时后段 miss 从 19% 掉到 2.8%，差的就是这一项。

    3. **必须只统计 decode 段**：在线的滑窗是对最近若干个 **decode** token 求
       miss 率的，而 decode 阶段的 miss 明显高于 prefill —— 生成出来的 token 不再
       来自回放语料的分布，隐状态逐步漂移，路由跟着漂。把 prefill 混进基线会把
       基线拉低，于是滑窗（纯 decode）永远高于基线，绑对的池也照样报警。
       所以这里跑完整轨迹但**只累计 decode 步的统计**，与滑窗口径一致。

    用 1−覆盖率 当告警线的后果是可复现的：**绑对池也会持续误报**，触发无谓换绑，
    每次换绑要重算一遍后段 prefill。所以基线必须实测 —— 就在这里，用与在线
    完全相同的配置（前段=并集、后段=该 task 驻留集）跑一遍完整轨迹。
    """
    out: dict[str, float] = {}
    for u, plc in backs.items():
        F = SegmentModel(cfg, {l: front.at(l) for l in range(1, l0 + 1)})
        B = SegmentModel(cfg, {l: plc.at(l) for l in range(l0 + 1, cfg.n_layers + 1)})
        miss = tot = 0
        for j, ids in enumerate(corpus[u].sequences[:n_seq]):
            req = f"cal-{u}-{j}"
            rng = np.random.default_rng(1000 + j)
            h, _ = F.forward(req, embed_tokens(cfg, ids))
            h, _st = B.forward(req, h)   # prefill 不计入 —— 滑窗看的是 decode
            tok = _sample(cfg, h, rng)
            for _ in range(n_decode):
                h, _ = F.forward(req, embed_tokens(cfg, [tok]))
                h, st = B.forward(req, h)
                miss += st.miss_token_layer
                tot += st.n_token_layer
                tok = _sample(cfg, h, rng)
            F.drop_kv(req)
            B.drop_kv(req)
        out[u] = miss / tot if tot else 0.0
    return out


def measure_front_refs(
    cfg: ToyMoEConfig,
    corpus: Mapping[str, TaskCorpus],
    front: ExpertPlacement,
    l0: int,
    *,
    n_seq: int = 12,
    n_decode: int = 4,
) -> dict[str, np.ndarray]:
    """实测分类器的参考直方图 —— 同样要在**在线配置**下测。

    直接拿离线画像（全专家模型的逐层质量）当参考是不对的：在线时前段只装并集，
    它的直方图是被截断并重归一过的，与全专家画像不同分布。用错的参考做余弦匹配，
    识别准确率会明显掉（本 demo 实测 1/3 → 3/3）。

    这与 measure_baseline_miss 是同一条教训：**任何在线要用的统计量，都必须在
    与在线一致的配置和轨迹上标定**。
    """
    out: dict[str, np.ndarray] = {}
    for u, c in corpus.items():
        F = SegmentModel(cfg, {l: front.at(l) for l in range(1, l0 + 1)})
        acc = np.zeros(cfg.n_experts)
        for j, ids in enumerate(c.sequences[:n_seq]):
            req = f"ref-{u}-{j}"
            rng = np.random.default_rng(2000 + j)
            h, st = F.forward(req, embed_tokens(cfg, ids))
            acc += st.hist
            F.drop_kv(req)
        out[u] = acc / max(acc.sum(), 1e-12)
    return out


def sample_prompt(
    corpus: Mapping[str, TaskCorpus], task: str, length: int, *, seed: int = 0
) -> list[int]:
    """从某 task 的池子里抽一条在线请求的 prompt。

    在线时 task 是未知的 —— 这个函数只是 demo 用来「知道正确答案」以便核对
    识别是否正确。真实系统看不到它。
    """
    rng = np.random.default_rng(seed)
    return rng.choice(corpus[task].token_pool, size=length, replace=True).tolist()
