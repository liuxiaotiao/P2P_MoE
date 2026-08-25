"""分散网络模拟器 —— 实现 planner.network.NetworkOracle。

延迟按 II.3.1 给出的物理成因建模：

    延迟 ≈ 出口接入段 + 骨干 + 入口接入段

接入段主导且两端独立。**注意**：规划器本身并不假设这个结构（II.3.1(a)
明确说明公共带是纯实测驱动的），这里用它只是为了让模拟出来的矩阵具备真实
网络的定性特征 —— 存在「对全网普遍偏差的接入点」，从而能复现异类入口诊断
这条路径。把本类换成打真实探测包的实现，规划器代码一行不用改。

每次 probe 真的抽 k 个样本再取经验分位数，所以 k 小的时候分位数本身带噪声 ——
这与文档要求 k ≥ 8（常规）/ k ≥ 16（终审）的动机一致。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from math import log
from typing import Mapping, Sequence

import numpy as np

from ..planner.network import Probe

__all__ = ["AccessProfile", "SimNetwork"]

_LN2 = log(2.0)
_LN20 = log(20.0)


@dataclass
class AccessProfile:
    """一个节点的接入质量。好坏与算力/内存无关（I.1.1 算例设定）。"""

    out_ms: float
    in_ms: float
    jitter_ms: float


class SimNetwork:
    """可复现的分散网络。

    Parameters
    ----------
    node_ids : 全体节点
    seed : 随机种子，决定接入画像与骨干矩阵
    good_access / bad_access : 优质/劣质接入的单向延迟区间（ms）
    bad_frac : 劣质接入节点的比例
    backbone : 骨干段延迟区间（ms）
    jitter : 单节点抖动贡献区间（ms）
    outliers : 强制指定为劣质接入的节点 id —— 用于构造异类入口场景
    """

    def __init__(
        self,
        node_ids: Sequence[str],
        *,
        seed: int = 0,
        good_access: tuple[float, float] = (11.0, 17.0),
        bad_access: tuple[float, float] = (26.0, 34.0),
        bad_frac: float = 0.25,
        backbone: tuple[float, float] = (2.0, 8.0),
        jitter: tuple[float, float] = (4.0, 9.0),
        outliers: Sequence[str] = (),
    ):
        self.node_ids = list(node_ids)
        self.seed = seed
        rng = np.random.default_rng(seed)

        forced = set(outliers)
        n = len(self.node_ids)
        n_bad = max(0, int(round(bad_frac * n)) - len(forced))
        pool = [v for v in self.node_ids if v not in forced]
        bad = set(forced) | set(rng.choice(pool, size=min(n_bad, len(pool)), replace=False).tolist())

        self.access: dict[str, AccessProfile] = {}
        for v in self.node_ids:
            lo, hi = bad_access if v in bad else good_access
            self.access[v] = AccessProfile(
                out_ms=float(rng.uniform(lo, hi)),
                in_ms=float(rng.uniform(lo, hi)),
                jitter_ms=float(rng.uniform(*jitter)) * (1.7 if v in bad else 1.0),
            )
        self.bad_nodes = bad

        self._backbone: dict[tuple[str, str], float] = {}
        for i, a in enumerate(self.node_ids):
            for b in self.node_ids[i + 1 :]:
                x = float(rng.uniform(*backbone))
                self._backbone[(a, b)] = x
                self._backbone[(b, a)] = x

    # -- 真值 -------------------------------------------------------------- #
    def true_p50(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.access[a].out_ms + self.access[b].in_ms + self._backbone[(a, b)]

    def true_jitter(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.access[a].jitter_ms + self.access[b].jitter_ms

    # -- NetworkOracle ----------------------------------------------------- #
    def probe(self, a: str, b: str, k: int) -> Probe:
        """抽 k 个样本取经验 p50 / p95。

        单次样本 = floor + Exp(scale)，参数反解自目标 (p50, p95−p50)：
        中位 = floor + scale·ln2，p95 = floor + scale·ln20
        ⇒ scale = jitter / ln10。
        """
        if a == b:
            return Probe(p50=0.0, p95=0.0, k=k)
        m = self.true_p50(a, b)
        j = self.true_jitter(a, b)
        scale = max(j / (_LN20 - _LN2), 1e-6)
        floor = m - scale * _LN2

        h = hashlib.blake2b(f"{self.seed}|{a}|{b}|{k}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(h, "big"))
        s = floor + rng.exponential(scale, size=k)
        s = np.maximum(s, 0.5)
        return Probe(p50=float(np.quantile(s, 0.5)), p95=float(np.quantile(s, 0.95)), k=k)

    # -- churn / 劣化，供维护层测试用 -------------------------------------- #
    def degrade(self, v: str, add_ms: float, add_jitter: float = 0.0) -> None:
        """模拟某节点接入劣化 —— 触发 II.6 的周期层与即时层。"""
        p = self.access[v]
        self.access[v] = AccessProfile(
            out_ms=p.out_ms + add_ms,
            in_ms=p.in_ms + add_ms,
            jitter_ms=p.jitter_ms + add_jitter,
        )

    def summary(self) -> str:
        pairs = [
            self.true_p50(a, b)
            for i, a in enumerate(self.node_ids)
            for b in self.node_ids[i + 1 :]
        ]
        jit = [
            self.true_jitter(a, b)
            for i, a in enumerate(self.node_ids)
            for b in self.node_ids[i + 1 :]
        ]
        return (
            f"逐对 p50 {min(pairs):.0f}–{max(pairs):.0f}ms（中位 {np.median(pairs):.0f}），"
            f"抖动 {min(jit):.0f}–{max(jit):.0f}ms，劣质接入 {len(self.bad_nodes)}/{len(self.node_ids)} 台"
        )
