"""task 识别：前段激活直方图分类器（II.5 的识别通道）。

文档 II.5 的口径：

    识别: 前段逐节点把本地专家激活直方图(字节级)捎带在 hidden state 后传;
          至 tail(f) 聚齐 → 本地分类出 û → 缓存 layer L₀ 输出

三点值得强调：

* **分类在 tail(f) 本地做**，不是回协调器算。直方图只有 n_experts 个浮点数，
  跟着 hidden state 走，不额外付 RTT —— 这是「捎带」的全部意义。
* **识别是旁路统计**，不进计算图。前段照常做精确 forward，直方图只是顺手记的
  副产品。这正是命题 III.7.1（前段 KV 复用合法）的前提。
* **参考画像来自同一份回放统计**（`corpus.profile_from_corpus`），所以分类器
  和驻留集是同源的：能不能分开，是模型与语料的性质，不是调参调出来的。

置信三区（II.5）：
    c_max ≥ τ_hi        → 提交 argmax 池，释放 L₀ 缓存
    τ_lo ≤ c_max < τ_hi → 暂定 argmax 池，保留缓存，监控 D token
    c_max < τ_lo        → 绑**最大先验池**并全程监控（分散环境无兜底池，III.7.5）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..planner.experts import ActivationProfile

__all__ = ["HistogramClassifier", "Verdict"]


@dataclass
class Verdict:
    task: str
    confidence: float
    zone: str
    """"commit" | "observe" | "prior" —— 置信三区（II.5）。"""
    scores: dict[str, float]

    @property
    def keep_cache(self) -> bool:
        """观察区与低置信区都要保留 L₀ 缓存，以便换绑时不重放前段。"""
        return self.zone != "commit"


class HistogramClassifier:
    """按前段各层的参考直方图做余弦匹配。"""

    def __init__(
        self,
        refs: Mapping[str, np.ndarray],
        priors: Mapping[str, float],
        *,
        tau_hi: float = 0.55,
        tau_lo: float = 0.40,
        temp: float = 12.0,
    ):
        self.tasks = sorted(refs)
        self.refs = {u: self._norm(np.asarray(refs[u], dtype=float)) for u in self.tasks}
        self.priors = dict(priors)
        self.tau_hi = tau_hi
        self.tau_lo = tau_lo
        self.temp = temp

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def from_profiles(
        cls,
        profiles: Mapping[str, ActivationProfile],
        l0: int,
        priors: Mapping[str, float],
        **kw,
    ) -> "HistogramClassifier":
        """参考画像 = 各 task 在前段层（1..L₀）上的激活质量之和。"""
        refs = {}
        for u, p in profiles.items():
            acc = np.zeros(p.n_experts)
            for l in range(1, l0 + 1):
                acc += np.asarray(p.at(l))
            refs[u] = acc
        return cls(refs, priors, **kw)

    def to_wire(self) -> dict:
        return {
            "refs": {u: [float(x) for x in v] for u, v in self.refs.items()},
            "priors": self.priors,
            "tau_hi": self.tau_hi,
            "tau_lo": self.tau_lo,
            "temp": self.temp,
        }

    @classmethod
    def from_wire(cls, d: Mapping) -> "HistogramClassifier":
        return cls(
            {u: np.asarray(v) for u, v in d["refs"].items()},
            d["priors"],
            tau_hi=d["tau_hi"],
            tau_lo=d["tau_lo"],
            temp=d["temp"],
        )

    # -- 推断 -------------------------------------------------------------- #
    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def predict(self, hist: Sequence[float]) -> Verdict:
        h = self._norm(np.asarray(hist, dtype=float))
        sims = np.array([float(h @ self.refs[u]) for u in self.tasks])
        e = np.exp((sims - sims.max()) * self.temp)
        probs = e / e.sum()
        scores = {u: float(p) for u, p in zip(self.tasks, probs)}

        i = int(np.argmax(probs))
        best, c = self.tasks[i], float(probs[i])
        if c >= self.tau_hi:
            return Verdict(best, c, "commit", scores)
        if c >= self.tau_lo:
            return Verdict(best, c, "observe", scores)
        # 低置信：绑最大先验池并全程监控（分散环境取消兜底池，命题 III.7.5）
        prior_best = max(self.priors, key=lambda u: self.priors.get(u, 0.0))
        return Verdict(prior_best, c, "prior", scores)
