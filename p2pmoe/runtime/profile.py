"""激活画像：**逐层**统计每个专家分到多少路由质量，用来决定后段该装哪些专家。

这是整套方案缺的最后一环。「后段只装 n_{u,l} 个专家」（I.1.1）要成立，前提是
知道**哪些** —— 而这个知识只能来自真实数据上的真实路由。没有它，驻留集就是瞎选的，
等于随机丢专家。

怎么产出：全装跑一遍，路由自己会说话
------------------------------------
关键事实：**路由是全量的，即使专家不是。** `mlp.gate.weight` 每层都完整加载
（它只有 E×d，比一个专家还小），所以 `MoEStats.hist` 记的是 top-k 在**全部** E 个
专家上的质量分布 —— 哪怕本地只驻留了 3 个。

于是画像不需要任何额外的离线流程：

    1. 按 `--resident-frac 1.0`（全装）部署一次；
    2. 把某个 task 的真实请求打进去；
    3. 收各节点累计的逐层直方图 → 这就是画像；
    4. 按覆盖率阈值取驻留集，重新部署。

第 1 步全装是必须的：只驻留子集时输出会被 drop-expert 近似带偏，后面几层的
路由就不是真实路由了。画像必须在**无近似**的前提下采。

为什么存质量而不是直接存 id
---------------------------
覆盖率阈值（0.9？0.95？0.99）是个部署期的权衡 —— 内存换质量。存质量分布的话，
换阈值不用重新采样；存 id 就把这个选择冻死在采样那一刻了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..planner.experts import ActivationProfile, ExpertPlacement

__all__ = [
    "LayerProfiler", "ActivationRecord", "merge_records",
    "save_profile", "load_profile", "placement_from_profile",
]


# --------------------------------------------------------------------------- #
class LayerProfiler:
    """挂在 `SegmentModel` 上的累加器。默认不开 —— 开着也只是加法，但没必要。"""

    def __init__(self, n_experts: int):
        self.n_experts = int(n_experts)
        self.mass: dict[int, np.ndarray] = {}
        self.tokens: dict[int, int] = {}

    def record(self, layer: int, hist: np.ndarray, n_token_layer: int) -> None:
        l = int(layer)
        if l not in self.mass:
            self.mass[l] = np.zeros(self.n_experts, dtype=np.float64)
        self.mass[l] += np.asarray(hist, dtype=np.float64)
        self.tokens[l] = self.tokens.get(l, 0) + int(n_token_layer)

    def reset(self) -> None:
        self.mass.clear()
        self.tokens.clear()

    def to_wire(self) -> dict:
        return {
            "n_experts": self.n_experts,
            "layers": {str(l): [round(float(x), 8) for x in m]
                       for l, m in sorted(self.mass.items())},
            "tokens": {str(l): int(n) for l, n in sorted(self.tokens.items())},
        }


# --------------------------------------------------------------------------- #
@dataclass
class ActivationRecord:
    """一个 task 的逐层累计质量。层号是**全局**的（1-based），只含实际统计到的层。"""

    task: str
    n_experts: int
    mass: dict[int, np.ndarray] = field(default_factory=dict)
    tokens: dict[int, int] = field(default_factory=dict)

    def add_wire(self, d: Mapping) -> None:
        for k, v in d.get("layers", {}).items():
            l = int(k)
            arr = np.asarray(v, dtype=np.float64)
            self.mass[l] = self.mass.get(l, np.zeros(self.n_experts)) + arr
        for k, v in d.get("tokens", {}).items():
            l = int(k)
            self.tokens[l] = self.tokens.get(l, 0) + int(v)

    @property
    def layers(self) -> list[int]:
        return sorted(self.mass)

    @property
    def n_tokens(self) -> int:
        """按层取最大 —— 各层的 token 数本该相同，不同就说明有层漏采了。"""
        return max(self.tokens.values(), default=0)

    def normalised(self, layer: int) -> np.ndarray:
        m = self.mass[layer]
        s = m.sum()
        return m / s if s > 0 else np.full_like(m, 1.0 / len(m))

    def to_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "layers": {str(l): [round(float(x), 8) for x in self.normalised(l)]
                       for l in self.layers},
        }


def merge_records(records: Sequence[ActivationRecord]) -> ActivationRecord:
    if not records:
        raise ValueError("没有可合并的画像")
    out = ActivationRecord(task=records[0].task, n_experts=records[0].n_experts)
    for r in records:
        for l, m in r.mass.items():
            out.mass[l] = out.mass.get(l, np.zeros(out.n_experts)) + m
        for l, n in r.tokens.items():
            out.tokens[l] = out.tokens.get(l, 0) + n
    return out


# --------------------------------------------------------------------------- #
def save_profile(path: str | Path, records: Mapping[str, ActivationRecord], *,
                 model: str = "", n_layers: int = 0) -> Path:
    p = Path(path)
    n_experts = next(iter(records.values())).n_experts if records else 0
    p.write_text(json.dumps({
        "model": model,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "tasks": {u: r.to_dict() for u, r in sorted(records.items())},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_profile(path: str | Path) -> dict:
    """读画像文件。两种格式都认。

    * **质量格式**（本模块产出）：`{"tasks": {u: {"layers": {l: [mass...]}}}}`；
    * **id 格式**（外部流程直接给驻留集）：`{u: [[ids] × n_layers]}`。

    认第二种是因为画像未必出自这个仓库 —— 别人拿别的工具统计出来的驻留集
    也该能喂进来。但它把覆盖率的选择冻在了产出那一刻，能选就选第一种。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "tasks" in raw:
        return raw
    return {"format": "ids", "tasks": raw}


def placement_from_profile(
    raw: Mapping, task: str, *, coverage: float, n_experts: int,
    layers: Sequence[int] | None = None, min_experts: int = 1,
) -> dict[int, tuple[int, ...]]:
    """画像 → `{层号: 驻留专家 id}`。**只覆盖画像里有的层**，其余层由调用方决定。

    取法与 `planner.experts.build_placement` 一致：逐层按质量降序累加，
    到覆盖率就停。层与层独立，所以 n_{u,l} 天然逐层异构 —— 文档说的就是这个。

    `min_experts` 该传 `top_k`：某一层的质量高度集中时，覆盖率规则可能只选出
    一个专家，而 top-k 路由每个 token 要 k 个 —— 那样**每个 token 都必然 miss**，
    drop-expert 兜不住这种程度。这个下界在生成时就补齐，而不是留给校验去拦。
    """
    tasks = raw.get("tasks", raw)
    if task not in tasks:
        raise ValueError(f"画像里没有 task {task!r}；有的是 {sorted(tasks)}")
    entry = tasks[task]

    if raw.get("format") == "ids" or isinstance(entry, list):
        rows = entry
        want = list(layers) if layers else list(range(1, len(rows) + 1))
        if len(rows) != len(want):
            raise ValueError(f"id 格式的画像有 {len(rows)} 层，要填 {len(want)} 层")
        return {l: tuple(sorted(int(e) for e in ids)) for l, ids in zip(want, rows)}

    out: dict[int, tuple[int, ...]] = {}
    for k, mass in entry.get("layers", {}).items():
        l = int(k)
        if layers is not None and l not in layers:
            continue
        m = np.asarray(mass, dtype=np.float64)
        if m.size != n_experts:
            raise ValueError(f"第 {l} 层画像有 {m.size} 个专家，模型是 {n_experts} 个")
        order = np.argsort(-m, kind="stable")
        acc, chosen = 0.0, []
        for e in order:
            chosen.append(int(e))
            acc += float(m[e])
            if acc >= coverage - 1e-12 and len(chosen) >= min_experts:
                break
        out[l] = tuple(sorted(chosen))
    return out


def to_activation_profile(raw: Mapping, task: str, *, n_layers: int,
                          n_experts: int) -> ActivationProfile:
    """转成规划器认识的 `ActivationProfile`（画像里没有的层填均匀分布）。"""
    tasks = raw.get("tasks", raw)
    layers = tasks[task].get("layers", {})
    uniform = tuple(1.0 / n_experts for _ in range(n_experts))
    return ActivationProfile(
        task=task, n_layers=n_layers, n_experts=n_experts,
        mass=tuple(tuple(layers[str(l)]) if str(l) in layers else uniform
                   for l in range(1, n_layers + 1)),
    )


def summarise(raw: Mapping, *, coverage: float, min_experts: int = 1) -> list[str]:
    """给人看的一段话：每个 task 在这个覆盖率下每层要装几个。"""
    n_e = int(raw.get("n_experts", 0))
    out: list[str] = []
    for u, entry in sorted(raw.get("tasks", raw).items()):
        sets = placement_from_profile(raw, u, coverage=coverage, n_experts=n_e,
                                      min_experts=min_experts)
        if not sets:
            continue
        sizes = [len(v) for _, v in sorted(sets.items())]
        out.append(
            f"{u}: {len(sizes)} 层，每层 {min(sizes)}–{max(sizes)} 个"
            f"（层均 {sum(sizes)/len(sizes):.1f}/{n_e}，"
            f"约 {sum(sizes)/len(sizes)/n_e:.0%}），"
            f"采样 {entry.get('n_tokens', 0)} 个 token"
        )
    return out
