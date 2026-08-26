"""手动放置：**你说哪台装哪几层、哪条前段连哪条后段**，其余一概不算。

规划器整条链路（探测 → 分档估上限 → 选 L₀ → 建段 → 公共中值域 → 回环裁剪）
在这里全部跳过。留下的只有两件事：

1. 把你写的布局翻译成逐节点逐层的加载指令（`DeploymentManifest`）；
2. 校验它自洽 —— 层区间连不连得上、节点有没有被用两次、内存够不够。

什么时候该用这条路
------------------
* 就想先跑通「链路建起来、token 绕出来」，不关心放置好不好；
* 机器的角色你已经定死了（这台有 A100、那台只能当后段）；
* 调试：把某个特定的分法固定下来复现问题。

什么时候不该用
--------------
它**不做任何优化，也不做任何测量**。放置对不对、通道之间延迟齐不齐、
某台机器是不是异类入口 —— 一概不管，全按你说的来。方案文档里那些结论
（零后悔、任意组合、均匀性）都以「离线把组合极差压到抖动量级以下」为前提，
手动放置不提供这个前提，所以那些结论在这里不成立。想要它们，走
`deploy/control.py`。

布局文件长这样
--------------
最短的形式 —— 给 L₀ 和每条通道用哪些机器，层怎么切自动均分::

    {
      "model_dir": "/data/qwen3-30b-a3b",
      "l0": 6,
      "channels": [
        {"front": "n1",          "back": ["n2", "n3", "n4"]},
        {"front": ["n5", "n6"],  "back": ["n7", "n8", "n9"]}
      ]
    }

要精确控制每台装哪几层，就把层区间写出来（闭区间、1-based）::

    {"front": [{"node": "n1", "layers": [1, 4]},
               {"node": "n2", "layers": [5, 6]}],
     "back":  [{"node": "n3", "layers": [7, 30]},
               {"node": "n4", "layers": [31, 48]}]}

专家默认**全装**（无 drop-expert 近似，输出与单机逐位一致）。要只装子集，
在顶层给 `experts`，按层指定::

    "experts": {"7": [0, 3, 5, 12], "8": [1, 3, 9]}

一层只属于一条段，所以按层指定不会有歧义。**没有真实激活画像就别用它** ——
瞎选的驻留集等于随机丢专家，输出会是废的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..planner.manifest import DeploymentManifest, LayerLoad, NodePlan
from ..planner.types import ModelSpec

__all__ = ["ManualSpec", "SegmentLayout", "ChannelLayout", "build_manual_manifest"]


# --------------------------------------------------------------------------- #
def split_layers(lo: int, hi: int, n: int) -> list[tuple[int, int]]:
    """把闭区间 [lo, hi] 连续切成 n 段，尽量均分，前面的段多担一层。

    **必须连续** —— 段内是流水线，一台算完把 hidden state 交给下一台；
    层区间跳着分就没法这么接（I.2.2 的层区间连续切分）。
    """
    total = hi - lo + 1
    if n <= 0 or n > total:
        raise ValueError(f"{total} 层分不到 {n} 台机器上（每台至少一层）")
    out: list[tuple[int, int]] = []
    cur = lo
    for i in range(n):
        size = total // n + (1 if i < total % n else 0)
        out.append((cur, cur + size - 1))
        cur += size
    return out


@dataclass(frozen=True)
class SegmentLayout:
    """一条段：有序的 (节点, 层区间) 列表。第 0 个是 head，最后一个是 tail。"""

    nodes: tuple[str, ...]
    splits: tuple[tuple[int, int], ...]

    @property
    def head(self) -> str:
        return self.nodes[0]

    @property
    def tail(self) -> str:
        return self.nodes[-1]

    @property
    def layer_lo(self) -> int:
        return self.splits[0][0]

    @property
    def layer_hi(self) -> int:
        return self.splits[-1][1]

    def check_contiguous(self, who: str) -> None:
        for i, (lo, hi) in enumerate(self.splits):
            if lo > hi:
                raise ValueError(f"{who}: {self.nodes[i]} 的层区间 [{lo},{hi}] 是空的")
            if i and lo != self.splits[i - 1][1] + 1:
                raise ValueError(
                    f"{who}: {self.nodes[i-1]} 到 {self.nodes[i]} 的层区间断了 —— "
                    f"前一台到第 {self.splits[i-1][1]} 层，后一台从第 {lo} 层开始。"
                    f"段内是流水线，层必须连着"
                )


@dataclass
class ChannelLayout:
    """一条通道 = 一条前段 + 一条后段，连接在这里就定死了。"""

    front: SegmentLayout
    back: SegmentLayout
    task: str = "general"
    experts: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """本通道的逐层驻留专家。按 task 来 —— 不同 task 的后段装的本来就不一样。"""


@dataclass
class ManualSpec:
    channels: list[ChannelLayout]
    model_dir: str | None = None
    experts: dict[int, tuple[int, ...]] = field(default_factory=dict)
    """逐层驻留专家。缺的层默认全装。"""

    # -- 解析 -------------------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: Mapping, *, n_layers: int) -> "ManualSpec":
        chans = d.get("channels")
        if not chans:
            raise ValueError("布局文件里没有 channels")
        l0 = d.get("l0")

        out: list[ChannelLayout] = []
        for i, ch in enumerate(chans):
            who = f"channel[{i}]"
            front = cls._segment(ch.get("front"), f"{who}.front", 1, l0, n_layers)
            back_lo = front.layer_hi + 1
            back = cls._segment(ch.get("back"), f"{who}.back", back_lo, n_layers,
                                n_layers)
            per_ch = {int(k): tuple(sorted(int(e) for e in v))
                      for k, v in (ch.get("experts") or {}).items()}
            out.append(ChannelLayout(front=front, back=back,
                                     task=str(ch.get("task", "general")),
                                     experts=per_ch))

        experts: dict[int, tuple[int, ...]] = {}
        for k, v in (d.get("experts") or {}).items():
            experts[int(k)] = tuple(sorted(int(e) for e in v))
        spec = cls(channels=out, model_dir=d.get("model_dir"), experts=experts)
        if d.get("profile"):
            spec.apply_profile(d["profile"], coverage=float(d.get("coverage", 0.95)),
                               n_experts=None, top_k=int(d.get("top_k", 1)))
        return spec

    @staticmethod
    def _segment(spec, who: str, lo: int, hi, n_layers: int) -> SegmentLayout:
        """接受三种写法：'n1'、['n1','n2']、[{'node':..,'layers':[lo,hi]}, ...]。"""
        if spec is None:
            raise ValueError(f"{who} 没写")
        if isinstance(spec, str):
            spec = [spec]
        if not isinstance(spec, (list, tuple)) or not spec:
            raise ValueError(f"{who} 应该是节点名或节点名列表")

        if all(isinstance(x, Mapping) for x in spec):
            nodes = tuple(str(x["node"]) for x in spec)
            splits = tuple((int(x["layers"][0]), int(x["layers"][1])) for x in spec)
        elif all(isinstance(x, str) for x in spec):
            if hi is None:
                raise ValueError(
                    f"{who} 只给了机器名，那就得有 l0 才知道层怎么切 —— "
                    f"要么在顶层给 l0，要么逐台写出 layers")
            nodes = tuple(str(x) for x in spec)
            splits = tuple(split_layers(lo, int(hi), len(nodes)))
        else:
            raise ValueError(f"{who} 里混了机器名和 {{node, layers}}，请统一")

        seg = SegmentLayout(nodes=nodes, splits=splits)
        seg.check_contiguous(who)
        return seg

    @classmethod
    def from_file(cls, path: str | Path, *, n_layers: int) -> "ManualSpec":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")), n_layers=n_layers)

    # -- 查询 -------------------------------------------------------------- #
    @property
    def l0(self) -> int:
        return self.channels[0].front.layer_hi

    def experts_at(self, layer: int, n_experts: int,
                   channel: "ChannelLayout | None" = None) -> tuple[int, ...]:
        """优先级：本通道指定 > 全局指定 > 全装。

        通道优先是必须的：不同 task 的后段驻留集本来就不同（I.1.1 的 S_{u,l}），
        全局那份只在「所有通道同一个 task」时够用。
        """
        if channel is not None and layer in channel.experts:
            return channel.experts[layer]
        return self.experts.get(layer, tuple(range(n_experts)))

    def apply_profile(self, path: str | Path, *, coverage: float,
                      n_experts: int | None, top_k: int = 1) -> None:
        """按激活画像填每条通道**后段**层的驻留集。

        只填后段：前段是 task 无关的（I.1.1），装全集或并集，不按 task 裁。
        画像里没有的层保持全装 —— 宁可多装，也不瞎选。
        """
        from ..runtime.profile import load_profile, placement_from_profile

        raw = load_profile(path)
        n_e = n_experts or int(raw.get("n_experts", 0))
        if not n_e:
            raise ValueError(f"{path} 里没有 n_experts，也没从模型传进来")
        for ch in self.channels:
            back_layers = list(range(ch.back.layer_lo, ch.back.layer_hi + 1))
            ch.experts = placement_from_profile(
                raw, ch.task, coverage=coverage, n_experts=n_e, layers=back_layers,
                min_experts=top_k)

    def all_nodes(self) -> list[str]:
        return [v for ch in self.channels
                for seg in (ch.front, ch.back) for v in seg.nodes]


# --------------------------------------------------------------------------- #
def build_manual_manifest(
    spec: ManualSpec, model: ModelSpec, *, model_name: str = "manual",
) -> tuple[DeploymentManifest, dict[str, tuple[str, str]]]:
    """布局 → (部署清单, 静态连线表)。

    返回的连线表就是 `{前段 id: (后段 id, task)}` —— 直接喂给
    `Coordinator(static_wiring=...)`。连接不是算出来的，是你写的。
    """
    _validate(spec, model)

    nodes: list[NodePlan] = []
    segments: dict[str, dict] = {}
    wired: dict[str, tuple[str, str]] = {}

    def add(seg: SegmentLayout, sid: str, role: str, task: str | None,
            chan: ChannelLayout | None = None) -> None:
        segments[sid] = {
            "role": role, "task": task, "nodes": list(seg.nodes),
            "splits": [list(s) for s in seg.splits],
            "head": seg.head, "tail": seg.tail, "hops": len(seg.nodes) - 1,
            # 手动放置不做任何测量，所以这三项是 0 而不是猜的数字。
            # 想要真实延迟画像走 deploy/control.py。
            "compute_ms": 0.0, "hop_ms": 0.0, "delay_ms": 0.0,
        }
        for i, v in enumerate(seg.nodes):
            lo, hi = seg.splits[i]
            nodes.append(NodePlan(
                node=v, role=role, segment=sid, position=i,
                is_head=(i == 0), is_tail=(i == len(seg.nodes) - 1),
                layers=tuple(_load(spec, model, l, chan)
                             for l in range(lo, hi + 1)),
            ))

    for i, ch in enumerate(spec.channels):
        fid, bid = f"F{i}", f"B{ch.task}{i}"
        add(ch.front, fid, "front", None)        # 前段 task 无关：不按 task 裁专家
        add(ch.back, bid, f"back:{ch.task}", ch.task, ch)
        wired[fid] = (bid, ch.task)

    man = DeploymentManifest(
        model=model_name, l0=spec.l0, nodes=nodes, segments=segments,
        pairings=[], band=(0.0, 0.0), standby_fronts=[],
    )
    return man, wired


def _load(spec: ManualSpec, model: ModelSpec, layer: int,
          chan: "ChannelLayout | None") -> LayerLoad:
    es = spec.experts_at(layer, model.n_experts, chan)
    return LayerLoad(layer=layer, experts=es,
                     weight_gb=model.weight_gb_per_layer(len(es)),
                     kv_gb=model.kv_gb_per_layer)


def _validate(spec: ManualSpec, model: ModelSpec) -> None:
    """把能在下发**之前**发现的错误全找出来，一次报完。

    手动放置最容易错的就是这几件事，而它们的症状都很难查：层没接上会在推理时
    表现成「输出是乱的」（张量形状对得上，语义错了），节点用两次会表现成
    「两条请求互相污染 KV」。所以宁可啰嗦。
    """
    errs: list[str] = []
    seen: dict[str, str] = {}

    for i, ch in enumerate(spec.channels):
        f, b = ch.front, ch.back
        if f.layer_lo != 1:
            errs.append(f"channel[{i}] 前段从第 {f.layer_lo} 层开始 —— 必须从第 1 层")
        if b.layer_lo != f.layer_hi + 1:
            errs.append(
                f"channel[{i}] 前段到第 {f.layer_hi} 层、后段从第 {b.layer_lo} 层 —— "
                f"中间有 {b.layer_lo - f.layer_hi - 1} 层没人算")
        if b.layer_hi != model.n_layers:
            errs.append(
                f"channel[{i}] 后段到第 {b.layer_hi} 层，模型有 {model.n_layers} 层 —— "
                f"最后 {model.n_layers - b.layer_hi} 层没人算")
        for seg, what in ((f, "front"), (b, "back")):
            for v in seg.nodes:
                if v in seen:
                    errs.append(
                        f"节点 {v} 同时出现在 {seen[v]} 和 channel[{i}].{what} —— "
                        f"一台机器至多承载一条段（I.2.2 排他），"
                        f"否则两条请求会在它上面互相污染 KV")
                seen[v] = f"channel[{i}].{what}"

    l0s = {ch.front.layer_hi for ch in spec.channels}
    if len(l0s) > 1:
        errs.append(
            f"各通道的 L₀ 不一致：{sorted(l0s)}。前段是 task 无关的、后段按 task 建，"
            f"切点不同就不是同一套划分了 —— 真要这样请分成两次部署")

    boxes = [("experts", spec.experts)] + [
        (f"channel[{i}].experts", ch.experts) for i, ch in enumerate(spec.channels)]
    for who, box in boxes:
        for l, es in box.items():
            if not 1 <= l <= model.n_layers:
                errs.append(f"{who} 里的层号 {l} 越界（模型 1..{model.n_layers} 层）")
            elif not es or max(es) >= model.n_experts:
                errs.append(f"{who} 第 {l} 层的专家 id 越界或为空"
                            f"（模型有 {model.n_experts} 个）")
            elif len(es) < model.top_k:
                errs.append(f"{who} 第 {l} 层只装了 {len(es)} 个专家，"
                            f"但 top-k={model.top_k} —— 每个 token 都会 miss，"
                            f"drop-expert 兜不住这种程度")

    if errs:
        raise ValueError("布局有问题：\n  " + "\n  ".join(errs))


def memory_report(
    spec: ManualSpec, model: ModelSpec, available_gb: Mapping[str, float],
) -> list[tuple[str, float, float]]:
    """逐节点 (节点, 需要 GB, 可用 GB)，按超出量降序。

    只报不拦：可用内存是 agent 自己报的，未必扣干净了别的进程；真放不下会在
    加载时以 OOM 的形式表现出来，那时错误信息比这里的估算准。
    """
    need: dict[str, float] = {}
    for ch in spec.channels:
        for seg, chan in ((ch.front, None), (ch.back, ch)):
            for i, v in enumerate(seg.nodes):
                lo, hi = seg.splits[i]
                need[v] = sum(
                    model.weight_gb_per_layer(
                        len(spec.experts_at(l, model.n_experts, chan)))
                    + model.kv_gb_per_layer
                    for l in range(lo, hi + 1)
                )
    rows = [(v, g, float(available_gb.get(v, 0.0))) for v, g in need.items()]
    rows.sort(key=lambda r: r[2] - r[1])
    return rows
