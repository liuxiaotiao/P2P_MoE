"""部署清单与组合矩阵 —— 规划器的可执行产物。

两个问题，两个答案：

**「每个节点部署特定 layer、每个 layer 指定的专家」**
    → `NodePlan.layers`：逐层给出该节点要加载哪些专家（专家 id 列表，不是个数）。
      这就是发给一台节点的加载指令，配合 safetensors 的逐张量 mmap 可以直接
      翻译成「打开哪些 key」。

**「前后 cluster 可以任意组合」**
    → `DeploymentManifest.pairings`：N_F × N_B(u) 的完整组合矩阵。它不是事后
      罗列，而是被 `validate` 逐条核过的 —— 任何一对不满足接口闸门，校验就不通过。
      这正是整套方案的价值所在：公共中值域（II.3）保证了**每个** (f,b) 的正向接口
      都落在同一条窄带里，回环裁剪（Step 6）保证了每条前段对**全体**后段的回环都小，
      于是在线才敢盲绑 —— 弹一条前段，事后无论识别成哪个 task，都能配上任意一条
      该池的后段而不后悔（定理 III.3.1、推论 III.3.2）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .experts import ExpertPlacement
from .network import MeasurementCache
from .types import ModelSpec, Node, PlannerConfig, Segment

__all__ = [
    "LayerLoad",
    "NodePlan",
    "Pairing",
    "DeploymentManifest",
    "build_manifest",
]


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LayerLoad:
    """一个节点上一层的加载指令。"""

    layer: int
    experts: tuple[int, ...]
    """该层要驻留的专家 id（已排序）。前段是全 task 并集，后段是该 task 专用集。"""
    weight_gb: float
    kv_gb: float

    @property
    def total_gb(self) -> float:
        return self.weight_gb + self.kv_gb


@dataclass(frozen=True)
class NodePlan:
    """一台节点的完整部署内容。排他：一节点至多服务一条段（I.2.2）。"""

    node: str
    role: str
    """"front" 或 "back:<task>" """
    segment: str
    position: int
    """在段内链上的位置（0 = head）。"""
    is_head: bool
    is_tail: bool
    layers: tuple[LayerLoad, ...]

    @property
    def layer_range(self) -> tuple[int, int]:
        return (self.layers[0].layer, self.layers[-1].layer)

    @property
    def weight_gb(self) -> float:
        return sum(l.weight_gb for l in self.layers)

    @property
    def kv_gb(self) -> float:
        return sum(l.kv_gb for l in self.layers)

    @property
    def total_gb(self) -> float:
        return self.weight_gb + self.kv_gb

    @property
    def n_experts_total(self) -> int:
        return sum(len(l.experts) for l in self.layers)


@dataclass(frozen=True)
class Pairing:
    """组合矩阵的一行：某条前段 × 某条后段。"""

    front: str
    back: str
    task: str
    forward: tuple[str, str]
    """正向接口 (tail(f), head(b))。"""
    loop: tuple[str, str]
    """回环接口 (tail(b), head(f))。"""
    w_p50: float
    w_p95: float
    w_jitter: float
    d_loop_p50: float
    d_loop_jitter: float
    t50: float


@dataclass
class DeploymentManifest:
    model: str
    l0: int
    nodes: list[NodePlan]
    segments: dict[str, dict]
    pairings: list[Pairing]
    band: tuple[float, float]
    standby_fronts: list[str]
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    # -- 查询 -------------------------------------------------------------- #
    def plan_for(self, node: str) -> NodePlan | None:
        for p in self.nodes:
            if p.node == node:
                return p
        return None

    def partners_of(self, front: str) -> list[Pairing]:
        return [p for p in self.pairings if p.front == front]

    def combination_matrix(self) -> dict[str, dict[str, float]]:
        """前段 → 后段 → 组合 p50。缺格即表示该组合不可用。"""
        out: dict[str, dict[str, float]] = {}
        for p in self.pairings:
            out.setdefault(p.front, {})[p.back] = p.t50
        return out

    # -- 序列化 ------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "l0": self.l0,
            "band": {"w_lo": self.band[0], "w_hi": self.band[1]},
            "nodes": [_node_dict(p) for p in self.nodes],
            "segments": self.segments,
            "pairings": [
                {
                    "front": p.front, "back": p.back, "task": p.task,
                    "forward": list(p.forward), "loop": list(p.loop),
                    "w_p50": round(p.w_p50, 2), "w_p95": round(p.w_p95, 2),
                    "d_loop_p50": round(p.d_loop_p50, 2),
                    "t50": round(p.t50, 2),
                }
                for p in self.pairings
            ],
            "standby_fronts": self.standby_fronts,
            "violations": self.violations,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kw)

    # -- 反序列化 ---------------------------------------------------------- #
    @classmethod
    def from_dict(cls, d: Mapping) -> "DeploymentManifest":
        """从存下来的清单还原 —— 让部署可以**不重新规划**就重放。

        规划的输入里有一项是不可复现的：逐对延迟实测。同一个池子换个时间跑，
        探测值会变，段的构成与 id 编号都可能跟着变。于是「我要 F0 连 BX1」
        这种指定在下一次规划里可能指到别的东西上。

        存清单 → 改连接 → 载清单，这条路把放置固定住，人工指定的连接才有稳定
        的所指。代价是这份清单反映的是**当时**的网络与节点集合；机器换了或
        链路劣化了要重跑规划。
        """
        nodes = [
            NodePlan(
                node=p["node"], role=p["role"], segment=p["segment"],
                position=int(p["position"]),
                is_head=bool(p["is_head"]), is_tail=bool(p["is_tail"]),
                layers=tuple(
                    LayerLoad(layer=int(l["layer"]), experts=tuple(l["experts"]),
                              weight_gb=float(l["weight_gb"]),
                              kv_gb=float(l.get("kv_gb", 0.0)))
                    for l in p["layers"]
                ),
            )
            for p in d["nodes"]
        ]
        pairings = [
            Pairing(front=q["front"], back=q["back"], task=q["task"],
                    forward=tuple(q["forward"]), loop=tuple(q["loop"]),
                    w_p50=float(q["w_p50"]), w_p95=float(q.get("w_p95", 0.0)),
                    w_jitter=float(q.get("w_jitter", 0.0)),
                    d_loop_p50=float(q["d_loop_p50"]),
                    d_loop_jitter=float(q.get("d_loop_jitter", 0.0)),
                    t50=float(q["t50"]))
            for q in d.get("pairings", [])
        ]
        band = d.get("band", {})
        return cls(
            model=d["model"], l0=int(d["l0"]), nodes=nodes,
            segments=dict(d["segments"]), pairings=pairings,
            band=(float(band.get("w_lo", 0.0)), float(band.get("w_hi", 0.0))),
            standby_fronts=list(d.get("standby_fronts", [])),
            violations=list(d.get("violations", [])),
        )

    @classmethod
    def from_json(cls, text: str) -> "DeploymentManifest":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
_ND = 6


def _node_dict(p: "NodePlan") -> dict:
    """逐节点的 JSON。

    **逐节点的三个合计值由已经四舍五入过的逐层值加出来**，不是把精确合计再舍入。
    两种算法在最后一位上会差 1e-6，于是「导出 → 载入 → 再导出」得不到同一份
    文件，`--load-plan` 的可重放性就成了一句空话。以读者能复算的口径为准。
    """
    layers = [
        {"layer": l.layer, "experts": list(l.experts),
         "weight_gb": round(l.weight_gb, _ND),
         # KV 也要落盘：逐节点的 total_gb 是由逐层数据算出来的，少了它
         # from_dict 还原出来的清单对不上原来的账
         "kv_gb": round(l.kv_gb, _ND)}
        for l in p.layers
    ]
    w = round(sum(x["weight_gb"] for x in layers), _ND)
    kv = round(sum(x["kv_gb"] for x in layers), _ND)
    return {
        "node": p.node, "role": p.role, "segment": p.segment,
        "position": p.position, "is_head": p.is_head, "is_tail": p.is_tail,
        "layer_range": list(p.layer_range),
        "weight_gb": w, "kv_gb": kv, "total_gb": round(w + kv, _ND),
        "layers": layers,
    }


def _layer_loads(
    seg: Segment,
    idx: int,
    placement: ExpertPlacement,
    model: ModelSpec,
) -> tuple[LayerLoad, ...]:
    lo, hi = seg.splits[idx]
    out: list[LayerLoad] = []
    for l in range(lo, hi + 1):
        experts = tuple(sorted(placement.at(l)))
        out.append(
            LayerLoad(
                layer=l,
                experts=experts,
                weight_gb=model.weight_gb_per_layer(len(experts)),
                kv_gb=model.kv_gb_per_layer,
            )
        )
    return tuple(out)


def build_manifest(
    *,
    model: ModelSpec,
    model_name: str,
    l0: int,
    fronts: Sequence[Segment],
    standby: Sequence[Segment],
    backs: Mapping[str, Sequence[Segment]],
    front_placement: ExpertPlacement,
    back_placements: Mapping[str, ExpertPlacement],
    node_map: Mapping[str, Node],
    net: MeasurementCache,
    cfg: PlannerConfig,
    band: tuple[float, float],
    w_cap: float,
) -> DeploymentManifest:
    """把规划结果翻译成逐节点的加载指令 + 完整组合矩阵，并逐项校验。"""
    node_plans: list[NodePlan] = []
    segments: dict[str, dict] = {}

    def add_segment(seg: Segment, sid: str, role: str, placement: ExpertPlacement) -> None:
        segments[sid] = {
            "role": role,
            "task": seg.task,
            "nodes": list(seg.nodes),
            "splits": [list(s) for s in seg.splits],
            "head": seg.head,
            "tail": seg.tail,
            "hops": seg.hops,
            "compute_ms": round(seg.compute_ms, 2),
            "hop_ms": round(seg.hop_ms, 2),
            "delay_ms": round(seg.delay_ms, 2),
        }
        for i, v in enumerate(seg.nodes):
            node_plans.append(
                NodePlan(
                    node=v,
                    role=role,
                    segment=sid,
                    position=i,
                    is_head=(i == 0),
                    is_tail=(i == len(seg.nodes) - 1),
                    layers=_layer_loads(seg, i, placement, model),
                )
            )

    front_ids: list[str] = []
    for i, f in enumerate(fronts):
        sid = f"F{i}"
        front_ids.append(sid)
        add_segment(f, sid, "front", front_placement)
    for i, f in enumerate(standby):
        add_segment(f, f"F-standby{i}", "front-standby", front_placement)

    back_ids: dict[str, list[str]] = {}
    for u, segs in backs.items():
        for i, b in enumerate(segs):
            sid = f"B{u}{i}"
            back_ids.setdefault(u, []).append(sid)
            add_segment(b, sid, f"back:{u}", back_placements[u])

    # --- 组合矩阵：每条前段 × 每条同 task 后段 --------------------------- #
    pairings: list[Pairing] = []
    for fi, f in enumerate(fronts):
        for u, segs in backs.items():
            for bi, b in enumerate(segs):
                fwd = net.get(f.tail, b.head, cfg.k_audit)
                loop = net.get(b.tail, f.head, cfg.k_audit)
                pairings.append(
                    Pairing(
                        front=front_ids[fi],
                        back=back_ids[u][bi],
                        task=u,
                        forward=(f.tail, b.head),
                        loop=(b.tail, f.head),
                        w_p50=fwd.p50,
                        w_p95=fwd.p95,
                        w_jitter=net.get(f.tail, b.head, cfg.k_gate).jitter,
                        d_loop_p50=loop.p50,
                        d_loop_jitter=net.get(b.tail, f.head, cfg.k_gate).jitter,
                        t50=f.delay_ms + fwd.p50 + b.delay_ms + loop.p50,
                    )
                )

    man = DeploymentManifest(
        model=model_name,
        l0=l0,
        nodes=node_plans,
        segments=segments,
        pairings=pairings,
        band=band,
        standby_fronts=[v for s in standby for v in s.nodes],
    )
    man.violations = _validate(
        man, model, node_map, front_placement, back_placements, cfg, w_cap,
        n_fronts=len(fronts), back_counts={u: len(v) for u, v in backs.items()},
    )
    return man


# --------------------------------------------------------------------------- #
def _validate(
    man: DeploymentManifest,
    model: ModelSpec,
    node_map: Mapping[str, Node],
    front_placement: ExpertPlacement,
    back_placements: Mapping[str, ExpertPlacement],
    cfg: PlannerConfig,
    w_cap: float,
    *,
    n_fronts: int,
    back_counts: Mapping[str, int],
) -> list[str]:
    """七项一致性校验。任何一条不过，清单就不能上线。"""
    bad: list[str] = []

    # 1. 排他：一节点至多服务一条段（I.2.2）
    seen: dict[str, str] = {}
    for p in man.nodes:
        if p.node in seen:
            bad.append(f"[排他] 节点 {p.node} 同时属于 {seen[p.node]} 与 {p.segment}")
        seen[p.node] = p.segment

    # 2. 内存（含 KV）：Σ mem(l) + kv ≤ M_v − 预留（I.2.2）
    for p in man.nodes:
        cap = node_map[p.node].usable_gb
        if p.total_gb > cap + 1e-6:
            bad.append(
                f"[内存] {p.node} 需 {p.total_gb:.2f}GB > 可用 {cap:.2f}GB"
                f"（权重 {p.weight_gb:.2f} + KV {p.kv_gb:.2f}）"
            )

    # 3. 层区间连续、完整、无重叠
    for sid, info in man.segments.items():
        splits = [tuple(s) for s in info["splits"]]
        for (a_lo, a_hi), (b_lo, b_hi) in zip(splits, splits[1:]):
            if b_lo != a_hi + 1:
                bad.append(f"[层切分] {sid} 在 {a_hi}→{b_lo} 处不连续")
        want = (1, man.l0) if info["role"].startswith("front") else (man.l0 + 1, model.n_layers)
        if splits and (splits[0][0], splits[-1][1]) != want:
            bad.append(f"[层覆盖] {sid} 覆盖 {splits[0][0]}–{splits[-1][1]}，应为 {want[0]}–{want[1]}")

    # 4. 并集支配性：前段每层的专家集 ⊇ 各 task 该层集合（I.1.1 / III.5.4）
    for l in range(1, man.l0 + 1):
        uni = front_placement.at(l)
        for u, plc in back_placements.items():
            miss = plc.at(l) - uni
            if miss:
                bad.append(f"[并集] 前段第 {l} 层缺 task {u} 的专家 {sorted(miss)[:5]}")

    # 5. 后段每层的专家集 = 该 task 专用集（不多不少）
    for p in man.nodes:
        if not p.role.startswith("back:"):
            continue
        u = p.role.split(":", 1)[1]
        for ll in p.layers:
            want = back_placements[u].at(ll.layer)
            if set(ll.experts) != want:
                bad.append(
                    f"[驻留集] {p.node} 第 {ll.layer} 层装了 {len(ll.experts)} 个专家，"
                    f"task {u} 该层应为 {len(want)} 个"
                )
                break

    # 6. 组合矩阵完整：每条前段 × 每条同 task 后段都在表里
    want_pairs = n_fronts * sum(back_counts.values())
    if len(man.pairings) != want_pairs:
        bad.append(f"[组合矩阵] 只有 {len(man.pairings)} 组，应为 {want_pairs} 组")
    got = {(p.front, p.back) for p in man.pairings}
    if len(got) != len(man.pairings):
        bad.append("[组合矩阵] 存在重复组合")

    # 7. 逐对过闸：正向落在公共带内且 ≤ w_cap；两个接口的抖动都 ≤ J_cap（I.2.3）
    lo, hi = man.band
    for p in man.pairings:
        if not (lo - 1e-6 <= p.w_p50 <= hi + 1e-6):
            bad.append(
                f"[公共带] {p.front}×{p.back} 的正向 w={p.w_p50:.1f}ms 落在带 "
                f"[{lo:.1f},{hi:.1f}] 之外 —— 任意组合的前提被破坏"
            )
        if p.w_p50 > w_cap + 1e-6:
            bad.append(f"[w_cap] {p.front}×{p.back} 正向 {p.w_p50:.1f} > {w_cap:.1f}")
        if p.w_jitter > cfg.j_cap_ms + 1e-6:
            bad.append(f"[抖动闸] {p.front}×{p.back} 正向抖动 {p.w_jitter:.1f} > {cfg.j_cap_ms}")
        if p.d_loop_jitter > cfg.j_cap_ms + 1e-6:
            bad.append(f"[抖动闸] {p.front}×{p.back} 回环抖动 {p.d_loop_jitter:.1f} > {cfg.j_cap_ms}")

    return bad
