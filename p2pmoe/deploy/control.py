"""控制器 —— 真机部署的编排入口。

    python -m p2pmoe.deploy.control \\
        --agents v1=10.0.0.11:9101,v2=10.0.0.12:9101,g1=10.0.0.21:9101 \\
        --advertise 10.0.0.5 \\
        --requests 4

五步，每一步都对应文档里的一段：

    1. 采集能力   连每个 agent 问内存/算力 → 拼出规划器的 Node 表
    2. 真实探测   下发探测指令，**由节点自己**量逐对 p50/p95（deploy/probe.py）
    3. 离线规划   planner.plan()：分档估上限 → 配额 → 建段 → 公共带 → 回环裁剪
    4. 下发清单   每个节点只收到属于自己的那份 NodePlan（层 + 专家 id）
    5. 在线服务   协调器盲绑派发，节点之间自己转发

控制器**不在数据面上**：请求到达后它只发一条 prefill 给 head(f)，之后整个环
在节点之间自己转，控制器只收控制面的上报（识别结果、token、miss 统计）。
这是文档 II.5「在线零计算」的直接体现 —— 也意味着控制器挂了不影响在途请求。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..planner.experts import (
    ExpertPlacement,
    build_placement,
    full_placement,
    union_placement,
)
from ..planner.hf_config import granularity_verdict, model_spec_from_hf
from ..planner.manifest import DeploymentManifest
from ..planner.network import MeasurementCache
from ..planner.pipeline import PlanningError, plan
from ..planner.static_pairing import assign_static_pairs
from ..planner.types import ModelSpec, Node, PlannerConfig, TaskProfile
from ..runtime.corpus import (
    make_corpus,
    measure_baseline_miss,
    measure_front_refs,
    profile_from_corpus,
    sample_prompt,
)
from ..runtime.coordinator import Coordinator
from ..runtime.identify import HistogramClassifier
from ..runtime.model import ToyMoEConfig
from ..runtime.text import TextIO
from ..runtime.node import NodeConfig
from ..runtime.wire import Addr, LinkTable, PeerPool, rpc
from .probe import RemoteNetworkOracle

log = logging.getLogger("p2pmoe.control")

TASKS = [("X", 0.5), ("Y", 0.3), ("Z", 0.2)]
CTX_MAX = 64


# --------------------------------------------------------------------------- #
def parse_agents(s: str) -> dict[str, Addr]:
    """--agents v1=host:port,v2=host:port"""
    out: dict[str, Addr] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, hp = item.partition("=")
        host, _, port = hp.rpartition(":")
        out[name.strip()] = (host.strip(), int(port))
    if not out:
        raise ValueError("--agents 为空")
    return out


def parse_tasks(s: str) -> list[tuple[str, float]]:
    """--tasks 'general' 或 'code=0.6,chat=0.4' → [(名字, 到达率占比)]。

    省略权重就按均分。权重只影响后段的**配额分配**（II.7.1 最大余额法）——
    流量大的 task 分到更多条后段。
    """
    items = [x.strip() for x in s.split(",") if x.strip()]
    out: list[tuple[str, float]] = []
    for it in items:
        name, _, w = it.partition("=")
        out.append((name.strip(), float(w) if w else 0.0))
    if all(w == 0.0 for _, w in out):
        out = [(u, 1.0 / len(out)) for u, _ in out]
    tot = sum(w for _, w in out)
    return [(u, w / tot) for u, w in out]


RELAY: Addr | None = None
"""进程级的中继地址。控制面所有 rpc 都从这里取 —— 与其给十几个函数各加一个
参数，不如认一个事实：一次部署要么全走中继，要么全不走，没有一半一半。"""


def _rpc(node: str, addr: Addr, header: dict, *, timeout: float = 30.0) -> dict:
    return rpc(addr, header, timeout=timeout, relay=RELAY, to=node)


def check_model_dirs(addrs: dict[str, Addr], model_dir: str) -> list[str]:
    """预检：每台节点自己看得见 checkpoint 吗？

    **这是真机上最常见的翻车点。** 权重分发还没做（TODO.md P0），
    `--model-dir` 是各节点上的**本地路径** —— 控制机能读到不代表节点能。
    不预检的话，故障会推迟到下发清单那一刻才爆，而那时前面的探测（几分钟）
    已经白跑了。
    """
    bad: list[str] = []
    for name, addr in addrs.items():
        try:
            r = _rpc(name, addr, {"type": "check_model", "dir": model_dir}, timeout=30.0)
        except Exception as e:
            bad.append(f"{name}: 问不到（{e}）")
            continue
        if not r.get("ok"):
            bad.append(f"{name}: {r.get('why', '未知')}")
    return bad


def toy_model_spec(cfg: ToyMoEConfig) -> ModelSpec:
    """toy 模型的真实字节数 → 规划器的内存模型（单位统一取 MB）。"""
    mb = 1e6
    return ModelSpec(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_experts=cfg.n_experts,
        top_k=cfg.top_k, base_gb_per_layer=cfg.base_params * 8 / mb,
        expert_gb=cfg.expert_params * 8 / mb, ctx_max=CTX_MAX, kv_bytes_per_elem=8,
    )


# --------------------------------------------------------------------------- #
def resolve_wiring(path: Path, manifest, fronts_ids: Sequence[str]) -> dict[str, tuple[str, str]]:
    """读**人工指定**的前后段连接。

    格式（JSON）::

        {"pairs": [{"front": "F0",  "back": "BX0", "task": "X"},
                   {"front": "n07", "back": "n11"}]}

    `front` / `back` 既接受段 id（F0、BX0 —— 规划器给的），也接受**节点 id**
    （n07 —— 你自己给机器起的名字）。后者更实用：段 id 是每次规划现编的，
    换一次探测就可能重排；节点 id 是你写在 hosts.txt 里的，跨规划稳定。

    `task` 省略时取后段自己的 task —— 后段本来就是按 task 建的，写不写都一样，
    写了就多一道校验。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs = raw.get("pairs", raw if isinstance(raw, list) else None)
    if not pairs:
        raise SystemExit(f"{path} 里没有 pairs")

    # 段 id → 段信息；节点 id → 它所属的段（一节点至多一条段，I.2.2）
    by_node: dict[str, str] = {}
    for sid, info in manifest.segments.items():
        for v in info["nodes"]:
            by_node[v] = sid

    def to_sid(name: str, want: str) -> str:
        sid = name if name in manifest.segments else by_node.get(name)
        if sid is None:
            raise SystemExit(
                f"--wiring 里的 {name!r} 既不是段 id 也不是本次规划用到的节点。\n"
                f"  可用前段: {sorted(fronts_ids)}\n"
                f"  可用后段: {sorted(k for k in manifest.segments if k.startswith('B'))}\n"
                f"  提示：先用 --save-wiring 导出自动配对，改完再用 --wiring 喂回来"
            )
        role = manifest.segments[sid]["role"]
        if want == "front" and role != "front":
            raise SystemExit(f"--wiring: {name!r} 解析成 {sid}，但它是 {role}，不是前段")
        if want == "back" and not role.startswith("back:"):
            raise SystemExit(f"--wiring: {name!r} 解析成 {sid}，但它是 {role}，不是后段")
        return sid

    out: dict[str, tuple[str, str]] = {}
    used_back: dict[str, str] = {}
    for i, item in enumerate(pairs):
        f = to_sid(str(item["front"]), "front")
        b = to_sid(str(item["back"]), "back")
        task = manifest.segments[b]["role"].split(":", 1)[1]
        if item.get("task") and item["task"] != task:
            raise SystemExit(
                f"--wiring 第 {i} 条写的 task={item['task']}，但后段 {b} 是按 "
                f"{task} 建的。后段的 task 由它装了哪些专家决定，改不了 —— "
                f"要么改这一行，要么重规划")
        if f in out:
            raise SystemExit(f"--wiring: 前段 {f} 被指了两次（{out[f][0]} 和 {b}）。"
                             f"一条前段同时只能服务一条请求（I.2.4），不能一对多")
        if b in used_back:
            raise SystemExit(f"--wiring: 后段 {b} 被指了两次（{used_back[b]} 和 {f}）")
        out[f], used_back[b] = (b, task), f
    return out


def wiring_to_json(wired: dict[str, tuple[str, str]], manifest) -> str:
    """导出成 --wiring 能吃回去的格式，顺便把节点 id 也写上便于人读。"""
    seg = manifest.segments
    return json.dumps({"pairs": [
        {"front": f, "back": b, "task": t,
         "front_nodes": list(seg[f]["nodes"]), "back_nodes": list(seg[b]["nodes"])}
        for f, (b, t) in sorted(wired.items())
    ]}, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
@dataclass
class ModelSetup:
    """「跑哪个模型、每层装哪些专家」—— toy 与真模型两条路的统一出口。"""

    label: str
    spec: ModelSpec
    n_layers: int
    n_experts: int
    backend: str
    """"numpy"（toy）| "torch"（真 checkpoint）"""
    node_model: dict
    """下发给节点的模型描述：toy 是 ToyMoEConfig 的字段，torch 是 HF config。"""
    front_plc: ExpertPlacement
    back_plcs: dict[str, ExpertPlacement]
    tasks: list[TaskProfile]
    mcfg: ToyMoEConfig | None = None
    corpus: object | None = None
    approximate: bool = True
    """后段是否只驻留子集。False = 全装，输出与单机参考实现逐位一致。"""
    profiles: dict | None = None
    """逐 task 的 `ActivationProfile`（真模型 + --profile 时才有）。
    动态模式的分类器要从它构造 —— 见 main() 里 clf 的分支。"""


def load_cases(path: Path, tasks: Sequence[str]) -> list[tuple[str | None, str]]:
    """读测试集文件 → [(真实 task 或 None, prompt), ...]。

    格式：一行一条。想标注真实 task 就写 `mbpp<TAB>prompt` —— 标注只用来
    核对在线识别对不对，**不影响派发**（派发靠前段自己识别，那才是被测的东西）。

    制表符前的字段只有**在 tasks 里认识**时才当成标注。否则整行都是 prompt ——
    正文里出现制表符不该被误读成标注，而这在代码类 prompt 里很常见。
    """
    known = set(tasks)
    cases: list[tuple[str | None, str]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        # 先只去行尾换行 —— rstrip() 会把 `mbpp<TAB>` 末尾的制表符也吃掉，
        # 于是「有标注、prompt 为空」这一行会被误判成「没标注」。
        ln = ln.rstrip("\r\n")
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        head, tab, rest = ln.partition("\t")
        if tab and head.strip() in known:
            cases.append((head.strip(), rest.rstrip()))
        else:
            cases.append((None, ln.rstrip()))
    return cases


def _load_profile_file(path: Path, tasks: Sequence[str], n_layers: int,
                       n_experts: int, *, coverage: float = 0.95,
                       top_k: int = 1) -> tuple[dict, dict]:
    """读激活画像 → (逐 task 驻留集, 逐 task 原始分布)。

    两种格式都认（与 `runtime/profile.py` 同一套）：

    * **质量格式**（推荐）`{"tasks": {u: {"layers": {l: [mass…]}}}}` —— 存的是
      归一化的逐层激活质量。换覆盖率不用重新采样，而且**动态模式的分类器只能
      从它构造**（分类器要的是分布，不是 id 列表）。
    * **id 格式** `{u: [[ids] × n_layers]}` —— 外部流程直接给驻留集。
      能用来部署，但**没有分布就没有分类器**，只能配 `--static`。

    返回的第二项在 id 格式下是 None —— 调用方据此判断能不能走动态模式。
    """
    from ..runtime.profile import (
        load_profile, placement_from_profile, to_activation_profile,
    )

    raw = load_profile(path)
    is_ids = raw.get("format") == "ids"
    have = raw.get("tasks", raw)
    missing = [u for u in tasks if u not in have]
    if missing:
        raise SystemExit(f"--profile 里没有 task {missing}；有的是 {sorted(have)}")

    plcs: dict[str, ExpertPlacement] = {}
    profs: dict = {}
    for u in tasks:
        sets = placement_from_profile(raw, u, coverage=coverage,
                                      n_experts=n_experts, min_experts=top_k,
                                      layers=list(range(1, n_layers + 1)))
        full = tuple(range(n_experts))
        plcs[u] = ExpertPlacement(
            name=u, n_layers=n_layers,
            sets=tuple(frozenset(sets.get(l, full)) for l in range(1, n_layers + 1)),
            achieved_coverage=tuple(coverage if l in sets else 1.0
                                    for l in range(1, n_layers + 1)),
            coverage_target=coverage,
        )
        if not is_ids:
            profs[u] = to_activation_profile(raw, u, n_layers=n_layers,
                                             n_experts=n_experts)
    return plcs, (profs or None)


def build_model_setup(args, tasks_lam: Sequence[tuple[str, float]]) -> ModelSetup:
    """步骤 0：决定跑哪个模型、每层装哪些专家。"""
    names = [u for u, _ in tasks_lam]

    # ---------------- toy 模型（默认）---------------------------------- #
    if not args.model_dir:
        mcfg = ToyMoEConfig()
        spec = toy_model_spec(mcfg)
        corpus = make_corpus(mcfg, names, seed=args.seed, shared_clusters=0)
        plcs = {u: build_placement(p, args.coverage)
                for u, p in profile_from_corpus(mcfg, corpus).items()}
        uni = union_placement(list(plcs.values()))
        log.info("      驻留集层均 %s，并集 %.1f/%d",
                 {u: round(sum(p.sizes()) / mcfg.n_layers, 1) for u, p in plcs.items()},
                 sum(uni.sizes()) / mcfg.n_layers, mcfg.n_experts)
        front = (full_placement(mcfg.n_layers, mcfg.n_experts) if args.static else uni)
        if args.static:
            log.info("      --static：前段改装**全部** %d 个专家/层（并集是 %.1f）",
                     mcfg.n_experts, sum(uni.sizes()) / mcfg.n_layers)
        return ModelSetup(
            label="toy", spec=spec, n_layers=mcfg.n_layers, n_experts=mcfg.n_experts,
            backend="numpy", node_model=dict(mcfg.__dict__), front_plc=front,
            back_plcs=plcs, mcfg=mcfg, corpus=corpus,
            tasks=[TaskProfile(name=u, lam=l,
                               experts_per_layer=plcs[u].as_experts_per_layer(),
                               placement=plcs[u]) for u, l in tasks_lam],
        )

    # ---------------- 真实 checkpoint ------------------------------------ #
    d = Path(args.model_dir)
    cfg_f = d / "config.json"
    if not cfg_f.exists():
        raise SystemExit(f"{cfg_f} 不存在 —— --model-dir 要指向 HF checkpoint 目录")
    hf = json.loads(cfg_f.read_text(encoding="utf-8"))
    spec, info = model_spec_from_hf(hf, name=d.name, ctx_max=args.ctx,
                                    dtype_bytes=args.dtype_bytes)
    log.info("      %s", info.summary())
    ok, why = granularity_verdict(info)
    log.info("      %s %s", "✓" if ok else "✗", why)

    # 动态模式要两样东西，都来自激活画像：前段的并集、在线分类器的参考直方图。
    # 有 --profile 就都有；没有就只能走 --static（配对定死、请求自报 task）。
    if not args.static and not args.profile:
        raise SystemExit(
            "真实模型走动态模式需要 --profile —— 前段并集与在线分类器都从画像来。"
            "没有画像就用 --static（配对定死、请求自报 task）"
        )
    front = full_placement(info.n_layers, info.n_experts)

    raw_profiles = None
    if args.profile:
        back_plcs, raw_profiles = _load_profile_file(
            Path(args.profile), names, info.n_layers, info.n_experts,
            coverage=args.coverage, top_k=info.top_k)
        approximate = True
        log.info("      后段驻留集来自 %s，层均 %s", args.profile,
                 {u: round(sum(p.sizes()) / info.n_layers, 1)
                  for u, p in back_plcs.items()})
    elif args.resident_frac >= 1.0:
        back_plcs = {u: full_placement(info.n_layers, info.n_experts, name=u)
                     for u in names}
        approximate = False
        log.info("      后段也装**全部**专家 —— 没有 drop-expert 近似，"
                 "输出与单机参考实现逐位一致。首次跑真权重应该用这个口径")
    else:
        # 没有画像却要只装子集 —— 能跑，但输出会烂，必须说清楚
        n = max(info.top_k, round(args.resident_frac * info.n_experts))
        back_plcs = {u: ExpertPlacement(
            name=u, n_layers=info.n_layers,
            sets=tuple(frozenset(range(n)) for _ in range(info.n_layers)),
            achieved_coverage=tuple(0.0 for _ in range(info.n_layers))) for u in names}
        approximate = True
        log.warning(
            "      ⚠ --resident-frac %.2f 但没给 --profile —— 驻留集只能按 id 顺序"
            "硬取前 %d 个，**不是**按激活质量选的。miss 率会极高、drop-expert 兜不住，"
            "输出基本是废的。这条路只用来压内存看放置，不要拿它评估质量。",
            args.resident_frac, n)

    if not args.static and raw_profiles is not None:
        # 前段是 task 无关的，装**并集** ∪_u S_{u,l}（I.1.1）——
        # 静态模式装全集是因为没有画像；有了画像就该用并集，省下的内存直接
        # 换成更大的 L₀ 或更多通道
        front = union_placement(list(back_plcs.values()), name="front-union")
        log.info("      前段改装并集：层均 %.0f/%d（全集是 %d）",
                 sum(front.sizes()) / info.n_layers, info.n_experts, info.n_experts)

    return ModelSetup(
        label=d.name, spec=spec, n_layers=info.n_layers, n_experts=info.n_experts,
        backend="torch", node_model=hf, front_plc=front, back_plcs=back_plcs,
        approximate=approximate, profiles=raw_profiles,
        tasks=[TaskProfile(name=u, lam=l,
                           experts_per_layer=back_plcs[u].as_experts_per_layer(),
                           placement=back_plcs[u]) for u, l in tasks_lam],
    )


# --------------------------------------------------------------------------- #
def collect_capabilities(addrs: dict[str, Addr], *, mem_cap_mb: float | None,
                         real_units: bool = False) -> list[Node]:
    """步骤 1：问每个 agent「你有多少内存、算力多快」。

    算力用 agent 自己跑的 matmul 基准，归一化成相对值。比填铭牌值好，因为它
    包含了当时的实际负载与降频 —— 规划器要的就是「现在真能跑多快」。
    """
    caps: dict[str, dict] = {}
    for name, addr in addrs.items():
        try:
            r = _rpc(name, addr, {"type": "capabilities"}, timeout=15.0)
        except Exception as e:
            log.error("agent %s@%s:%d 无法连接：%s —— 已从池中剔除", name, *addr, e)
            continue
        if r.get("configured"):
            log.warning("agent %s 已被配置过，将被重新下发", name)
        caps[name] = r
        log.info("  %-8s 内存 %8.0fMB  基准 %.3fms", name, r["mem_mb"], r["ms_per_layer"])

    if not caps:
        raise SystemExit("没有任何 agent 可用")

    fastest = min(c["ms_per_layer"] for c in caps.values())
    nodes: list[Node] = []
    for name, c in caps.items():
        mb = c["mem_mb"] if mem_cap_mb is None else min(c["mem_mb"], mem_cap_mb)
        # 单位：toy 模型整个都以 MB 记账（toy_model_spec 里 base/expert 都是 MB），
        # 所以那条路把 agent 报的 MB **直接当 GB 用** —— 量纲自洽，数值好读。
        # 真模型的内存公式是 GB，必须真的换算，否则会把 47GB 当成 47000GB。
        mem = mb / 1024.0 if real_units else mb
        rel = c["ms_per_layer"] / fastest
        step = 1.0 if real_units else 8.0
        nodes.append(Node(
            id=name,
            tier=f"{round(mem / step) * step:.0f}{'GB' if real_units else 'MB'}",
            mem_gb=mem,
            ms_per_layer=round(0.35 * rel, 4),
            # 真机上留一档给激活/碎片/系统；toy 那边 5% 就够
            reserve_gb=(max(1.0, mem * 0.05) if real_units else mem * 0.05),
            avail=0.95,
        ))
    return nodes


def distribute(
    manifest, addrs: dict[str, Addr], setup: "ModelSetup",
    clf: HistogramClassifier | None,
    coord_addr: Addr, l0: int,
    static_wiring: dict[str, tuple[str, str]] | None = None,
    stop_ids: list[int] | None = None,
    model_dir: str | None = None,
    device: str = "cpu",
    profile: bool = False,
    miss_policy: str = "drop",
) -> list[dict]:
    """步骤 4：把清单拆开，每个节点只收自己那份。

    节点拿到的是「层区间 + 每层的专家 id 列表」——它不知道组合矩阵、不知道配额、
    不知道公共带。那些是离线规划的产物，在线用不到。
    """
    by_seg: dict[str, list] = {}
    for p in manifest.nodes:
        by_seg.setdefault(p.segment, []).append(p)
    for sid in by_seg:
        by_seg[sid].sort(key=lambda x: x.position)

    peers = {n: list(a) for n, a in addrs.items()}
    wiring = dict(static_wiring or {})
    acks: list[dict] = []
    for p in manifest.nodes:
        chain = by_seg[p.segment]
        i = p.position
        role = "front" if p.role.startswith("front") else p.role
        # 静态模式：前段 tail 在配置期就拿到自己那一个后段 head 的名字。
        # 之后它既不识别也不问协调器 —— 链路是随配置一起下发的。
        peer = task = None
        if wiring:
            if role == "front" and p.is_tail and p.segment in wiring:
                peer = manifest.segments[wiring[p.segment][0]]["head"]
                task = wiring[p.segment][1]
            elif role.startswith("back:"):
                task = role.split(":", 1)[1]
        cfg = NodeConfig(
            node_id=p.node,
            role=role,
            segment=p.segment,
            layer_experts={l.layer: list(l.experts) for l in p.layers},
            next_hop=chain[i + 1].node if i + 1 < len(chain) else None,
            seg_head=chain[0].node,
            is_head=p.is_head,
            is_tail=p.is_tail,
            peers=peers,
            # 真机上不注入延迟 —— 网络自己会给
            links=LinkTable().to_dict(),
            coordinator=list(coord_addr),
            model=dict(setup.node_model),
            backend=setup.backend,
            # 原样下发（可能含 {node} 占位）—— 由节点自己代入，见 node.resolve_dir
            model_dir=model_dir,
            device=device,
            with_embed=(role == "front" and p.is_head),
            with_lm_head=(role.startswith("back:") and p.is_tail),
            classifier=(clf.to_wire()
                        if (clf is not None and not wiring
                            and role == "front" and p.is_tail) else None),
            static_peer=peer,
            static_task=task,
            # 只有后段的 tail 会采样，也只有它需要知道 EOS —— 一个整数列表，
            # 节点仍然不需要 tokenizer（runtime/text.py 开头）
            stop_ids=list(stop_ids or []) if (role.startswith("back:") and p.is_tail)
            else [],
            # 只有后段采画像：前段是 task 无关的，它装全集/并集，不按 task 裁
            profile=profile and role.startswith("back:"),
            miss_policy=miss_policy,
        )
        ack = _rpc(p.node, addrs[p.node], {"type": "configure", "config": cfg.to_dict()},
                   timeout=120.0)
        acks.append(ack)
        rng = ack["layers"]
        span = f"{rng[0]}–{rng[-1]}" if len(rng) > 1 else str(rng[0])
        log.info(
            "  %-8s %-12s 层 %-7s %4d 个专家  驻留 %7.1fMB（全装 %7.1fMB）  加载 %.0fms%s",
            ack["node"], ack["role"] + "/" + ack["segment"], span,
            ack["n_experts"], ack["resident_mb"], ack["full_mb"], ack["load_ms"],
            (f"  从 checkpoint 读了 {ack['load_fraction']:.1%}"
             f"（{ack['shards']} 个分片）" if "load_fraction" in ack else ""),
        )
    return acks


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2pmoe-control", description="P2P MoE 控制器")
    ap.add_argument("--agents", required=True,
                    help="v1=host:port,v2=host:port,… 节点 id 与地址")
    ap.add_argument("--advertise", default=None,
                    help="控制器对节点可见的 IP。跨机部署必填 —— 节点要靠它回连上报")
    ap.add_argument("--bind", default="0.0.0.0", help="协调器监听网卡")
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同时打入多少条请求。超过池子容量的会排队（有界等待），"
                         "前后段一完成就自动接走队首 —— 默认 1 是串行，便于看清时序")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--coverage", type=float, default=0.95)
    ap.add_argument("--k-probe", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--j-cap", type=float, default=30.0)
    ap.add_argument("--mem-cap-mb", type=float, default=None,
                    help="人为压低每台节点的可用内存，便于在大机器上演示分档效果")
    ap.add_argument("--asymmetric-probe", action="store_true",
                    help="逐向分别探测（默认对称，见 probe.py 的口径说明）")
    ap.add_argument("--save-plan", type=Path, default=None,
                    help="把这次的部署清单存下来 —— 配合 --load-plan 可以重放")
    ap.add_argument("--load-plan", type=Path, default=None,
                    help="载入存下来的清单，**跳过探测与规划**。放置因此完全固定，"
                         "人工指定的 --wiring 才有稳定的所指")
    ap.add_argument("--model-dir", default=None,
                    help="真实 HF checkpoint 目录（**每台节点上的本地路径**，"
                         "控制机也要能读到 config.json / tokenizer.json）。"
                         "给了就跑真模型 + 文本进出；不给是 toy 模型 + token id")
    ap.add_argument("--ctx", type=int, default=2048, help="KV 预算的上下文上限")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="权重字节数（bf16=2）")
    ap.add_argument("--device", default="cpu", help="节点上的 torch 设备，如 cuda:0")
    ap.add_argument("--tasks", default=None,
                    help="task 名与权重，如 'general' 或 'code=0.6,chat=0.4'。"
                         "后段全装专家时各 task 其实无差别，用单个 task 就行")
    ap.add_argument("--profile", default=None,
                    help="离线激活画像 JSON：{task: [[专家id...] × 层数]}。"
                         "只驻留子集**必须**有它，否则驻留集是瞎选的")
    ap.add_argument("--resident-frac", type=float, default=1.0,
                    help="后段每层驻留比例。1.0（默认）= 全装、无近似、"
                         "输出与单机逐位一致；要压内存请配 --profile 一起用")
    ap.add_argument("--wiring", default=None,
                    help="人工指定前后段连接的 JSON。省略则自动配对（贪心最小化组合延迟）")
    ap.add_argument("--save-wiring", type=Path, default=None,
                    help="把这次用的连接导出来 —— 改完可以用 --wiring 喂回去")
    ap.add_argument("--miss-policy", default="drop",
                    choices=("drop", "drop_noscale", "local_topk"),
                    help="路由到的专家不在本地时怎么补救。drop=文档 II.5 的重归一；"
                         "drop_noscale=不重归一（实测更好）；"
                         "local_topk=在驻留集里重新取 top-k")
    ap.add_argument("--relay", default=None, metavar="HOST:PORT",
                    help="节点之间没有直连时的中继（deploy/relay.py）。"
                         "各节点的 agent 也要加同一个 --relay")
    ap.add_argument("--skip-model-check", action="store_true",
                    help="跳过「每台节点能不能读到 checkpoint」的预检")
    ap.add_argument("--prompt", action="append", default=None,
                    help="文本 prompt，可重复。需要 --model-dir")
    ap.add_argument("--prompts-file", type=Path, default=None,
                    help="一行一条 prompt 的文件。想标注真实 task 就写成 "
                         "`mbpp\t写一个反转链表的函数` —— 制表符前是 task 名，"
                         "用来核对在线识别对不对。`#` 开头与空行跳过。"
                         "**--requests 比条数多时会循环**，少时只跑前几条")
    ap.add_argument("--save-results", type=Path, default=None,
                    help="把逐请求结果写成 JSON：时序拆解、各节点算力占比、"
                         "识别对错、生成的文本。测完就有一份可复算的底稿，"
                         "不用回头翻滚屏日志")
    ap.add_argument("--chat", action="store_true", help="套对话模板（指令模型必须）")
    ap.add_argument("--static", action="store_true",
                    help="静态简化模式：前段装全部专家，前后段配对离线定死。"
                         "在线不识别、不派发、不换绑，请求自报 task。"
                         "见 examples/static_qwen.py 的取舍说明")
    ap.add_argument("--once", action="store_true",
                    help="跑完这批请求就退出（不常驻）。节点 agent 不受影响")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    global RELAY
    if args.relay:
        rh, _, rp = args.relay.rpartition(":")
        RELAY = (rh or "127.0.0.1", int(rp))
        log.info("中继模式：%s:%d —— 节点之间不直连，每跳绕一圈，逐 token 延迟大致翻倍",
                 *RELAY)
    addrs = parse_agents(args.agents)

    # ---------------- 0. 模型与放置 -------------------------------------- #
    tasks_lam = parse_tasks(args.tasks) if args.tasks else TASKS
    log.info("[0/5] 模型与驻留专家集（task %s）",
             {u: round(l, 2) for u, l in tasks_lam})
    setup = build_model_setup(args, tasks_lam)
    mcfg, spec, front_plc = setup.mcfg, setup.spec, setup.front_plc
    plcs, corpus = setup.back_plcs, setup.corpus
    profiles_raw = setup.profiles

    # ---------------- 1. 采集能力 ---------------------------------------- #
    log.info("[1/5] 采集节点能力（%d 台）", len(addrs))
    nodes = collect_capabilities(addrs, mem_cap_mb=args.mem_cap_mb,
                                 real_units=setup.backend == "torch")
    if setup.backend == "torch":
        log.info("      可用内存/台 %s（已扣预留）",
                 sorted({f"{n.usable_gb:.1f}GB" for n in nodes}))
    live = {n.id for n in nodes}

    # toy 模型只有几十 MB。真实节点报几十 GB 的话，一台就装得下整条通道，
    # 规划会退化成「全放一台、零跳」—— 部署路径照样验证得了，但看不到分段、
    # 跳数、公共带这些真正的机制。明确提示，而不是悄悄替用户做决定。
    front_mb = (spec.base_gb_per_layer
                + spec.expert_gb * sum(front_plc.sizes()) / setup.n_layers)
    chan_mb = front_mb * setup.n_layers   # 粗估：整模型驻留量的量级
    smallest = min(n.usable_gb for n in nodes)
    if setup.backend == "numpy" and args.mem_cap_mb is None and smallest > 20 * chan_mb:
        log.warning(
            "      节点最小可用内存 %.0fMB，而整个 toy 模型才 ~%.0fMB —— "
            "一台就装得下，规划会退化成「全放一台、零跳」。",
            smallest, chan_mb,
        )
        log.warning(
            "      想看到真正的分段/跳数/公共带，加 --mem-cap-mb %.0f 左右；"
            "这是 toy 模型的产物，接真实 MoE 后不需要。", max(26.0, chan_mb * 0.9),
        )
    addrs = {k: v for k, v in addrs.items() if k in live}

    if args.model_dir and not args.skip_model_check:
        log.info("      预检：%d 台节点能不能读到 %s", len(addrs), args.model_dir)
        bad = check_model_dirs(addrs, args.model_dir)
        if bad:
            log.error("以下节点读不到 checkpoint：")
            for b in bad:
                log.error("    %s", b)
            log.error("权重分发还没做（TODO.md P0）—— --model-dir 是**各节点上的"
                      "本地路径**。先把 checkpoint 同步到每台机器（rsync / "
                      "huggingface-cli download / 共享 NFS 挂载），或加 "
                      "--skip-model-check 跳过预检自担风险")
            return 4
        log.info("      全部就绪")

    # ---------------- 2. 探测 + 3. 规划 ---------------------------------- #
    cfg = PlannerConfig(eta=args.eta, beta=1.3, j_cap_ms=args.j_cap, theta=0.8,
                        kappa_over=0.3, n_standby=0, k_probe=args.k_probe,
                        k_gate=args.k_probe, k_audit=args.k_probe, seed=args.seed)
    net = None
    res = None

    if args.load_plan:
        # 载入清单 = 把放置固定住。规划的输入里有一项不可复现：逐对延迟实测。
        # 同一个池子换个时间跑，探测值会变，段的构成与 id 编号都可能跟着变 ——
        # 于是「F0 连 BX1」这种人工指定在下一次规划里可能指到别的东西上。
        # 存清单 → 改连接 → 载清单，这条路让指定有稳定的所指。
        man = DeploymentManifest.from_json(args.load_plan.read_text(encoding="utf-8"))
        log.info("[2/5] 跳过探测（--load-plan）")
        log.info("[3/5] 载入清单 %s：L₀=%d，%d 个节点，%d 条段",
                 args.load_plan, man.l0, len(man.nodes), len(man.segments))
        missing = {p.node for p in man.nodes} - set(addrs)
        if missing:
            log.error("清单里的节点 %s 不在 --agents 里。清单绑定的是**当时**那批"
                      "机器名；机器换了就得重跑规划（去掉 --load-plan）",
                      sorted(missing))
            return 3
        n_ck = sum(len(p.layers) for p in man.nodes
                   if p.segment == next(iter(man.segments)))
        if n_ck and man.l0 >= setup.n_layers:
            log.error("清单的 L₀=%d 与当前模型的 %d 层对不上 —— 清单不是这个模型的",
                      man.l0, setup.n_layers)
            return 3
        l0 = man.l0
    else:
        log.info("[2/5] 逐对探测（由节点自己发起，k=%d）", args.k_probe)
        t0 = time.perf_counter()
        oracle = RemoteNetworkOracle(addrs, k_default=args.k_probe,
                                     symmetric=not args.asymmetric_probe,
                                     relay=RELAY)
        n_pairs = oracle.warm_all(sorted(live), k=args.k_probe)
        log.info("      %d 对，用时 %.1fs，%d 次 RPC", n_pairs,
                 time.perf_counter() - t0, oracle.n_rpc)
        reach = oracle.reachability(sorted(live))
        dead = [n for n, c in reach.items() if c < len(live) - 1]
        if dead:
            log.warning("      可达性不完整: %s —— 分散环境常态（NAT/防火墙），"
                        "规划会自然绕开不可达链路",
                        {n: f"{reach[n]}/{len(live)-1}" for n in dead})
        for f in oracle.failures[:5]:
            log.debug("      %s", f)

        log.info("[3/5] 离线规划")
        net = MeasurementCache(oracle, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms,
                               k_gate=cfg.k_gate)
        p_curve = {l: min(0.97, 0.70 + 0.05 * l) for l in range(1, setup.n_layers)}
        try:
            res = plan(nodes, spec, setup.tasks, front_plc, net, cfg, p_curve,
                       p_min=0.80)
        except PlanningError as e:
            for line in e.log:
                log.info("      %s", line)
            log.error("规划失败: %s", str(e).split("。")[0])
            return 2

        man = res.manifest
        l0 = res.l0
        log.info("      L₀=%d，配额 %s，前段 %d 条，组合矩阵 %d 组，清单校验 %s",
                 res.l0, {u: len(v) for u, v in res.backs.items()},
                 len(res.fronts_final), len(man.pairings),
                 "通过" if man.ok else f"未通过 {man.violations[:1]}")
        if not man.ok:
            return 3
        if args.save_plan:
            args.save_plan.write_text(man.to_json(), encoding="utf-8")
            log.info("      清单已存 %s（--load-plan 可以原样重放）", args.save_plan)

    # ---------------- 3b. 静态模式：把链路定死 --------------------------- #
    wired: dict[str, tuple[str, str]] = {}
    if args.static:
        front_ids = sorted(k for k, v in man.segments.items() if v["role"] == "front")
        if args.wiring:
            wired = resolve_wiring(Path(args.wiring), man, front_ids)
            log.info("      静态链路：按 %s 指定的 %d 条", args.wiring, len(wired))
        elif res is not None:
            wiring = assign_static_pairs(res.fronts_final, res.backs, net,
                                         k=cfg.k_audit,
                                         front_ids=[f"F{i}" for i in
                                                    range(len(res.fronts_final))])
            if not wiring.pairs:
                log.error("静态配对没配出任何通道 —— 前段或后段为空")
                return 3
            wired = wiring.as_map()
            log.info("      静态链路（自动配对）：%s", wiring.summary())
            if wiring.unpaired_backs:
                log.warning("        ⚠ 没配上前段的后段: %s —— 这些后段收不到请求",
                            wiring.unpaired_backs)
        else:
            # --load-plan 但没给 --wiring：用清单里存着的组合矩阵重放一次贪心。
            # 拿的是当时测的 t50，不是现在的 —— 网络变了这个选择就未必还最优。
            log.info("      静态链路：按清单里存的组合矩阵贪心配对（延迟是**当时**测的）")
            used_f: set[str] = set()
            used_b: set[str] = set()
            for pr in sorted(man.pairings, key=lambda x: x.t50):
                if pr.front in used_f or pr.back in used_b:
                    continue
                wired[pr.front] = (pr.back, pr.task)
                used_f.add(pr.front)
                used_b.add(pr.back)
            if not wired:
                log.error("清单里没有组合矩阵，无法自动配对 —— 请给 --wiring")
                return 3

        # 无论哪条路，都把这几对的延迟重报一遍
        t50 = {(q.front, q.back): q for q in man.pairings}
        for i, (f, (b, u)) in enumerate(sorted(wired.items())):
            sf, sb = man.segments[f], man.segments[b]
            if net is not None:
                w = net.get(sf["tail"], sb["head"], cfg.k_audit).p50
                dl = net.get(sb["tail"], sf["head"], cfg.k_audit).p50
                tot, mark = sf["delay_ms"] + w + sb["delay_ms"] + dl, ""
            elif (f, b) in t50:
                q = t50[(f, b)]
                w, dl, tot, mark = q.w_p50, q.d_loop_p50, q.t50, "（清单值）"
            else:
                w = dl = tot = float("nan")
                mark = "（清单里没测过这一对 —— 它不在当时的组合矩阵里）"
            log.info("        ch%d  %-8s %s%s → %s%s  正向 %.0fms  回环 %.0fms  组合 %.0fms%s",
                     i, u, f, tuple(sf["nodes"]), b, tuple(sb["nodes"]), w, dl, tot, mark)
        if args.save_wiring:
            args.save_wiring.write_text(wiring_to_json(wired, man), encoding="utf-8")
            log.info("      连接已存 %s（改完可用 --wiring 喂回来）", args.save_wiring)

    # ---------------- 4. 下发清单 ---------------------------------------- #
    host = args.advertise
    if host is None:
        host = "127.0.0.1" if all(a[0] in ("127.0.0.1", "localhost") for a in addrs.values()) \
            else _guess_local_ip(next(iter(addrs.values())))
        log.info("      未指定 --advertise，推断控制器地址为 %s", host)

    clf = None
    if not args.static:
        if setup.backend == "numpy":
            # toy 模型：拿同一份合成语料实测参考直方图
            refs = measure_front_refs(mcfg, corpus, front_plc, l0)
            clf = HistogramClassifier(refs, dict(tasks_lam), tau_hi=0.55, tau_lo=0.40)
        elif profiles_raw is not None:
            # 真模型：参考直方图 = 各 task 在前段层上的激活质量之和。
            # **这是动态模式在真模型上唯一的分类器来源** —— 没有画像就没有识别，
            # 而没有识别就只能走 --static（配对定死、请求自报 task）。
            clf = HistogramClassifier.from_profiles(profiles_raw, l0,
                                                    dict(tasks_lam),
                                                    tau_hi=0.55, tau_lo=0.40)
            log.info("      分类器：从 %s 的前 %d 层激活质量构造",
                     args.profile, l0)
        else:
            log.error("动态模式在真模型上需要 --profile 来构造分类器 —— "
                      "没有它就没有识别。要么给 --profile，要么用 --static")
            return 2
    if setup.backend == "numpy":
        # toy 模型能**实测**基线：拿同一份语料、同一条轨迹跑一遍就知道了
        baselines = measure_baseline_miss(mcfg, corpus, front_plc, plcs, l0)
        how = "实测"
    else:
        # 真模型没有可回放的语料，只能用「1 − 覆盖率」这个估计值。
        # 注意它**偏低 3–6 倍**（README「与文档的偏差」第八条）—— 动态模式下
        # 拿它当告警线会对绑对的池持续误报。静态模式不触发换绑，所以只是个读数。
        baselines = {u: p_.baseline_miss(l0 + 1, setup.n_layers)
                     for u, p_ in plcs.items()}
        how = "按 1−覆盖率 估计，偏低，仅供参考"
    log.info("      通道二基线（%s）: %s%s", how,
             {u: f"{v:.1%}" for u, v in baselines.items()},
             "（静态模式下只统计，不触发换绑）" if args.static else "")

    # ---- 文本层：只在控制机上 ---- #
    textio = None
    if args.model_dir:
        try:
            textio = TextIO.from_model_dir(args.model_dir, chat=args.chat)
            log.info("      tokenizer: vocab %d，停止 token %s，%s",
                     textio.tok.vocab_size, sorted(textio.stop.ids),
                     "套对话模板" if args.chat else "completion 模式")
        except (FileNotFoundError, ImportError) as e:
            log.warning("      没有文本层（%s）→ 请求仍走 token id", e)
    elif args.prompt:
        log.warning("      给了 --prompt 但没有 --model-dir，tokenizer 无从加载 —— 忽略")

    coord = Coordinator(man, baselines=baselines, priors=dict(tasks_lam),
                        alarm_factor=3.0, host=args.bind, static_wiring=wired or None,
                        textio=textio, relay=RELAY)
    coord_addr = (host, coord.port)
    log.info("[4/5] 下发清单（协调器 %s:%d）", *coord_addr)
    distribute(man, addrs, setup, clf, coord_addr, l0, wired or None,
               stop_ids=sorted(textio.stop.ids) if textio else None,
               model_dir=args.model_dir, device=args.device,
               miss_policy=args.miss_policy)

    pool = PeerPool("__coord__", LinkTable(), seed=0)
    pool.use_relay(RELAY)
    for n, a in addrs.items():
        pool.register(n, a)
    coord.start(pool)
    coord.max_tokens = args.tokens
    pool.warm(addrs)   # 保温网格（II.4 Step 6）

    # ---------------- 5. 在线服务 ---------------------------------------- #
    if args.static:
        names = sorted({t for _, t in wired.values()})
        log.info("[5/5] 在线服务（静态）：%d 条请求 × %d token，并发 %d，"
                 "通道 %s —— task 由请求给定，不识别",
                 args.requests, args.tokens, args.concurrency,
                 {u: len(q) for u, q in coord.front_pools.items()})
    else:
        names = [u for u, _ in tasks_lam]
        log.info("[5/5] 在线服务：%d 条请求 × %d token，并发 %d（池子 %d 前段 / %s 后段）",
                 args.requests, args.tokens, args.concurrency,
                 len(coord.free_fronts), {u: len(q) for u, q in coord.free_backs.items()})
    # 来自 --prompts-file 的测试集：[(true_task|None, 文本), ...]
    cases: list[tuple[str | None, str]] = []
    if args.prompts_file:
        cases = load_cases(args.prompts_file, names)
        if not cases:
            log.error("%s 里一条 prompt 都没有", args.prompts_file)
            return 2
        blank = [i for i, (_, t) in enumerate(cases) if not t.strip()]
        if blank:
            log.warning("  第 %s 行的 prompt 是空的 —— 采集端的问题。"
                        "它们会照样发出去（0 token 的请求），"
                        "好让问题露出来而不是被悄悄跳过",
                        ", ".join(str(i + 1) for i in blank[:10]))
        labelled = sum(1 for u, _ in cases if u)
        log.info("测试集 %s：%d 条，其中 %d 条带 task 标注%s",
                 args.prompts_file, len(cases), labelled,
                 "" if labelled == len(cases) else "（没标注的按轮转指派真实 task）")
        if args.requests > len(cases):
            log.info("  --requests %d > %d 条 —— 会循环重跑",
                     args.requests, len(cases))
        elif args.requests < len(cases):
            log.info("  --requests %d < %d 条 —— 只跑前 %d 条",
                     args.requests, len(cases), args.requests)

    rows, batch = [], []

    def drain(batch_):
        for rec in batch_:
            if not rec.done.wait(timeout=300):
                log.error("%s 超时。已发生的事件：", rec.req)
                for e in rec.events:
                    log.error("    %s", e)
                continue
            per = sorted(rec.token_ms)
            p50 = per[len(per) // 2] if per else 0.0
            rows.append((rec, p50))
            if rec.text:
                log.info("  %-7s %s×%s  停于 %s  %d token  首token %.0fms  «%s»",
                         rec.req, rec.front, rec.back, rec.stop_reason,
                         len(rec.tokens), (rec.t_first - rec.t0) * 1000,
                         rec.text[:80].replace("\n", "⏎"))
            elif args.static:
                log.info("  %-7s task %s  %s×%s（定死）  排队 %.0fms  "
                         "首token %.0fms  逐token p50 %.0fms",
                         rec.req, rec.task, rec.front, rec.back,
                         rec.wait_front_ms, (rec.t_first - rec.t0) * 1000, p50)
            else:
                log.info("  %-7s 真实 %s → 识别 %s %s  %s×%s  换绑 %d  "
                         "排队 %.0f/%.0fms  首token %.0fms  逐token p50 %.0fms",
                         rec.req, rec.true_task, rec.task, "✓" if rec.correct else "✗",
                         rec.front, rec.back, rec.rebinds,
                         rec.wait_front_ms, rec.wait_back_ms,
                         (rec.t_first - rec.t0) * 1000, p50)

    for i in range(args.requests):
        u = names[i % len(names)]
        if textio and cases:
            true_u, txt = cases[i % len(cases)]
            true_u = true_u or u
            batch.append(coord.submit(f"req{i}", text=txt, true_task=true_u,
                                      task=true_u if args.static else None))
        elif textio and args.prompt:
            batch.append(coord.submit(f"req{i}", text=args.prompt[i % len(args.prompt)],
                                      true_task=u, task=u if args.static else None))
        else:
            batch.append(coord.submit(f"req{i}", sample_prompt(corpus, u, 12, seed=1000 + i),
                                      true_task=u, task=u if args.static else None))
        if len(batch) >= args.concurrency:
            if args.concurrency > 1:
                log.info("  已打入 %d 条；池深 %s", len(batch), coord.queue_depths())
            drain(batch)
            batch = []
    if batch:
        drain(batch)

    if rows:
        p50 = float(np.median([m for _, m in rows]))
        if args.static:
            log.info("汇总：%d 条完成，逐 token p50 %.0fms，换绑 %d 次（静态模式恒为 0）",
                     len(rows), p50, sum(r.rebinds for r, _ in rows))
        else:
            log.info("汇总：识别 %d/%d，逐 token p50 %.0fms，换绑 %d 次",
                     sum(r.correct for r, _ in rows), len(rows), p50,
                     sum(r.rebinds for r, _ in rows))
        log.info("配对历史（每条请求用了哪对段）：")
        for req, f, b, u in coord.pairings:
            log.info("    %-7s %s × %s  (%s)", req, f, b, u)
    if args.save_results and rows:
        from p2pmoe.runtime.timing import summarise_request

        out = {
            "model": setup.label,
            "l0": int(getattr(spec, "l0", 0) or 0),
            "backend": setup.backend,
            "mode": "static" if args.static else "dynamic",
            "miss_policy": args.miss_policy,
            "coverage": args.coverage,
            "tokens": args.tokens,
            "concurrency": args.concurrency,
            "requests": [],
        }
        for rec, p50 in rows:
            t = summarise_request(rec, coord)
            out["requests"].append({
                "req": rec.req,
                "prompt": rec.prompt,
                "text": rec.text,
                "stop_reason": rec.stop_reason,
                "true_task": rec.true_task,
                "task": rec.task,
                # 静态模式不识别，correct 恒真没有信息量 —— 置 None 免得被当成 100%
                "correct": None if args.static else bool(rec.correct),
                "confidence": round(rec.conf, 4),
                "zone": rec.zone,
                "rebinds": rec.rebinds,
                "front": rec.front, "back": rec.back,
                "n_prompt": t.n_prompt, "n_generated": t.n_generated,
                "total_ms": round(t.total_ms, 2),
                "queue_ms": round(t.queue_ms, 2),
                "prefill_ms": round(t.prefill_ms, 2),
                "decode_ms": round(t.decode_ms, 2),
                "per_token_p50_ms": round(p50, 2),
                "compute_ms": round(t.compute_ms, 2),
                # 反推的，不是测量值：15 台没有时钟同步，跨机时刻拼不到一条轴上
                "network_and_overhead_ms": round(t.other_ms, 2),
                "utilisation": round(t.utilisation, 4),
                "token_ms": [round(x, 2) for x in t.token_ms],
                "nodes": [{
                    "node": n.node, "role": n.role, "segment": n.segment,
                    "layers": n.layer_span, "n_experts": n.n_experts,
                    "compute_ms": round(n.compute_ms, 2),
                    "n_forward": n.n_forward, "bytes_out": n.bytes_out,
                    "share": round(t.node_utilisation(n), 4),
                } for n in t.nodes],
                "missing_traces": t.missing_traces,
            })
        rs = out["requests"]
        out["summary"] = {
            "n": len(rs),
            "total_ms_p50": round(float(np.median([r["total_ms"] for r in rs])), 2),
            "per_token_ms_p50": round(float(np.median(
                [r["per_token_p50_ms"] for r in rs])), 2),
            "utilisation_mean": round(float(np.mean(
                [r["utilisation"] for r in rs])), 4),
            "rebinds": sum(r["rebinds"] for r in rs),
            "accuracy": (None if args.static else round(
                sum(bool(r["correct"]) for r in rs) / len(rs), 4)),
            "errors": len(coord.errors),
        }
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        args.save_results.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("结果已写入 %s（%d 条）—— 算力使用率均值 %.1f%%，总时延 p50 %.0fms",
                 args.save_results, len(rs),
                 out["summary"]["utilisation_mean"] * 100,
                 out["summary"]["total_ms_p50"])

    if coord.errors:
        log.error("节点上报的错误：")
        for e in coord.errors[:5]:
            log.error("  %s", e.replace("\n", " ")[:300])

    if not args.once:
        log.info("保持运行中，Ctrl-C 停止（节点 agent 不会被关闭）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    coord.stop()
    pool.close()
    return 1 if coord.errors else 0


def _guess_local_ip(peer: Addr) -> str:
    """用一个到对端的 UDP socket 反查本机在该路由上的出口 IP（不发包）。"""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(peer)
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
