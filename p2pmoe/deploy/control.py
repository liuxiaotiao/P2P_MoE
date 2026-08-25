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
from pathlib import Path

import numpy as np

from ..planner.experts import build_placement, union_placement
from ..planner.network import MeasurementCache
from ..planner.pipeline import PlanningError, plan
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


def toy_model_spec(cfg: ToyMoEConfig) -> ModelSpec:
    """toy 模型的真实字节数 → 规划器的内存模型（单位统一取 MB）。"""
    mb = 1e6
    return ModelSpec(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_experts=cfg.n_experts,
        top_k=cfg.top_k, base_gb_per_layer=cfg.base_params * 8 / mb,
        expert_gb=cfg.expert_params * 8 / mb, ctx_max=CTX_MAX, kv_bytes_per_elem=8,
    )


# --------------------------------------------------------------------------- #
def collect_capabilities(addrs: dict[str, Addr], *, mem_cap_mb: float | None) -> list[Node]:
    """步骤 1：问每个 agent「你有多少内存、算力多快」。

    算力用 agent 自己跑的 matmul 基准，归一化成相对值。比填铭牌值好，因为它
    包含了当时的实际负载与降频 —— 规划器要的就是「现在真能跑多快」。
    """
    caps: dict[str, dict] = {}
    for name, addr in addrs.items():
        try:
            r = rpc(addr, {"type": "capabilities"}, timeout=15.0)
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
        mem = c["mem_mb"] if mem_cap_mb is None else min(c["mem_mb"], mem_cap_mb)
        rel = c["ms_per_layer"] / fastest
        nodes.append(Node(
            id=name,
            tier=f"{round(mem / 8) * 8:.0f}MB",   # 按内存粗分档（II.7.1 的分档估算要用）
            mem_gb=mem,
            ms_per_layer=round(0.35 * rel, 4),
            reserve_gb=max(1.0, mem * 0.05),
            avail=0.95,
        ))
    return nodes


def distribute(
    manifest, addrs: dict[str, Addr], mcfg: ToyMoEConfig, clf: HistogramClassifier,
    coord_addr: Addr, l0: int,
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
    acks: list[dict] = []
    for p in manifest.nodes:
        chain = by_seg[p.segment]
        i = p.position
        role = "front" if p.role.startswith("front") else p.role
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
            model=dict(mcfg.__dict__),
            classifier=clf.to_wire() if (role == "front" and p.is_tail) else None,
        )
        ack = rpc(addrs[p.node], {"type": "configure", "config": cfg.to_dict()}, timeout=120.0)
        acks.append(ack)
        log.info(
            "  %-8s %-12s 层 %s  %d 个专家  驻留 %.2fMB（全装要 %.2fMB）  加载 %.0fms",
            ack["node"], ack["role"] + "/" + ack["segment"], ack["layers"],
            ack["n_experts"], ack["resident_mb"], ack["full_mb"], ack["load_ms"],
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
    ap.add_argument("--save-plan", type=Path, default=None)
    ap.add_argument("--once", action="store_true",
                    help="跑完这批请求就退出（不常驻）。节点 agent 不受影响")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    addrs = parse_agents(args.agents)

    # ---------------- 0. 离线画像（回放语料） ---------------------------- #
    mcfg = ToyMoEConfig()
    spec = toy_model_spec(mcfg)
    log.info("[0/5] 回放语料 → 激活画像 → 驻留专家集")
    corpus = make_corpus(mcfg, [t for t, _ in TASKS], seed=args.seed, shared_clusters=0)
    profiles = profile_from_corpus(mcfg, corpus)
    plcs = {u: build_placement(p, args.coverage) for u, p in profiles.items()}
    uni = union_placement(list(plcs.values()))
    log.info("      驻留集层均 %s，并集 %.1f/%d",
             {u: round(sum(p.sizes()) / mcfg.n_layers, 1) for u, p in plcs.items()},
             sum(uni.sizes()) / mcfg.n_layers, mcfg.n_experts)

    # ---------------- 1. 采集能力 ---------------------------------------- #
    log.info("[1/5] 采集节点能力（%d 台）", len(addrs))
    nodes = collect_capabilities(addrs, mem_cap_mb=args.mem_cap_mb)
    live = {n.id for n in nodes}

    # toy 模型只有几十 MB。真实节点报几十 GB 的话，一台就装得下整条通道，
    # 规划会退化成「全放一台、零跳」—— 部署路径照样验证得了，但看不到分段、
    # 跳数、公共带这些真正的机制。明确提示，而不是悄悄替用户做决定。
    front_mb = (spec.base_gb_per_layer + spec.expert_gb * sum(uni.sizes()) / mcfg.n_layers)
    chan_mb = front_mb * mcfg.n_layers   # 粗估：整模型驻留量的量级
    smallest = min(n.usable_gb for n in nodes)
    if args.mem_cap_mb is None and smallest > 20 * chan_mb:
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

    # ---------------- 2. 真实探测 ---------------------------------------- #
    log.info("[2/5] 逐对探测（由节点自己发起，k=%d）", args.k_probe)
    t0 = time.perf_counter()
    oracle = RemoteNetworkOracle(addrs, k_default=args.k_probe,
                                 symmetric=not args.asymmetric_probe)
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

    # ---------------- 3. 规划 -------------------------------------------- #
    log.info("[3/5] 离线规划")
    cfg = PlannerConfig(eta=args.eta, beta=1.3, j_cap_ms=args.j_cap, theta=0.8,
                        kappa_over=0.3, n_standby=0, k_probe=args.k_probe,
                        k_gate=args.k_probe, k_audit=args.k_probe, seed=args.seed)
    net = MeasurementCache(oracle, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)
    tasks = [TaskProfile(name=u, lam=l, experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u, l in TASKS]
    p_curve = {l: 0.70 + 0.05 * l for l in range(2, mcfg.n_layers)}
    try:
        res = plan(nodes, spec, tasks, uni, net, cfg, p_curve, p_min=0.80)
    except PlanningError as e:
        for line in e.log:
            log.info("      %s", line)
        log.error("规划失败: %s", str(e).split("。")[0])
        return 2

    man = res.manifest
    log.info("      L₀=%d，配额 %s，前段 %d 条，组合矩阵 %d 组，清单校验 %s",
             res.l0, {u: len(v) for u, v in res.backs.items()},
             len(res.fronts_final), len(man.pairings),
             "通过" if man.ok else f"未通过 {man.violations[:1]}")
    if not man.ok:
        return 3
    if args.save_plan:
        args.save_plan.write_text(man.to_json(), encoding="utf-8")
        log.info("      清单已存 %s", args.save_plan)

    # ---------------- 4. 下发清单 ---------------------------------------- #
    host = args.advertise
    if host is None:
        host = "127.0.0.1" if all(a[0] in ("127.0.0.1", "localhost") for a in addrs.values()) \
            else _guess_local_ip(next(iter(addrs.values())))
        log.info("      未指定 --advertise，推断控制器地址为 %s", host)

    refs = measure_front_refs(mcfg, corpus, uni, res.l0)
    clf = HistogramClassifier(refs, {u: l for u, l in TASKS}, tau_hi=0.55, tau_lo=0.40)
    baselines = measure_baseline_miss(mcfg, corpus, uni, plcs, res.l0)
    log.info("      通道二基线（实测）: %s", {u: f"{v:.1%}" for u, v in baselines.items()})

    coord = Coordinator(man, baselines=baselines, priors={u: l for u, l in TASKS},
                        alarm_factor=3.0, host=args.bind)
    coord_addr = (host, coord.port)
    log.info("[4/5] 下发清单（协调器 %s:%d）", *coord_addr)
    distribute(man, addrs, mcfg, clf, coord_addr, res.l0)

    pool = PeerPool("__coord__", LinkTable(), seed=0)
    for n, a in addrs.items():
        pool.register(n, a)
    coord.start(pool)
    coord.max_tokens = args.tokens
    pool.warm(addrs)   # 保温网格（II.4 Step 6）

    # ---------------- 5. 在线服务 ---------------------------------------- #
    log.info("[5/5] 在线服务：%d 条请求 × %d token，并发 %d（池子 %d 前段 / %s 后段）",
             args.requests, args.tokens, args.concurrency,
             len(coord.free_fronts), {u: len(q) for u, q in coord.free_backs.items()})
    names = [t for t, _ in TASKS]
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
            log.info("  %-7s 真实 %s → 识别 %s %s  %s×%s  换绑 %d  "
                     "排队 %.0f/%.0fms  首token %.0fms  逐token p50 %.0fms",
                     rec.req, rec.true_task, rec.task, "✓" if rec.correct else "✗",
                     rec.front, rec.back, rec.rebinds,
                     rec.wait_front_ms, rec.wait_back_ms,
                     (rec.t_first - rec.t0) * 1000, p50)

    for i in range(args.requests):
        u = names[i % len(names)]
        batch.append(coord.submit(f"req{i}", sample_prompt(corpus, u, 12, seed=1000 + i),
                                  true_task=u))
        if len(batch) >= args.concurrency:
            if args.concurrency > 1:
                log.info("  已打入 %d 条；池深 %s", len(batch), coord.queue_depths())
            drain(batch)
            batch = []
    if batch:
        drain(batch)

    if rows:
        ok = sum(r.correct for r, _ in rows)
        p50 = float(np.median([m for _, m in rows]))
        log.info("汇总：识别 %d/%d，逐 token p50 %.0fms，换绑 %d 次", 
                 ok, len(rows), p50, sum(r.rebinds for r, _ in rows))
        log.info("配对历史（每条请求用了哪对段）：")
        for req, f, b, u in coord.pairings:
            log.info("    %-7s %s × %s  (%s)", req, f, b, u)
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
