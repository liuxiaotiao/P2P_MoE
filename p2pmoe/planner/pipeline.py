"""II.4 离线流水线：六步编排 + 终审。

    Step 1  分档估上限 N_max
    Step 2  打折得工作值 N_work → 按 λ_u 配额 → 公平比
    Step 3  建后段 + 平衡  ⟹ 后段 cluster 冻结，事实入口集 H
    Step 4  公共中值域     ⟹ Q_F^out（滑窗精确最优）
    Step 5  建前段 + 平衡，**故意超建**
    Step 6  回环裁剪 → 备胎池 → 终审 → 固化

一句话分工：内存定上限、λ 定配额、beam 定单条、平衡定一侧、公共带定接口、
回环定最终名单。前五步攒余量（超建），最后一步花余量（裁剪）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from statistics import median
from typing import Mapping, Sequence

from .capacity import (
    CapacityEstimate,
    estimate_capacity_by_tier,
    fair_ratios,
    largest_remainder,
)
from .common_band import (
    BandResult,
    ExitProbe,
    common_band,
    diagnose_outlier_entries,
    pick_bandwidth,
    pick_outlier,
    probe_exits,
    sweep_bandwidth,
)
from .experts import ExpertPlacement
from .loop_trim import TrimResult, loop_profile, trim_by_loop
from .manifest import DeploymentManifest, build_manifest
from .memory import L0Candidate, choose_l0, hops_min, make_back_spec, make_front_spec
from .network import MeasurementCache
from .solver import deploy_path
from .tighten import band_from_median, detect_gap_robust, blame_slow_tail, tighten_lex
from .types import ModelSpec, Node, Objective, PlannerConfig, Segment, SegmentSpec, TaskProfile

__all__ = ["PlanResult", "AuditReport", "PlanningError", "plan", "select_rent_nodes"]


class PlanningError(ValueError):
    """规划失败。带上到失败为止的完整日志 —— 分散环境下失败本身就是结论
    （例如「该池群不适合本架构，应做域分裂」），日志是判据。"""

    def __init__(self, message: str, log: Sequence[str]):
        super().__init__(message)
        self.log = list(log)


# --------------------------------------------------------------------------- #
# 影子租金 R_F
# --------------------------------------------------------------------------- #
def select_rent_nodes(
    nodes: Sequence[Node],
    front_spec: SegmentSpec,
    n: int,
    *,
    policy: str = "smallest_sufficient",
) -> frozenset[str]:
    """选出前段的高潜集 R_F（II.4「影子租金」）。

    policy:
      "largest"             —— 文档 III.8.4 口径：按 (M_v, 稳定性) 取前 N 名。
      "smallest_sufficient" —— 本实现默认：在**够装下前段**的节点里取最小的 N 台。

    为什么默认改了口径：III.8.4 的论证前提是「前段候选几乎全落在排序前列」，
    这在前段驻留量最重时成立。但前段只需装 L₀ 层并集，而后段要装 (L−L₀) 层
    专用集 —— 当 L₀ 很小时，后段的单节点需求反而**大于**前段（算例 C 里前段
    28.0GB，X 后段的大位要 47GB）。此时按 (M_v) 降序预留会把最稀缺的大内存
    节点划给根本用不上那么多内存的前段，逼得后段多一跳，代价是数十毫秒/token。
    取「够用的最小者」把大节点留给真正需要的一侧，与「跳数优先」一致。
    """
    need = front_spec.total_gb()
    ok = [v for v in nodes if v.usable_gb >= need - 1e-9]
    if policy == "largest":
        ok.sort(key=lambda v: (-v.usable_gb, -v.avail))
    else:
        ok.sort(key=lambda v: (v.usable_gb, -v.avail))
    return frozenset(v.id for v in ok[:n])


def scarcity_rent_nodes(
    spec_key: str,
    forms: Mapping[str, list[tuple[str, ...]]],
    nodes: Sequence[Node],
) -> frozenset[str]:
    """稀缺档保护租金 —— 对文档「影子租金」的一个必要补充。

    II.4 的影子租金只保护**前段**的高潜集 R_F，但分散环境下逐条顺序建段还有
    第二个抢占问题：先建的段会顺手占掉某个稀缺档，而它本可以不占。算例 C 里
    task Y 的后段有 (V100-32, V100-32) 这个可行形态，却因为先建而抓走了两张
    A40 —— 其中一张只用来放 2 层。等 X 池要建第二条时 A40 已经没了。

    判据是纯组合的、不含调参：
      * 某档 T 对 spec S「不可避免」  ⟺  S 的**每一个**可行形态都含 T；
      * 某档 T 对 spec S「可避免」    ⟺  S 存在不含 T 的可行形态。
    若 T 对别的 spec 不可避免、而对本 spec 可避免，则本 spec 用 T 要付租金。
    """
    all_tiers = {t for fs in forms.values() for f in fs for t in f}
    unavoidable_elsewhere = {
        t
        for t in all_tiers
        if any(
            fs and all(t in f for f in fs)
            for key, fs in forms.items()
            if key != spec_key
        )
    }
    mine = forms.get(spec_key, [])
    avoidable_here = {t for t in all_tiers if any(t not in f for f in mine)} if mine else set()
    charged = unavoidable_elsewhere & avoidable_here
    return frozenset(n.id for n in nodes if n.tier in charged)


# --------------------------------------------------------------------------- #
# 终审
# --------------------------------------------------------------------------- #
@dataclass
class PairAudit:
    front: str
    back: str
    task: str
    t_front: float
    w_fwd: float
    t_back: float
    d_loop: float
    t50: float
    t95_upper: float


@dataclass
class AuditReport:
    pairs: list[PairAudit]
    by_task: dict[str, list[PairAudit]]
    median_t50: float
    rel_spread_by_task: dict[str, float]
    worst_rel_spread: float
    max_t95_upper: float
    max_hop_jitter: float
    unserved_tasks: list[str]
    """一条段都没建出来的 task —— 这类请求在线时无处可去，是硬失败。"""
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def ledger(self) -> list[tuple[str, str, str]]:
        """账本（对应算例 C.4 的表）。"""
        rows: list[tuple[str, str, str]] = []
        t50 = [p.t50 for p in self.pairs]
        rows.append(("组合 p50 区间", f"{min(t50):.1f}–{max(t50):.1f}ms", "终审网格实测"))
        rows.append(
            ("组合 p50 极差", f"{max(t50) - min(t50):.1f}ms", "定理 III.2.1 的三项之和为其上界")
        )
        for u, v in self.rel_spread_by_task.items():
            rows.append((f"相对均匀性 · {u}", f"{v * 100:.1f}%", "I.2.3 目标 A1″"))
        rows.append(("p95 上界（保守合成）", f"{self.max_t95_upper:.1f}ms", "I.2.3 尾闸"))
        rows.append(("最大单跳抖动", f"{self.max_hop_jitter:.1f}ms", "I.2.3 抖动闸 J_cap"))
        if self.unserved_tasks:
            rows.append(("无法服务的 task", "、".join(self.unserved_tasks), "硬失败"))
        return rows


def _audit(
    fronts: Sequence[Segment],
    backs_by_task: Mapping[str, Sequence[Segment]],
    net: MeasurementCache,
    cfg: PlannerConfig,
    all_task_names: Sequence[str] = (),
) -> AuditReport:
    """最终端点对网格加密实测（每对 k ≥ 16），核 p50 相对极差 / p95 / 抖动。

    诚实标注：t95_upper 是把各分量的 p95 直接相加得到的**保守上界**，不是
    组合延迟的真实 p95 —— 各段与两个接口的抖动近似独立，真实 p95 明显小于
    分量 p95 之和。要拿到真实值须对端到端做联合采样，属在线仪表的职责（II.6）。
    """
    pairs: list[PairAudit] = []
    max_jit = 0.0
    for f in fronts:
        for a, b in zip(f.nodes, f.nodes[1:]):
            max_jit = max(max_jit, net.get(a, b, cfg.k_gate).jitter)
        for u, backs in backs_by_task.items():
            for bseg in backs:
                for a, b in zip(bseg.nodes, bseg.nodes[1:]):
                    max_jit = max(max_jit, net.get(a, b, cfg.k_gate).jitter)
                fwd = net.get(f.tail, bseg.head, cfg.k_audit)
                loop = net.get(bseg.tail, f.head, cfg.k_audit)
                max_jit = max(max_jit, net.get(f.tail, bseg.head, cfg.k_gate).jitter,
                              net.get(bseg.tail, f.head, cfg.k_gate).jitter)
                t50 = f.delay_ms + fwd.p50 + bseg.delay_ms + loop.p50
                t95 = (
                    f.compute_ms
                    + sum(net.get(x, y, cfg.k_audit).p95 for x, y in zip(f.nodes, f.nodes[1:]))
                    + fwd.p95
                    + bseg.compute_ms
                    + sum(
                        net.get(x, y, cfg.k_audit).p95
                        for x, y in zip(bseg.nodes, bseg.nodes[1:])
                    )
                    + loop.p95
                )
                pairs.append(
                    PairAudit(
                        front=f.label(),
                        back=bseg.label(),
                        task=u,
                        t_front=f.delay_ms,
                        w_fwd=fwd.p50,
                        t_back=bseg.delay_ms,
                        d_loop=loop.p50,
                        t50=t50,
                        t95_upper=t95,
                    )
                )

    by_task: dict[str, list[PairAudit]] = {}
    for p in pairs:
        by_task.setdefault(p.task, []).append(p)

    med = median([p.t50 for p in pairs]) if pairs else 0.0
    rel: dict[str, float] = {}
    for u, ps in by_task.items():
        ts = [p.t50 for p in ps]
        rel[u] = (max(ts) - min(ts)) / med if med else 0.0

    worst = max(rel.values(), default=0.0)
    max95 = max((p.t95_upper for p in pairs), default=0.0)

    unserved = [u for u in (all_task_names or backs_by_task) if not backs_by_task.get(u)]

    reasons: list[str] = []
    ok = True
    if unserved:
        ok = False
        reasons.append(
            f"task {'、'.join(unserved)} 一条后段都没建出来 —— 这类请求在线时无处可去。"
            f"处置：合并到画像重叠的池（推论 III.7.4），或扩容"
        )
    if worst > cfg.eta:
        ok = False
        reasons.append(f"相对均匀性 {worst * 100:.1f}% > η = {cfg.eta * 100:.0f}%")
    if cfg.t_cap95_ms is not None and max95 > cfg.t_cap95_ms:
        ok = False
        reasons.append(f"p95 上界 {max95:.0f}ms > T_cap95 {cfg.t_cap95_ms:.0f}ms")
    if max_jit > cfg.j_cap_ms + 1e-6:
        ok = False
        reasons.append(f"最大单跳抖动 {max_jit:.1f}ms > J_cap {cfg.j_cap_ms:.0f}ms")

    return AuditReport(
        pairs=pairs,
        by_task=by_task,
        median_t50=med,
        rel_spread_by_task=rel,
        worst_rel_spread=worst,
        max_t95_upper=max95,
        max_hop_jitter=max_jit,
        unserved_tasks=unserved,
        passed=ok,
        reasons=reasons,
    )


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #
@dataclass
class PlanResult:
    l0: int
    l0_table: list[L0Candidate]
    capacity: CapacityEstimate
    n_work: int
    quota: dict[str, int]
    fair: dict[str, float]
    backs: dict[str, list[Segment]]
    entries: list[str]
    band: BandResult
    band_curve: list[tuple[float, BandResult]]
    w_cap: float
    fronts_build: list[Segment]
    fronts_final: list[Segment]
    standby: list[Segment]
    trim: TrimResult
    audit: AuditReport
    manifest: DeploymentManifest | None = None
    """逐节点逐层的部署清单 + 前后段组合矩阵。仅当 task 带专家身份
    （TaskProfile.placement）且 union_experts 是 ExpertPlacement 时产出 ——
    没有专家身份就写不出「这一层装哪些专家」。"""
    log: list[str] = field(default_factory=list)
    n_probes: int = 0

    @property
    def n_back_total(self) -> int:
        return sum(len(v) for v in self.backs.values())


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def plan(
    nodes: Sequence[Node],
    model: ModelSpec,
    tasks: Sequence[TaskProfile],
    union_experts,
    net: MeasurementCache,
    cfg: PlannerConfig,
    p_curve: Mapping[int, float],
    *,
    p_min: float = 0.85,
    rent_policy: str = "smallest_sufficient",
    max_feedback: int = 4,
) -> PlanResult:
    node_map = {n.id: n for n in nodes}
    log: list[str] = []

    # ---------------- L₀ ---------------------------------------------------
    l0c, l0_table = choose_l0(model, tasks, union_experts, nodes, p_curve, p_min=p_min)
    l0 = l0c.l0
    log.append(
        f"[L₀] 取 L₀={l0}（p={l0c.p_accuracy:.2f}，前段 {l0c.front_gb:.1f}GB，"
        f"加权总跳数 {l0c.total_hops_weighted:.2f}，前段单节点合格 {l0c.front_single_node_count} 台）"
    )

    front_spec_free = make_front_spec(model, union_experts, l0)
    back_specs = {t.name: make_back_spec(model, t, l0) for t in tasks}

    # ---------------- Step 1：分档估上限 -----------------------------------
    cap = estimate_capacity_by_tier(nodes, front_spec_free, back_specs, tasks)
    log.append(
        f"[Step 1] 分档估算：活跃供给 {cap.active_supply_gb:.0f}GB，"
        f"归零档 {cap.zeroed_supply_gb:.0f}GB（废料率 {cap.waste_ratio * 100:.0f}%），"
        f"单通道需求 {cap.channel_demand_gb:.1f}GB → N_max={cap.n_max}"
        f"（精确位配平 {cap.n_max_slots}）"
    )
    log.extend(f"          {n}" for n in cap.notes)

    # ---------------- Step 2：打折 + 配额 ----------------------------------
    # 文档写 N_work = θ·N_max，其中 N_max 是「活跃供给 ÷ 单通道需求」。这里改用
    # 精确位配平的 n_max_slots 作为 θ 的基数：两者都是上界，但后者严格更紧
    # （它已经把「档内碎片」这一项排除了），对同一个 θ 更不容易高估。
    # θ 仍然吸收它看不见的部分：跳数溢出、网络不可行的配对、churn 备胎。
    base = cap.n_max_slots if cap.n_max_slots > 0 else cap.n_max
    n_work = int(floor(cfg.theta * base))
    if n_work <= 0:
        raise PlanningError(f"N_work = ⌊{cfg.theta}×{base}⌋ = 0，该池群建不出任何通道", log)
    q = largest_remainder([t.lam for t in tasks], n_work)
    quota = {t.name: v for t, v in zip(tasks, q)}
    fair = fair_ratios(quota, {t.name: t.lam for t in tasks})
    n_back_total = sum(quota.values())
    log.append(
        f"[Step 2] N_work = ⌊θ·N_base⌋ = ⌊{cfg.theta}×{base}⌋ = {n_work}"
        f"（N_base 取精确位配平值；内存上界为 {cap.n_max}）；"
        f"配额 {quota}；公平比 "
        + "、".join(f"{u}={v:.2f}" for u, v in fair.items())
        + f"（瓶颈池 {min(fair, key=fair.get)}）"
    )

    n_f_build_target = ceil((1 + cfg.kappa_over) * n_back_total)
    # R_F 只预留**最终**所需的前段条数，不预留超建量：超建（κ_over）是机会性的，
    # 它的目的是给 Step 6 的回环裁剪一个选择池，不值得为它饿死后段。
    rent = select_rent_nodes(
        nodes, front_spec_free, n_back_total + cfg.n_standby, policy=rent_policy
    )
    log.append(f"[租金] R_F（policy={rent_policy}，预留 {len(rent)} 台）= {sorted(rent)}")

    # 端点画像：接口的另一端将来必然落在「能承载前段的节点」里，故以它们为参照系
    front_capable = sorted(n.id for n in nodes if n.usable_gb >= front_spec_free.total_gb())
    head_cost, tail_cost = net.endpoint_costs([n.id for n in nodes], front_capable)
    if cfg.endpoint_w:
        rank = sorted(head_cost.items(), key=lambda kv: kv[1])
        log.append(
            f"[端点项] w={cfg.endpoint_w}；入口接入代价最好 3 台 "
            + "、".join(f"{v}={c:.0f}ms" for v, c in rank[:3])
            + " / 最差 3 台 "
            + "、".join(f"{v}={c:.0f}ms" for v, c in rank[-3:])
        )

    # ---------------- Step 3：建后段 + 平衡 --------------------------------
    free = {n.id for n in nodes}
    backs: dict[str, list[Segment]] = {}
    shortfall_build: dict[str, int] = {}

    # 建段顺序：形态最少（最受约束）的池先建 —— 否则宽松的池会先占光稀缺档。
    order = sorted(tasks, key=lambda t: (len(cap.forms.get(f"back:{t.name}", [])), -t.lam))
    log.append("[Step 3] 建段顺序（受约束者优先）: " + " → ".join(t.name for t in order))

    task_rent: dict[str, frozenset[str]] = {}
    for t in tasks:
        sc = scarcity_rent_nodes(f"back:{t.name}", cap.forms, nodes)
        task_rent[t.name] = rent | sc
        if sc:
            log.append(
                f"[租金] task {t.name} 存在不占 {sorted({node_map[v].tier for v in sc})} 的可行形态"
                f" → 对这些档收稀缺租金"
            )

    for t in order:
        spec = back_specs[t.name]
        build_obj = Objective(
            mu_ms=cfg.mu_ms,
            jitter_w=cfg.jitter_w,
            rent_nu_ms=cfg.rent_nu_ms,
            rent_nodes=task_rent[t.name],
            head_cost=head_cost,
            tail_cost=tail_cost,
            endpoint_w=cfg.endpoint_w,
        )
        segs: list[Segment] = []
        for _ in range(quota[t.name]):
            s = deploy_path(
                spec, sorted(free), node_map, net, build_obj,
                beam_width=cfg.beam_width, prune_topk=cfg.prune_topk,
            )
            if s is None:
                log.append(
                    f"[Step 3] task {t.name} 只建出 {len(segs)}/{quota[t.name]} 条"
                    f"（无可行放置）—— 记入公平比缺口"
                )
                break
            # 质量下限：跳数超出 III.5.2 的整数下界太多，等于建不出（cfg.max_hops_slack）
            floor_hops = hops_min(
                spec.total_gb(), max(node_map[v].usable_gb for v in free)
            )
            if s.hops > floor_hops + cfg.max_hops_slack:
                log.append(
                    f"[Step 3] task {t.name} 第 {len(segs) + 1} 条被质量下限否决："
                    f"{s.label()} 用了 {s.hops} 跳，而剩余池的跳数下界只有 {floor_hops}"
                    f"（{s.delay_ms:.0f}ms，多出的跳每 token 各付一次）—— 记入公平比缺口"
                )
                break
            segs.append(s)
            free -= set(s.nodes)
        backs[t.name] = segs

    # 两轮筛：先间隙检测去慢尾，再字典序收紧
    for t in tasks:
        segs = backs[t.name]
        if len(segs) < 2:
            continue
        gap = detect_gap_robust([s.delay_ms for s in segs], theta1=cfg.gap_theta1, k=cfg.gap_k)
        if gap.tail:
            for line in blame_slow_tail(segs, gap, net):
                log.append(f"[Step 3·间隙] {t.name} 慢尾诊断 — {line}")
        band = band_from_median([s.delay_ms for s in segs], cfg.beta)
        res = tighten_lex(
            segs, back_specs[t.name], band, free, node_map, net, cfg,
            rent_nodes=task_rent[t.name],
            head_cost=head_cost,
            tail_cost=tail_cost,
        )
        backs[t.name] = res.segments
        free = res.free_nodes
        log.append(
            f"[Step 3·收紧] {t.name} 目标区间 [{band[0]:.1f},{band[1]:.1f}]ms，"
            f"{res.committed} 轮提交 / {res.rolled_back} 轮回退，"
            f"(D_max,D_Σ): {res.history[0][0]:.1f}→{res.history[-1][0]:.1f}"
        )

    all_backs = [s for v in backs.values() for s in v]
    if not all_backs:
        raise PlanningError("Step 3 未能建出任何后段", log)
    entries = sorted({s.head for s in all_backs})
    log.append(f"[Step 3] 后段 cluster 冻结，事实入口集 H = {entries}")

    # 条数以**实建**为准，不是以配额为准：Step 3 可能因无可行放置或质量下限
    # 否决而少建。不在这里对齐的话，后面的超建目标、备胎数、终审网格都会按
    # 一个不存在的条数来算。
    if sum(len(v) for v in backs.values()) != n_back_total:
        built = {u: len(v) for u, v in backs.items()}
        for u, want in quota.items():
            if built.get(u, 0) < want:
                shortfall_build[u] = want - built.get(u, 0)
        n_back_total = sum(built.values())
        n_f_build_target = ceil((1 + cfg.kappa_over) * n_back_total)
        quota = built
        fair = fair_ratios(quota, {t.name: t.lam for t in tasks})
        log.append(
            "[Step 3] 实建少于配额（"
            + "、".join(f"{u} 少 {n} 条" for u, n in shortfall_build.items())
            + f"）→ 总条数修正为 {n_back_total}，公平比 "
            + "、".join(f"{u}={v:.2f}" for u, v in fair.items())
        )

    # ---------------- w_cap 定标 -------------------------------------------
    typical = net.typical_hop_p50([n.id for n in nodes])
    w_cap = cfg.w_cap_ms if cfg.w_cap_ms is not None else cfg.rho_w * typical
    w_cap95 = cfg.w_cap95_ms if cfg.w_cap95_ms is not None else w_cap + cfg.j_cap_ms
    log.append(
        f"[w_cap] 典型跳 p50 = {typical:.1f}ms → w_cap = ρ_w×典型跳 = "
        f"{cfg.rho_w}×{typical:.1f} = {w_cap:.1f}ms，w_cap95 = {w_cap95:.1f}ms"
    )

    # ---------------- Step 4：公共中值域（含反馈口） ------------------------
    band_res: BandResult | None = None
    curve: list[tuple[float, BandResult]] = []
    probes: dict[str, ExitProbe] = {}

    # 反馈口也用「整轮回退制」：每轮记录状态，最后回到最好的一轮。
    # 没有这一条，诊断→重建会震荡（拆掉一条好后段换成三跳的，下一轮再换回来）。
    n_f_final_target = n_back_total + cfg.n_standby
    tried: set[str] = set()
    shortfall: dict[str, int] = {}
    best: tuple[tuple, dict[str, list[Segment]], list[str], BandResult, list] | None = None

    def state_key(pop: int, bks: Mapping[str, list[Segment]]) -> tuple:
        worst = max((s.delay_ms for v in bks.values() for s in v), default=0.0)
        return (min(pop, n_f_final_target), pop, -worst)

    for attempt in range(max_feedback + 1):
        cands = sorted(v for v in free if node_map[v].usable_gb >= front_spec_free.total_gb())
        if not cands:
            raise PlanningError("没有空闲节点能承载前段 —— 提高影子租金或降低条数", log)
        probes = probe_exits(cands, entries, net, cfg, w_cap95=w_cap95)
        n_dropped = sum(1 for p in probes.values() if p.dropped)

        med_t = median([s.delay_ms for s in all_backs]) + 2 * typical + l0 * min(
            n.ms_per_layer for n in nodes
        )
        w_max = cfg.eta * med_t
        grid = [round(x, 2) for x in _linspace(0.4 * w_max, 1.5 * w_max, 12)]
        curve = sweep_bandwidth(probes, entries, w_cap, grid)
        band_res, why = pick_bandwidth(curve, n_f_build_target, w_max)
        log.append(
            f"[Step 4] 候选出口 {len(cands)} 台，闸门剔除 {n_dropped} 台；"
            f"median T50 估计 {med_t:.0f}ms → W 上限 {w_max:.1f}ms；{why}；"
            f"带 [{band_res.w_lo:.1f},{band_res.w_hi:.1f}]，Q_F^out={band_res.exits}"
        )
        log.append(
            "[Step 4] 人口曲线 (W, |S|): "
            + ", ".join(f"{W:.1f}→{r.population}" for W, r in curve if W <= w_max + 1e-9)
        )

        key = state_key(band_res.population, backs)
        if best is None or key > best[0]:
            best = (key, {u: list(v) for u, v in backs.items()}, list(entries), band_res, curve)

        if band_res.population >= n_f_build_target or attempt == max_feedback:
            break

        # --- 反馈动作一：异类入口 → 回炉重建该后段换入口（II.3.3） ---------
        diag = diagnose_outlier_entries(probes, entries, band_res.W, w_cap)
        pick = pick_outlier(diag, exclude=tried)
        if pick is not None:
            outlier, pop_after, jump = pick
            tried.add(outlier)
            log.append(
                f"[Step 4→3 反馈] 异类入口诊断：剔除 {outlier} 后人口 "
                f"{band_res.population}→{pop_after}（+{jump}，第二名 "
                f"+{max((d[2] for d in diag[1:]), default=0)}）→ 回炉重建该后段换入口"
            )
            if _rebuild_entry(
                outlier, backs, back_specs, free, node_map, net, cfg, task_rent,
                head_cost, tail_cost, log,
            ):
                all_backs = [s for v in backs.values() for s in v]
                free = {n.id for n in nodes} - {v for s in all_backs for v in s.nodes}
                entries = sorted({s.head for s in all_backs})
                continue
            log.append(f"[Step 4→3 反馈] {outlier} 所在后段重建失败")
        else:
            log.append(
                "[Step 4] 异类入口诊断无显著跃升（"
                + "、".join(f"{v}:+{j}" for v, _, j in diag)
                + "）—— 人口不足不是某个坏入口造成的"
            )

        # --- 反馈动作二：降条数并记入公平比缺口（II.4「迭代」） -------------
        # 每条后段都往公共带上加一个约束（它的入口要被所有候选出口同时看齐），
        # 所以入口少一个，带就宽松一分。人口凑不够时的正确动作不是先放宽 η，
        # 而是先降条数 —— 这正是文档「某池建不出目标条数 → 回 Step 2 调配额
        # 或降该池条数并记入公平比缺口」。降谁：公平比最高（最不委屈）的池，
        # 且优先降它入口接入质量最差的那一条。
        if band_res.population >= n_back_total:
            break
        cur_fair = fair_ratios(
            {u: len(v) for u, v in backs.items()}, {t.name: t.lam for t in tasks}
        )
        donors = sorted((u for u, v in backs.items() if len(v) > 1), key=lambda u: -cur_fair[u])
        if not donors:
            break
        donor = donors[0]
        victim = max(
            range(len(backs[donor])), key=lambda i: head_cost.get(backs[donor][i].head, 0.0)
        )
        dropped = backs[donor].pop(victim)
        shortfall[donor] = shortfall.get(donor, 0) + 1
        n_back_total -= 1
        n_f_build_target = ceil((1 + cfg.kappa_over) * n_back_total)
        n_f_final_target = n_back_total + cfg.n_standby
        best = None  # 条数变了，之前记下的最优状态不可比
        log.append(
            f"[Step 4→2 反馈] 人口 {band_res.population} < 后段条数 {n_back_total + 1}："
            f"降 {donor} 池一条（{dropped.label()}，入口接入代价 "
            f"{head_cost.get(dropped.head, 0):.0f}ms 为该池最差），记入公平比缺口；"
            f"新总条数 {n_back_total}"
        )
        all_backs = [s for v in backs.values() for s in v]
        free = {n.id for n in nodes} - {v for s in all_backs for v in s.nodes}
        entries = sorted({s.head for s in all_backs})

    assert band_res is not None and best is not None

    # 回到最好的一轮
    if state_key(band_res.population, backs) < best[0]:
        _, backs, entries, band_res, curve = best
        all_backs = [s for v in backs.values() for s in v]
        free = {n.id for n in nodes} - {v for s in all_backs for v in s.nodes}
        log.append(
            f"[Step 4] 反馈整轮回退 —— 采用人口最优的一轮：|Q_F^out|={band_res.population}，H={entries}"
        )
    if band_res.population < n_back_total:
        raise PlanningError(
            f"公共中值域人口 {band_res.population} < 后段条数 {n_back_total}。"
            f"在 W ≤ η·median 内凑不够，异类入口诊断与降条数都已用尽。三条出路"
            f"（II.3.3 / II.6）：\n"
            f"  ① 放宽 w_cap 或 η —— 代价是绝对延迟上升，须核 β 政策；\n"
            f"  ② 降低条数（回 Step 2 调配额，把缺口记入公平比）；\n"
            f"  ③ 若逐个剔除入口后 |S| 始终无改善，说明节点池天然分属互联不佳的\n"
            f"     多个域 —— 正确处置是按域各建一套完整前后段独立部署，"
            f"而不是跨域拉一条链。\n"
            f"  当前带 [{band_res.w_lo:.1f},{band_res.w_hi:.1f}]，W={band_res.W:.1f}ms，"
            f"H={entries}",
            log,
        )

    if shortfall or shortfall_build:
        quota = {u: len(v) for u, v in backs.items()}
        fair = fair_ratios(quota, {t.name: t.lam for t in tasks})
        log.append(
            "[公平比缺口] "
            + "、".join(
                f"{u} 少建 {shortfall.get(u, 0) + shortfall_build.get(u, 0)} 条"
                for u in sorted(set(shortfall) | set(shortfall_build))
            )
            + " → 修正后配额 "
            + str(quota)
            + "，公平比 "
            + "、".join(f"{u}={v:.2f}" for u, v in fair.items())
            + f"（瓶颈池 {min(fair, key=fair.get)}，扩容第一条加给它）"
        )

    if band_res.population < n_back_total + cfg.n_standby:
        log.append(
            f"[预警] 公共带人口 {band_res.population} 只够 {n_back_total} 条主用 + "
            f"{band_res.population - n_back_total} 条备胎（目标 {cfg.n_standby} 条）——"
            f"churn 时无热备可即刻顶替，须走 II.6 即时层重构，恢复时延变长"
        )

    # ---------------- Step 5：建前段 + 平衡（故意超建） ---------------------
    q_out = frozenset(band_res.exits)
    n_f_build = min(len(q_out), n_f_build_target)
    front_spec = make_front_spec(model, union_experts, l0, tail_allowed=q_out)
    fronts: list[Segment] = []
    pool = set(free)
    front_obj = Objective(mu_ms=cfg.mu_ms, jitter_w=cfg.jitter_w)
    for _ in range(n_f_build):
        s = deploy_path(
            front_spec, sorted(pool), node_map, net, front_obj,
            beam_width=cfg.beam_width_front, prune_topk=cfg.prune_topk,
        )
        if s is None:
            break
        fronts.append(s)
        pool -= set(s.nodes)
    log.append(
        f"[Step 5] 超建目标 {n_f_build_target} 条（κ_over={cfg.kappa_over}），"
        f"受 |Q_F^out|={len(q_out)} 封顶为 {n_f_build}，实建 {len(fronts)} 条，"
        f"全部 {max((s.hops for s in fronts), default=0)} 跳"
    )

    if len(fronts) >= 2:
        fband = band_from_median([s.delay_ms for s in fronts], cfg.beta)
        fres = tighten_lex(fronts, front_spec, fband, pool, node_map, net, cfg)
        fronts, pool = fres.segments, fres.free_nodes
        log.append(
            f"[Step 5·收紧] 前段目标区间 [{fband[0]:.1f},{fband[1]:.1f}]ms，"
            f"{fres.committed} 轮提交，前段极差 "
            f"{max(s.delay_ms for s in fronts) - min(s.delay_ms for s in fronts):.1f}ms"
        )

    # ---------------- Step 6：回环裁剪 + 终审 ------------------------------
    profs = loop_profile(fronts, all_backs, net, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms)
    gated = [p for p in profs if p.gated]
    if gated:
        log.append(
            "[Step 6·抖动闸] 出局 "
            + "、".join(f"{p.front_label}({p.gated})" for p in gated)
        )
    n_live = sum(1 for p in profs if p.gated is None)
    n_f_final = min(n_live, n_back_total + cfg.n_standby)
    if n_live < n_back_total:
        raise PlanningError(
            f"回环抖动闸过后只剩 {n_live} 条前段可用，少于后段条数 {n_back_total}。"
            f"回环是 decode 每 token 必付的一跳，抖动超限的链路不能用。"
            f"出路：放宽 J_cap（须核 p95 尾闸）、或换接入更稳的节点做前段入口",
            log,
        )
    trim = trim_by_loop(profs, n_f_final)
    fronts_final = [fronts[i] for i in trim.kept]
    standby = [fronts[i] for i in trim.standby]
    log.append(
        "[Step 6] 回环画像 D(f): "
        + ", ".join(f"{p.front_label}={p.D:.1f}" for p in trim.profiles)
    )
    log.append(
        f"[Step 6] 升序取前 {n_f_final} 条 → 最坏回环 {trim.worst_D:.1f}ms；"
        f"保留集极差 {trim.spread_kept:.1f}ms ≤ 全集极差 {trim.spread_all:.1f}ms"
        f"（命题 III.8.3 的副产品）；备胎 {len(standby)} 条"
    )

    audit = _audit(fronts_final, backs, net, cfg, [t.name for t in tasks])
    log.append(
        f"[终审] 网格 {len(fronts_final)}×{sum(len(v) for v in backs.values())} = "
        f"{len(audit.pairs)} 组，每组 k={cfg.k_audit}；"
        f"最坏相对极差 {audit.worst_rel_spread * 100:.1f}% vs η={cfg.eta * 100:.0f}% → "
        + ("通过" if audit.passed else "未通过：" + "；".join(audit.reasons))
    )

    # ---------------- 部署清单 + 组合矩阵 ----------------------------------
    manifest: DeploymentManifest | None = None
    back_plc = {t.name: t.placement for t in tasks if t.placement is not None}
    if isinstance(union_experts, ExpertPlacement) and len(back_plc) == len(tasks):
        manifest = build_manifest(
            model=model,
            model_name=f"{model.n_layers}L-d{model.d_model}-E{model.n_experts}",
            l0=l0,
            fronts=fronts_final,
            standby=standby,
            backs=backs,
            front_placement=union_experts,
            back_placements=back_plc,
            node_map=node_map,
            net=net,
            cfg=cfg,
            band=(band_res.w_lo, band_res.w_hi),
            w_cap=w_cap,
        )
        n_pairs = len(manifest.pairings)
        log.append(
            f"[清单] {len(manifest.nodes)} 个节点分配，组合矩阵 {n_pairs} 组"
            f"（{len(fronts_final)} 前段 × {n_back_total} 后段）；校验"
            + ("通过" if manifest.ok else f"未通过 {len(manifest.violations)} 项：" + manifest.violations[0])
        )
    elif isinstance(union_experts, ExpertPlacement) or back_plc:
        log.append("[清单] 专家身份不完整（部分 task 无 placement）—— 跳过部署清单")

    return PlanResult(
        l0=l0,
        l0_table=l0_table,
        capacity=cap,
        n_work=n_work,
        quota=quota,
        fair=fair,
        backs=backs,
        entries=entries,
        band=band_res,
        band_curve=curve,
        w_cap=w_cap,
        fronts_build=fronts,
        fronts_final=fronts_final,
        standby=standby,
        trim=trim,
        audit=audit,
        manifest=manifest,
        log=log,
        n_probes=net.n_probes,
    )


# --------------------------------------------------------------------------- #
def _rebuild_entry(
    outlier: str,
    backs: dict[str, list[Segment]],
    back_specs: Mapping[str, SegmentSpec],
    free: set[str],
    node_map: Mapping[str, Node],
    net: MeasurementCache,
    cfg: PlannerConfig,
    task_rent: Mapping[str, frozenset[str]],
    head_cost: Mapping[str, float],
    tail_cost: Mapping[str, float],
    log: list[str],
) -> bool:
    """Step 4 → Step 3 的反馈口：让以 outlier 为入口的后段回炉重建，换入口。"""
    for u, segs in backs.items():
        for i, s in enumerate(segs):
            if s.head != outlier:
                continue
            pool = (free | set(s.nodes)) - {outlier}
            obj = Objective(
                mu_ms=cfg.mu_ms,
                jitter_w=cfg.jitter_w,
                rent_nu_ms=cfg.rent_nu_ms,
                rent_nodes=task_rent.get(u, frozenset()),
                head_cost=head_cost,
                tail_cost=tail_cost,
                endpoint_w=cfg.endpoint_w,
            )
            got = deploy_path(
                back_specs[u], sorted(pool), node_map, net, obj,
                beam_width=cfg.beam_width, prune_topk=cfg.prune_topk,
            )
            if got is None:
                return False
            log.append(
                f"[Step 4→3 反馈] {u} 的 {s.label()} → {got.label()}"
                f"（{s.delay_ms:.1f} → {got.delay_ms:.1f}ms）"
            )
            segs[i] = got
            return True
    return False


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]
