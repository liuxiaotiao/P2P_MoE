"""附录 C 的舞台：24 节点分散池，V100 → A40。

模型：32 层，d_model = 4096，每层 64 专家（top-2），基座 0.13GB/层，
单专家 0.27GB。回放语料给出的逐层驻留专家数：
    task X (λ=0.5)  8 专家 → 2.29GB/层
    task Y (λ=0.3)  6 专家 → 1.75GB/层
    task Z (λ=0.2)  7 专家 → 2.02GB/层
    并集           20 专家 → 5.53GB/层
ctx_max = 4096 → KV = 0.067GB/层。

节点池（无机房，各自接入）：4× A40(48GB)、10× V100-32(32GB)、
6× V100-16(16GB)、4× 杂项小卡(16GB)。算力 A40 0.35ms/层·token、
V100 0.50ms/层·token（近同质，差 1.4×）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..planner.experts import ExpertPlacement, build_placement, union_placement
from ..planner.network import MeasurementCache
from ..planner.types import ModelSpec, Node, PlannerConfig, TaskProfile
from .network import SimNetwork
from .replay import make_activation_profiles

__all__ = ["APPENDIX_C_MODEL", "APPENDIX_C_TASKS", "UNION_EXPERTS", "P_CURVE", "Stage", "appendix_c"]


APPENDIX_C_MODEL = ModelSpec(
    n_layers=32,
    d_model=4096,
    n_experts=64,
    top_k=2,
    base_gb_per_layer=0.13,
    expert_gb=0.27,
    ctx_max=4096,
)

APPENDIX_C_TASKS = [
    TaskProfile(name="X", lam=0.5, experts_per_layer=8),
    TaskProfile(name="Y", lam=0.3, experts_per_layer=6),
    TaskProfile(name="Z", lam=0.2, experts_per_layer=7),
]

UNION_EXPERTS = 20

P_CURVE = {2: 0.68, 3: 0.79, 4: 0.86, 5: 0.91, 6: 0.93, 7: 0.94, 8: 0.945}
"""识别准确率 p(L₀)，由回放语料给出。文档 C.1 只列了 L₀=3/4/5 三点，
这里向两侧延伸以便看清「多一层的边际收益递减 vs 多一跳的数十毫秒」这个取舍。"""


def build_nodes() -> list[Node]:
    nodes: list[Node] = []
    for i in range(4):
        nodes.append(Node(id=f"g{i+1}", tier="A40", mem_gb=48.0, ms_per_layer=0.35, avail=0.97))
    for i in range(10):
        nodes.append(Node(id=f"v{i+1}", tier="V100-32", mem_gb=32.0, ms_per_layer=0.50, avail=0.95))
    for i in range(6):
        nodes.append(Node(id=f"s{i+1}", tier="V100-16", mem_gb=16.0, ms_per_layer=0.50, avail=0.93))
    for i in range(4):
        nodes.append(Node(id=f"m{i+1}", tier="misc-16", mem_gb=16.0, ms_per_layer=0.50, avail=0.90))
    return nodes


@dataclass
class Stage:
    nodes: list[Node]
    model: ModelSpec
    tasks: list[TaskProfile]
    union_experts: object
    """int（只有基数）或 ExpertPlacement（有专家身份）。"""
    net: MeasurementCache
    sim: SimNetwork
    cfg: PlannerConfig
    p_curve: dict[int, float]
    profiles: dict = field(default_factory=dict)
    """逐 task 的激活质量画像（ActivationProfile）。"""
    placements: dict = field(default_factory=dict)
    """逐 task 的驻留专家集（ExpertPlacement）。"""


def appendix_c(
    *,
    seed: int = 7,
    eta: float = 0.12,
    j_cap: float = 25.0,
    beta: float = 1.25,
    with_experts: bool = False,
    coverage: float = 0.97,
    shared_core: int = 1,
) -> Stage:
    """构造算例 C 的完整舞台。

    网络参数按 C.0 的口径标定：逐对 p50 22–75ms、抖动 p95−p50 8–40ms，
    接入质量差异显著且与算力内存无关。
    """
    nodes = build_nodes()
    ids = [n.id for n in nodes]
    sim = SimNetwork(
        ids,
        seed=seed,
        good_access=(12.0, 16.0),
        bad_access=(28.0, 33.0),
        bad_frac=0.25,
        backbone=(2.0, 5.0),
        jitter=(4.0, 9.0),
    )
    cfg = PlannerConfig(
        eta=eta,
        beta=beta,
        j_cap_ms=j_cap,
        rho_w=1.5,
        k_probe=8,
        k_audit=16,
        theta=0.8,
        kappa_over=0.3,
        n_standby=1,
        seed=seed,
    )
    net = MeasurementCache(sim, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)

    tasks = list(APPENDIX_C_TASKS)
    union: object = UNION_EXPERTS
    profiles: dict = {}
    placements: dict = {}
    if with_experts:
        # 覆盖率阈值取 0.97：算例 C.5 说 X 池的 miss 基线是 3%，而 II.5 定义
        # 「基线 = 1 − 覆盖率」，反解即 0.97。
        profiles = make_activation_profiles(
            [t.name for t in APPENDIX_C_TASKS],
            {t.name: (t.experts_per_layer if isinstance(t.experts_per_layer, int) else 8)
             for t in APPENDIX_C_TASKS},
            n_layers=APPENDIX_C_MODEL.n_layers,
            n_experts=APPENDIX_C_MODEL.n_experts,
            coverage=coverage,
            shared_core=shared_core,
            seed=seed,
        )
        placements = {u: build_placement(pr, coverage) for u, pr in profiles.items()}
        tasks = [
            TaskProfile(name=t.name, lam=t.lam,
                        experts_per_layer=t.experts_per_layer,
                        placement=placements[t.name])
            for t in APPENDIX_C_TASKS
        ]
        union = union_placement([placements[t.name] for t in APPENDIX_C_TASKS])

    return Stage(
        nodes=nodes,
        model=APPENDIX_C_MODEL,
        tasks=tasks,
        union_experts=union,
        net=net,
        sim=sim,
        cfg=cfg,
        profiles=profiles,
        placements=placements,
        p_curve=dict(P_CURVE),
    )
