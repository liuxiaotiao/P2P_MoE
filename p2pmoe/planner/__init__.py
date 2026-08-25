"""离线规划器 —— 方案文档第二部分（算法）的完整落地。

本包**不依赖任何推理框架**：它的输入只有三张表 —— 节点内存、逐对延迟实测、
逐层内存形状。因此框架选型（HF / vLLM / 原生 torch）不阻塞规划器的开发与验证。
"""

from .types import Node, ModelSpec, TaskProfile, SegmentSpec, Segment, Objective, PlannerConfig
from .network import Probe, NetworkOracle, MeasurementCache
from .hf_config import model_spec_from_hf, granularity_verdict, HFModelInfo, PRESETS
from .memory import hops_min, choose_l0, make_front_spec, make_back_spec, feasibility_necessary
from .capacity import estimate_capacity_by_tier, largest_remainder, fair_ratios
from .solver import deploy_path
from .tighten import detect_gap_robust, tighten_lex, band_from_median
from .common_band import probe_exits, common_band, sweep_bandwidth, pick_bandwidth
from .loop_trim import loop_profile, trim_by_loop
from .experts import (
    ActivationProfile, ExpertPlacement, build_placement, union_placement,
    detectability, detectability_matrix, merge_candidates, expected_detection_tokens,
)
from .manifest import DeploymentManifest, NodePlan, LayerLoad, Pairing, build_manifest
from .pipeline import plan, PlanResult, AuditReport

__all__ = [
    "Node", "ModelSpec", "TaskProfile", "SegmentSpec", "Segment", "Objective", "PlannerConfig",
    "Probe", "NetworkOracle", "MeasurementCache",
    "model_spec_from_hf", "granularity_verdict", "HFModelInfo", "PRESETS",
    "hops_min", "choose_l0", "make_front_spec", "make_back_spec", "feasibility_necessary",
    "estimate_capacity_by_tier", "largest_remainder", "fair_ratios",
    "deploy_path",
    "detect_gap_robust", "tighten_lex", "band_from_median",
    "probe_exits", "common_band", "sweep_bandwidth", "pick_bandwidth",
    "loop_profile", "trim_by_loop",
    "ActivationProfile", "ExpertPlacement", "build_placement", "union_placement",
    "detectability", "detectability_matrix", "merge_candidates", "expected_detection_tokens",
    "DeploymentManifest", "NodePlan", "LayerLoad", "Pairing", "build_manifest",
    "plan", "PlanResult", "AuditReport",
]
