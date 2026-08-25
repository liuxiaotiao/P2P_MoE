"""在线运行时：toy MoE 执行层、通信层、节点进程、在线协议。

对应文档第二部分 II.5（在线协议：零计算）与 II.6（维护）。
`p2pmoe.planner` 产出施工图，这里把它跑起来。
"""

from .model import ToyMoEConfig, MoEStats, PartialExpertMoEBlock, SegmentModel
from .corpus import make_corpus, profile_from_corpus, sample_prompt
from .identify import HistogramClassifier, Verdict
from .wire import LinkTable, PeerPool
from .node import NodeConfig, NodeServer, run_node
from .coordinator import Coordinator, LocalCluster, RequestRecord

# 真实模型执行层是**可选**的（需要 torch + safetensors）。控制机不装也能跑规划、
# 探测、下发 —— 所以这里软导入，缺依赖时只是这几个名字不可用。
try:
    from .torch_model import TorchModelConfig, TorchPartialExpertMoEBlock, TorchSegmentModel
    from .weights import KeyPlan, SelectiveLoader, WeightIndex, qwen_moe_keys
    HAS_TORCH_BACKEND = True
except ImportError:  # pragma: no cover
    HAS_TORCH_BACKEND = False

__all__ = [
    "ToyMoEConfig", "MoEStats", "PartialExpertMoEBlock", "SegmentModel",
    "make_corpus", "profile_from_corpus", "sample_prompt",
    "HistogramClassifier", "Verdict",
    "LinkTable", "PeerPool",
    "NodeConfig", "NodeServer", "run_node",
    "Coordinator", "LocalCluster", "RequestRecord",
    "HAS_TORCH_BACKEND",
]
if HAS_TORCH_BACKEND:
    __all__ += [
        "TorchModelConfig", "TorchPartialExpertMoEBlock", "TorchSegmentModel",
        "KeyPlan", "SelectiveLoader", "WeightIndex", "qwen_moe_keys",
    ]
