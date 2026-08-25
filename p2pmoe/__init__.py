"""P2P MoE 双段模型放置优化 —— 分散环境 serving 框架。

对应技术方案 v24.0《P2P MoE 双段模型放置优化 — 分散环境完整方案》。

包结构
------
p2pmoe.planner   离线规划器（文档第二部分 II.1–II.7），纯 Python + numpy，
                 不依赖任何推理框架
p2pmoe.sim       分散网络与节点池模拟器，实现 planner 的 NetworkOracle 接口
"""

__version__ = "0.1.0"
