"""真机部署：节点 agent、真实网络探测、控制器。

    每台节点:  python -m p2pmoe.deploy.agent   --id v1 --bind 0.0.0.0:9101
    控制机:    python -m p2pmoe.deploy.control --agents v1=10.0.0.11:9101,… --advertise 10.0.0.5

与 runtime.LocalCluster 的差别只有三处，其余代码完全共用：
  1. agent 两阶段启动（先起服务、后收清单）—— 否则「量延迟要先起 agent、
     起 agent 要先知道装什么」这个循环解不开；
  2. 探测由**节点自己**发起（deploy/probe.py）—— 从控制器 ping 量到的是控制器
     的接入质量，不是两台节点之间的；
  3. 不注入延迟 —— 真实网络自己会给。
"""

from .probe import RemoteNetworkOracle

__all__ = ["RemoteNetworkOracle"]
