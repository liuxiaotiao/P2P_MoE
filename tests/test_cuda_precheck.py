"""要 GPU 就先确认每台真有可用的 GPU。

不查的话，CUDA 起不来的那台会挂在装载里（safetensors 把张量往一个不存在的
设备上搬），控制机等满 120 秒才发现 —— 而节点日志里只有一行
`CUDA unknown error` 的 UserWarning，淹在几十行正常输出里。

真踩过，查了好几轮才看见那一行。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import p2pmoe.deploy.control as C
from p2pmoe.runtime.node import _cuda_state

ROOT = Path(__file__).resolve().parent.parent


class _M:
    nodes: list = []
    segments: dict = {}


def _call(device, caps):
    return C.distribute(_M(), {}, None, None, ("1.2.3.4", 1), 11,
                        device=device, caps_raw=caps)


def test_a_node_without_cuda_stops_the_run(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            _call("cuda:0", {"N10": {"cuda": {"ok": False, "why": "CUDA unknown error"}}})
    assert "N10" in caplog.text


def test_the_check_runs_before_anything_else() -> None:
    """排在分组、内存账之后的话，那些步骤会先崩在别的地方 ——
    而那些崩溃与真正的原因无关。"""
    src = (ROOT / "p2pmoe" / "deploy" / "control.py").read_text(encoding="utf-8")
    i = src.index("def distribute(")
    body = src[i:i + 6000]
    i_cuda = body.index("CUDA 用不了")
    # 比的是这些步骤的**使用点**，不是函数签名里的形参
    for later in ("by_seg[", "if node_caps:"):
        assert later in body, f"找不到 {later}"
        assert i_cuda < body.index(later), f"CUDA 检查排在 {later} 后面了"


def test_cpu_runs_are_not_blocked() -> None:
    """CPU 跑不需要 GPU —— 别拦。"""
    _call("cpu", {"N10": {"cuda": {"ok": False, "why": "没有"}}})


def test_healthy_gpus_are_not_blocked() -> None:
    _call("cuda:0", {"N10": {"cuda": {"ok": True, "n": 1, "name": "RTX 6000"}}})


def test_the_advice_covers_persistence_mode_and_the_cpu_fallback(caplog) -> None:
    """`CUDA unknown error` 最常见的成因是驱动反复加载卸载；
    而「先拿到一份结果」的出路是走 CPU —— 两条都要说。"""
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            _call("cuda:0", {"N": {"cuda": {"ok": False, "why": "CUDA unknown error"}}})
    assert "nvidia-smi -pm 1" in caplog.text
    assert "DEVICE=cpu" in caplog.text


# --------------------------------------------------------------------------- #
def test_cuda_state_never_raises() -> None:
    """探测本身不能把 agent 带走 —— 它在 capabilities 里被调用，
    而 capabilities 是**未配置态也要能答**的那个。"""
    st = _cuda_state()
    assert isinstance(st, dict) and "ok" in st
    if not st["ok"]:
        assert st.get("why"), "不可用时要说清为什么"


def test_cuda_state_actually_allocates() -> None:
    """`is_available()` 为真也可能在真正分配时才炸（驱动刚卸载、GPU 被独占、
    ECC 复位中）—— 光问不够，要真摸一次。"""
    src = (ROOT / "p2pmoe" / "runtime" / "weights.py").read_text(encoding="utf-8")
    i = src.index("def cuda_state")
    blk = src[i:i + 1400]
    assert "torch.zeros" in blk, "只问了 is_available，没真分配"


def test_the_probe_lives_in_the_execution_layer() -> None:
    """`node.py` 不 import torch 是条硬边界 —— 控制面与数据面的装配代码
    不该拖进几 GB 的 CUDA 依赖。探测要放在执行层，node.py 只转发。

    这条是被 `test_heavy_deps_stay_in_the_execution_layer` 当场抓出来的。
    """
    import re

    node = (ROOT / "p2pmoe" / "runtime" / "node.py").read_text(encoding="utf-8")
    # 只看真正的 import 语句 —— 注释里提到 torch 是好事（说明为什么不引）
    real = re.findall(r"^\s*(?:import torch|from torch\b)", node, re.M)
    assert not real, f"node.py 真的 import 了 torch：{real}"
    i = node.index("def _cuda_state")
    assert "from .weights import cuda_state" in node[i:i + 1200]


def test_capabilities_carries_it() -> None:
    """未配置的 agent 也要答得出 —— 检查发生在下发清单**之前**。"""
    src = (ROOT / "p2pmoe" / "runtime" / "node.py").read_text(encoding="utf-8")
    i = src.index("def capabilities")
    assert '"cuda"' in src[i:i + 900]
