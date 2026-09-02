"""重新下发清单时，旧模型必须先被放掉。

真机上的症状：第二次 `configure` 报 `CUDA out of memory`，而报错里
「保留但未用」只有 1.58MiB —— 不是碎片，是旧张量还被引用着。

原因是顺序：`SelectiveLoader.load()` 把新张量全部分配完，最后才
`self.model = SegModel(...)`。整个装载过程里旧模型都挂在 `self` 上，
显存峰值是两份。14.6GB 的前段配 22GB 的卡，一份富余、两份必炸。

后果不只是这一次报错：它让同一批 agent 只能被 configure 一次，而
`measure` 每跑一遍都重新下发清单，于是每次实验前都必须重启 15 个进程。
"""

from __future__ import annotations

import gc
import weakref

from p2pmoe.runtime import node as node_mod
from p2pmoe.runtime.node import NodeConfig, NodeServer
from p2pmoe.runtime.wire import LinkTable


def _server() -> NodeServer:
    return NodeServer("n0", host="127.0.0.1", port=0)


def _cfg(layers: dict[int, list[int]]) -> NodeConfig:
    return NodeConfig(
        node_id="n0", role="back:X", segment="BX0",
        layer_experts=layers,
        next_hop=None, seg_head="n0", is_head=True, is_tail=True,
        peers={}, links=LinkTable().to_dict(),
        coordinator=("127.0.0.1", 1), model={},
    )


def test_the_old_model_is_gone_before_the_new_one_is_built(monkeypatch) -> None:
    """这是**顺序**的测试，不是「最终释放了吗」的测试。

    最终会不会释放，垃圾回收迟早都会办到；要紧的是它得发生在新张量开始
    分配**之前**，否则峰值仍然是两份，OOM 照样发生。所以这里在构造新模型
    的那一刻回头看 `self.model` —— 必须已经是 None。
    """
    s = _server()
    s.apply_config(_cfg({1: [0, 1, 2]}))
    assert s.model is not None

    seen: list[object] = []
    real = node_mod.SegmentModel

    class Probe(real):                     # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            seen.append(s.model)           # 造新的这一刻，旧的还在不在？
            super().__init__(*a, **k)

    monkeypatch.setattr(node_mod, "SegmentModel", Probe)
    s.apply_config(_cfg({2: [3, 4]}))

    assert seen == [None], "装新模型之前没有先把旧的放掉 —— 显存峰值会是两份"


def test_the_old_model_is_actually_collected() -> None:
    """光把字段置空不够 —— 模型是模块套模块、父子互指，成环。

    环靠引用计数放不掉，要等下一次 gc。而下一次 gc 很可能发生在新张量已经
    分配之后，那就白让了。所以 `_release_model` 里的 `gc.collect()` 不能省。
    """
    s = _server()
    s.apply_config(_cfg({1: [0, 1]}))
    ref = weakref.ref(s.model)
    assert ref() is not None

    s.apply_config(_cfg({1: [0, 1]}))
    assert ref() is None, "旧模型还活着 —— 显存不会回来"


def test_releasing_does_not_reach_for_torch_on_a_cpu_node() -> None:
    """CPU / toy 节点上不该去碰 CUDA。

    `release_cuda_cache()` 要 import torch，而 toy 后端的节点根本没装 torch
    （requirements-node.txt 是可选的）。按 device 前缀分流，不是无条件调用。
    """
    s = _server()
    s.apply_config(_cfg({1: [0]}))

    called = []
    import p2pmoe.runtime.weights as w
    real = w.release_cuda_cache
    w.release_cuda_cache = lambda: called.append(1)  # type: ignore[assignment]
    try:
        s.apply_config(_cfg({1: [0]}))          # cfg.device 默认不是 cuda
    finally:
        w.release_cuda_cache = real             # type: ignore[assignment]

    assert called == [], "CPU 节点不该调用 release_cuda_cache"


def test_release_is_a_no_op_on_the_first_configure() -> None:
    """第一次配置时没有旧模型，不该报错也不该做任何事。"""
    s = _server()
    s._release_model()          # 还没 apply_config 过
    assert s.model is None
    s.apply_config(_cfg({1: [0]}))
    assert s.model is not None


def test_release_cuda_cache_lives_in_the_execution_layer() -> None:
    """torch 只许出现在执行层。

    和 `cuda_state()` 同一条线：`node.py` 属于控制面，一个 torch 都不许
    import（`test_heavy_deps_stay_in_the_execution_layer` 守着）。所以清缓存
    这件事必须住在 `weights.py` 里，由 node 按需要延迟 import。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "def release_cuda_cache" in (
        root / "p2pmoe/runtime/weights.py").read_text(encoding="utf-8")

    src = (root / "p2pmoe/runtime/node.py").read_text(encoding="utf-8")
    assert "release_cuda_cache" in src
    for line in src.splitlines():
        assert not line.strip().startswith("import torch"), \
            "node.py 不许 import torch"


def test_gc_is_actually_invoked() -> None:
    """守着 `gc.collect()` 那一行别被当成多余的清理代码删掉。"""
    s = _server()
    s.apply_config(_cfg({1: [0]}))
    n = 0

    real = gc.collect

    def counting(*a, **k):
        nonlocal n
        n += 1
        return real(*a, **k)

    gc.collect = counting          # type: ignore[assignment]
    try:
        s.apply_config(_cfg({1: [0]}))
    finally:
        gc.collect = real          # type: ignore[assignment]

    assert n >= 1, "重新配置时没有触发 gc —— 成环的旧模型放不掉"
