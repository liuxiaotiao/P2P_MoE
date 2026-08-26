"""激活画像：采、存、用。

这一层要成立靠一个事实：**路由是全量的，即使专家不是**。
`mlp.gate.weight` 每层都完整加载，所以 `MoEStats.hist` 记的是 top-k 在全部 E 个
专家上的分布 —— 哪怕本地只驻留了几个。第 1 节就是在钉住这条。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from p2pmoe.runtime.profile import (
    ActivationRecord,
    LayerProfiler,
    load_profile,
    merge_records,
    placement_from_profile,
    save_profile,
    summarise,
    to_activation_profile,
)


# --------------------------------------------------------------------------- #
# 1. 采：路由是全量的
# --------------------------------------------------------------------------- #
def test_routing_is_observed_over_all_experts_even_with_a_subset() -> None:
    """**整套采样方案的前提。** 只驻留 2 个专家时，直方图仍覆盖全部 8 个。

    如果只能看到驻留的那几个，画像就永远只会确认已有的选择，
    「换一批专家会不会更好」这个问题就问不出来了。
    """
    from p2pmoe.runtime.model import PartialExpertMoEBlock, ToyMoEConfig

    cfg = ToyMoEConfig(n_layers=4, d_model=32, n_experts=8, d_ff=32, vocab=64)
    block = PartialExpertMoEBlock(cfg, 1, [0, 1])          # 只装 2 个
    rng = np.random.default_rng(0)
    _, st = block.forward(rng.normal(size=(16, cfg.d_model)).astype(np.float32), {})

    assert st.hist.shape == (cfg.n_experts,)
    assert (st.hist > 0).sum() > 2, "只看得到驻留的那两个，画像就没有意义了"
    assert st.hist.sum() == pytest.approx(st.n_token_layer * 1.0, rel=0.2)


def test_profiler_keeps_layers_apart() -> None:
    """驻留集是**逐层**决定的（n_{u,l} 异构），所以直方图不能先合并再记。"""
    from p2pmoe.runtime.model import SegmentModel, ToyMoEConfig, embed_tokens

    cfg = ToyMoEConfig(n_layers=4, d_model=32, n_experts=8, d_ff=32, vocab=64)
    m = SegmentModel(cfg, {l: list(range(8)) for l in (1, 2, 3)})
    m.enable_profiling()
    m.forward("r0", embed_tokens(cfg, [1, 2, 3, 4]))

    assert sorted(m.profiler.mass) == [1, 2, 3]
    a, b = m.profiler.mass[1], m.profiler.mass[2]
    assert not np.allclose(a / a.sum(), b / b.sum()), "各层的路由分布不该一模一样"


def test_profiling_is_off_by_default() -> None:
    from p2pmoe.runtime.model import SegmentModel, ToyMoEConfig, embed_tokens

    cfg = ToyMoEConfig(n_layers=4, d_model=32, n_experts=8, d_ff=32, vocab=64)
    m = SegmentModel(cfg, {1: list(range(8))})
    m.forward("r0", embed_tokens(cfg, [1, 2]))
    assert m.profiler is None


def test_accumulates_across_calls() -> None:
    p = LayerProfiler(4)
    p.record(7, np.array([1.0, 0, 0, 0]), 2)
    p.record(7, np.array([0, 3.0, 0, 0]), 2)
    assert p.mass[7].tolist() == [1.0, 3.0, 0.0, 0.0]
    assert p.tokens[7] == 4


# --------------------------------------------------------------------------- #
# 2. 合并：同一个 task 的多条通道
# --------------------------------------------------------------------------- #
def test_records_from_several_channels_add_up() -> None:
    """同一个 task 有多条通道时各自采一份 —— 加起来样本更多，分布更稳。"""
    a = ActivationRecord("u", 4, mass={5: np.array([2.0, 0, 0, 0])}, tokens={5: 10})
    b = ActivationRecord("u", 4, mass={5: np.array([0, 2.0, 0, 0])}, tokens={5: 10})
    m = merge_records([a, b])
    assert m.mass[5].tolist() == [2.0, 2.0, 0.0, 0.0]
    assert m.tokens[5] == 20


def test_wire_round_trip() -> None:
    p = LayerProfiler(4)
    p.record(3, np.array([1.0, 2.0, 0.0, 1.0]), 4)
    rec = ActivationRecord("u", 4)
    rec.add_wire(p.to_wire())
    assert rec.mass[3].tolist() == [1.0, 2.0, 0.0, 1.0]
    assert rec.layers == [3]


def test_mass_is_normalised_on_save(tmp_path) -> None:
    """存归一化后的比例 —— 采多久跟分布长什么样是两件事。"""
    rec = ActivationRecord("u", 4, mass={2: np.array([3.0, 1.0, 0.0, 0.0])},
                           tokens={2: 4})
    raw = json.loads(save_profile(tmp_path / "p.json", {"u": rec},
                                  model="m", n_layers=8).read_text())
    assert raw["tasks"]["u"]["layers"]["2"] == [0.75, 0.25, 0.0, 0.0]
    assert raw["tasks"]["u"]["n_tokens"] == 4


# --------------------------------------------------------------------------- #
# 3. 用：画像 → 驻留集
# --------------------------------------------------------------------------- #
def prof(layers: dict[int, list[float]], task: str = "u") -> dict:
    n = len(next(iter(layers.values())))
    return {"n_experts": n, "n_layers": 8,
            "tasks": {task: {"n_tokens": 100,
                             "layers": {str(l): m for l, m in layers.items()}}}}


def test_takes_the_heaviest_until_coverage() -> None:
    raw = prof({3: [0.5, 0.3, 0.15, 0.05]})
    assert placement_from_profile(raw, "u", coverage=0.79, n_experts=4)[3] == (0, 1)
    assert placement_from_profile(raw, "u", coverage=0.81, n_experts=4)[3] == (0, 1, 2)


def test_sizes_are_heterogeneous_across_layers() -> None:
    """n_{u,l} 逐层异构不是巧合，是逐层独立取的直接结果（I.1.1）。"""
    raw = prof({3: [0.9, 0.05, 0.03, 0.02], 4: [0.3, 0.3, 0.2, 0.2]})
    got = placement_from_profile(raw, "u", coverage=0.9, n_experts=4)
    assert len(got[3]) < len(got[4])


def test_never_returns_fewer_than_top_k() -> None:
    """质量再集中也要装够 k 个 —— 否则每个 token 都必然 miss。"""
    raw = prof({3: [0.99, 0.005, 0.003, 0.002]})
    assert len(placement_from_profile(raw, "u", coverage=0.9, n_experts=4,
                                      min_experts=2)[3]) == 2


def test_only_the_requested_layers_come_back() -> None:
    raw = prof({3: [0.7, 0.3], 4: [0.7, 0.3], 5: [0.7, 0.3]})
    got = placement_from_profile(raw, "u", coverage=0.9, n_experts=2, layers=[4, 5])
    assert sorted(got) == [4, 5]


def test_expert_count_mismatch_is_caught() -> None:
    raw = prof({3: [0.5, 0.5]})
    with pytest.raises(ValueError, match="模型是 8 个"):
        placement_from_profile(raw, "u", coverage=0.9, n_experts=8)


def test_unknown_task_lists_what_exists() -> None:
    with pytest.raises(ValueError, match=r"\['u'\]"):
        placement_from_profile(prof({3: [1.0]}), "nope", coverage=0.9, n_experts=1)


# --------------------------------------------------------------------------- #
# 4. 两种文件格式
# --------------------------------------------------------------------------- #
def test_id_format_is_accepted_too(tmp_path) -> None:
    """别人拿别的工具统计出来的驻留集也该能喂进来。"""
    f = tmp_path / "ids.json"
    f.write_text(json.dumps({"u": [[0, 1], [2, 3]]}), encoding="utf-8")
    raw = load_profile(f)
    assert raw["format"] == "ids"
    got = placement_from_profile(raw, "u", coverage=0.9, n_experts=4, layers=[7, 8])
    assert got == {7: (0, 1), 8: (2, 3)}


def test_id_format_layer_count_must_match(tmp_path) -> None:
    f = tmp_path / "ids.json"
    f.write_text(json.dumps({"u": [[0, 1]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="要填 2 层"):
        placement_from_profile(load_profile(f), "u", coverage=0.9, n_experts=4,
                               layers=[7, 8])


def test_mass_format_is_detected_without_a_marker(tmp_path) -> None:
    f = save_profile(tmp_path / "p.json",
                     {"u": ActivationRecord("u", 2, mass={3: np.array([1.0, 1.0])},
                                            tokens={3: 2})})
    assert "format" not in load_profile(f)


# --------------------------------------------------------------------------- #
# 5. 转给规划器 / 给人看
# --------------------------------------------------------------------------- #
def test_unprofiled_layers_become_uniform() -> None:
    """没采到的层填均匀分布 —— 「没有信息」而不是「没有激活」。"""
    ap = to_activation_profile(prof({5: [0.7, 0.2, 0.1, 0.0]}), "u",
                               n_layers=8, n_experts=4)
    assert ap.at(5) == (0.7, 0.2, 0.1, 0.0)
    assert ap.at(1) == pytest.approx((0.25,) * 4)


def test_summary_reports_the_actual_saving() -> None:
    lines = summarise(prof({3: [0.9, 0.05, 0.03, 0.02], 4: [0.9, 0.05, 0.03, 0.02]}),
                      coverage=0.9, min_experts=1)
    assert len(lines) == 1 and "1.0/4" in lines[0] and "25%" in lines[0]
