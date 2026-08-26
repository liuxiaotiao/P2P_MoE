"""手动放置：布局文件 → 部署清单 + 连线表。

规划器整条链路在这条路上被跳过，所以**校验就是这里唯一的护栏**。
这一组测试大半是在测错误消息 —— 手动放置最容易错的几件事症状都很隐蔽：
层没接上会表现成「输出是乱的」（张量形状对得上、语义错了），
一台机器用两次会表现成「两条请求互相污染 KV」。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.manual import (
    ManualSpec,
    build_manual_manifest,
    memory_report,
    split_layers,
)
from p2pmoe.planner.types import ModelSpec

MODEL = ModelSpec(n_layers=8, d_model=64, n_experts=8, top_k=2,
                  base_gb_per_layer=0.01, expert_gb=0.001, ctx_max=128,
                  kv_bytes_per_elem=4)


def build(d: dict):
    return build_manual_manifest(ManualSpec.from_dict(d, n_layers=8), MODEL)


# --------------------------------------------------------------------------- #
# 1. 层怎么切
# --------------------------------------------------------------------------- #
def test_split_is_contiguous_and_covers_everything() -> None:
    out = split_layers(1, 8, 3)
    assert out[0][0] == 1 and out[-1][1] == 8
    assert all(b[0] == a[1] + 1 for a, b in zip(out, out[1:]))


def test_split_is_as_even_as_possible() -> None:
    sizes = [hi - lo + 1 for lo, hi in split_layers(1, 8, 3)]
    assert sizes == [3, 3, 2]           # 前面的多担一层


def test_more_machines_than_layers_is_refused() -> None:
    """每台至少得算一层 —— 空段在流水线里没有意义。"""
    with pytest.raises(ValueError, match="每台至少一层"):
        split_layers(1, 3, 5)


# --------------------------------------------------------------------------- #
# 2. 三种写法
# --------------------------------------------------------------------------- #
def test_shortest_form_splits_automatically() -> None:
    man, wired = build({"l0": 3, "channels": [{"front": "n1", "back": ["n2", "n3"]}]})
    assert wired == {"F0": ("Bgeneral0", "general")}
    assert man.segments["F0"]["splits"] == [[1, 3]]
    assert man.segments["Bgeneral0"]["splits"] == [[4, 6], [7, 8]]


def test_a_bare_string_means_one_machine() -> None:
    man, _ = build({"l0": 3, "channels": [{"front": "n1", "back": "n2"}]})
    assert man.segments["F0"]["nodes"] == ["n1"]


def test_explicit_layer_ranges_win() -> None:
    man, _ = build({"channels": [{
        "front": [{"node": "n1", "layers": [1, 5]}],
        "back": [{"node": "n2", "layers": [6, 7]}, {"node": "n3", "layers": [8, 8]}],
    }]})
    assert man.l0 == 5
    assert man.segments["Bgeneral0"]["splits"] == [[6, 7], [8, 8]]


def test_bare_names_without_l0_say_why() -> None:
    with pytest.raises(ValueError, match="得有 l0"):
        build({"channels": [{"front": "n1", "back": "n2"}]})


def test_mixing_the_two_forms_is_refused() -> None:
    with pytest.raises(ValueError, match="统一"):
        build({"l0": 3, "channels": [{"front": ["n1", {"node": "n2", "layers": [1, 2]}],
                                      "back": "n3"}]})


def test_task_labels_split_the_channels() -> None:
    _, wired = build({"l0": 3, "channels": [
        {"front": "n1", "back": "n2", "task": "code"},
        {"front": "n3", "back": "n4", "task": "chat"},
    ]})
    assert set(wired.values()) == {("Bcode0", "code"), ("Bchat1", "chat")}


# --------------------------------------------------------------------------- #
# 3. 校验 —— 这一节是重点
# --------------------------------------------------------------------------- #
def test_a_gap_between_front_and_back_is_caught() -> None:
    """中间漏层不会报错，只会让输出静默变成垃圾 —— 必须在下发前拦住。"""
    with pytest.raises(ValueError, match="没人算"):
        build({"channels": [{"front": [{"node": "n1", "layers": [1, 3]}],
                             "back": [{"node": "n2", "layers": [5, 8]}]}]})


def test_not_reaching_the_last_layer_is_caught() -> None:
    with pytest.raises(ValueError, match=r"最后 2 层没人算"):
        build({"channels": [{"front": [{"node": "n1", "layers": [1, 3]}],
                             "back": [{"node": "n2", "layers": [4, 6]}]}]})


def test_not_starting_at_layer_one_is_caught() -> None:
    with pytest.raises(ValueError, match="必须从第 1 层"):
        build({"channels": [{"front": [{"node": "n1", "layers": [2, 3]}],
                             "back": [{"node": "n2", "layers": [4, 8]}]}]})


def test_a_hole_inside_one_segment_is_caught() -> None:
    with pytest.raises(ValueError, match="层区间断了"):
        build({"channels": [{
            "front": [{"node": "n1", "layers": [1, 2]}, {"node": "n2", "layers": [4, 5]}],
            "back": [{"node": "n3", "layers": [6, 8]}]}]})


def test_a_machine_cannot_appear_twice() -> None:
    """一节点至多一条段（I.2.2）—— 否则两条请求会在它上面互相污染 KV。"""
    with pytest.raises(ValueError, match="至多承载一条段"):
        build({"l0": 3, "channels": [{"front": "n1", "back": "n2"},
                                     {"front": "n1", "back": "n3"}]})


def test_channels_must_share_one_l0() -> None:
    with pytest.raises(ValueError, match="L₀ 不一致"):
        build({"channels": [
            {"front": [{"node": "n1", "layers": [1, 3]}],
             "back": [{"node": "n2", "layers": [4, 8]}]},
            {"front": [{"node": "n3", "layers": [1, 5]}],
             "back": [{"node": "n4", "layers": [6, 8]}]}]})


def test_all_problems_are_reported_at_once() -> None:
    """一次报完，别让人改一条跑一次。"""
    with pytest.raises(ValueError) as e:
        build({"l0": 3, "channels": [{"front": "n1", "back": "n2"},
                                     {"front": "n1", "back": "n3"},
                                     {"front": "n4", "back": "n1"}]})
    assert str(e.value).count("至多承载一条段") >= 2


def test_expert_ids_are_range_checked() -> None:
    with pytest.raises(ValueError, match="越界"):
        build({"l0": 3, "channels": [{"front": "n1", "back": "n2"}],
               "experts": {"4": [0, 1, 99]}})


def test_no_channels_is_an_error() -> None:
    with pytest.raises(ValueError, match="没有 channels"):
        build({"l0": 3})


# --------------------------------------------------------------------------- #
# 4. 产出的清单
# --------------------------------------------------------------------------- #
def test_every_layer_is_assigned_exactly_once_per_channel() -> None:
    man, _ = build({"l0": 3, "channels": [{"front": "n1", "back": ["n2", "n3"]}]})
    got = sorted(l.layer for p in man.nodes for l in p.layers)
    assert got == list(range(1, 9))


def test_experts_default_to_all() -> None:
    man, _ = build({"l0": 3, "channels": [{"front": "n1", "back": "n2"}]})
    assert all(len(l.experts) == MODEL.n_experts for p in man.nodes for l in p.layers)


def test_a_per_layer_subset_is_applied_to_whoever_holds_that_layer() -> None:
    man, _ = build({"l0": 3, "channels": [{"front": "n1", "back": ["n2", "n3"]}],
                    "experts": {"7": [0, 3, 5]}})
    by_layer = {l.layer: l.experts for p in man.nodes for l in p.layers}
    assert by_layer[7] == (0, 3, 5)
    assert len(by_layer[6]) == MODEL.n_experts       # 没点名的层照旧全装


def test_head_and_tail_are_the_ends_of_the_chain() -> None:
    man, _ = build({"l0": 3, "channels": [{"front": "n1", "back": ["n2", "n3"]}]})
    b = man.segments["Bgeneral0"]
    assert (b["head"], b["tail"]) == ("n2", "n3")
    heads = {p.node for p in man.nodes if p.is_head}
    assert heads == {"n1", "n2"}


def test_manifest_round_trips() -> None:
    """产出的清单要能存下来再读回去 —— 与 control.py 的 --load-plan 同一条路。"""
    from p2pmoe.planner.manifest import DeploymentManifest

    man, _ = build({"l0": 3, "channels": [{"front": "n1", "back": ["n2", "n3"]}]})
    assert json.loads(DeploymentManifest.from_json(man.to_json()).to_json()) == \
        json.loads(man.to_json())


def test_wiring_is_what_you_wrote_not_what_was_computed() -> None:
    """连接不是算出来的 —— 第 i 条通道的前段就配第 i 条通道的后段。"""
    _, wired = build({"l0": 3, "channels": [
        {"front": "n1", "back": "n2"}, {"front": "n3", "back": "n4"}]})
    assert wired == {"F0": ("Bgeneral0", "general"), "F1": ("Bgeneral1", "general")}


# --------------------------------------------------------------------------- #
# 5. 内存核对
# --------------------------------------------------------------------------- #
def test_memory_report_sorts_the_tightest_first() -> None:
    spec = ManualSpec.from_dict(
        {"l0": 2, "channels": [{"front": "n1", "back": ["n2"]}]}, n_layers=8)
    rows = memory_report(spec, MODEL, {"n1": 100.0, "n2": 0.05})
    assert rows[0][0] == "n2"                # 6 层却只有一点点内存
    assert rows[0][1] > rows[0][2]           # 需要 > 可用


def test_unknown_machines_count_as_zero_available() -> None:
    """问不到内存的机器不该被当成「内存无限」而悄悄放过。"""
    spec = ManualSpec.from_dict(
        {"l0": 2, "channels": [{"front": "n1", "back": ["n2"]}]}, n_layers=8)
    assert dict((v, have) for v, _, have in memory_report(spec, MODEL, {}))["n1"] == 0.0


# --------------------------------------------------------------------------- #
# 6. 按激活画像装子集 —— 后段的驻留集必须真的按 task 收窄
# --------------------------------------------------------------------------- #
def _profile(tmp_path: Path, tasks: dict[str, dict[int, list[float]]]) -> Path:
    f = tmp_path / "prof.json"
    f.write_text(json.dumps({
        "model": "t", "n_layers": 8, "n_experts": 8,
        "tasks": {u: {"n_tokens": 100,
                      "layers": {str(l): m for l, m in rows.items()}}
                  for u, rows in tasks.items()},
    }), encoding="utf-8")
    return f


def _mass(*heavy: int) -> list[float]:
    """把质量堆在 heavy 那几个专家上，其余均分一点点。"""
    m = [0.01] * 8
    for e in heavy:
        m[e] = 0.9 / len(heavy)
    s = sum(m)
    return [x / s for x in m]


def test_profile_narrows_the_back_and_leaves_the_front_alone(tmp_path) -> None:
    """前段是 task 无关的（I.1.1）—— 画像只该收窄后段。"""
    f = _profile(tmp_path, {"general": {l: _mass(1, 5) for l in range(4, 9)}})
    spec = ManualSpec.from_dict(
        {"l0": 3, "channels": [{"front": "n1", "back": "n2"}]}, n_layers=8)
    spec.apply_profile(f, coverage=0.9, n_experts=8)
    man, _ = build_manual_manifest(spec, MODEL)

    front = [p for p in man.nodes if p.role == "front"][0]
    back = [p for p in man.nodes if p.role.startswith("back:")][0]
    assert all(len(l.experts) == 8 for l in front.layers)      # 前段照旧全装
    assert all(set(l.experts) == {1, 5} for l in back.layers)  # 后段只装热的那两个


def test_two_tasks_get_different_back_placements(tmp_path) -> None:
    """S_{u,l} 是逐 task 的 —— 两条不同 task 的通道装的东西必须不同。"""
    f = _profile(tmp_path, {"code": {l: _mass(0, 1) for l in range(4, 9)},
                            "chat": {l: _mass(6, 7) for l in range(4, 9)}})
    spec = ManualSpec.from_dict({"l0": 3, "channels": [
        {"front": "n1", "back": "n2", "task": "code"},
        {"front": "n3", "back": "n4", "task": "chat"},
    ]}, n_layers=8)
    spec.apply_profile(f, coverage=0.9, n_experts=8)
    man, _ = build_manual_manifest(spec, MODEL)

    by_node = {p.node: {e for l in p.layers for e in l.experts} for p in man.nodes}
    assert by_node["n2"] == {0, 1}
    assert by_node["n4"] == {6, 7}


def test_coverage_is_a_knob_not_baked_into_the_profile(tmp_path) -> None:
    """存的是质量分布而不是 id —— 换阈值不用重新采样。"""
    f = _profile(tmp_path, {"general": {4: [0.5, 0.25, 0.15, 0.05, 0.03, 0.01, 0.01, 0.0]}})
    sizes = []
    for cov in (0.5, 0.9, 0.99):
        spec = ManualSpec.from_dict(
            {"l0": 3, "channels": [{"front": "n1", "back": "n2"}]}, n_layers=8)
        spec.apply_profile(f, coverage=cov, n_experts=8)
        sizes.append(len(spec.channels[0].experts[4]))
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_layers_without_profile_data_stay_full(tmp_path) -> None:
    """画像里没有的层保持全装 —— 宁可多装，也不瞎选。"""
    f = _profile(tmp_path, {"general": {4: _mass(2)}})     # 只采了第 4 层
    spec = ManualSpec.from_dict(
        {"l0": 3, "channels": [{"front": "n1", "back": "n2"}]}, n_layers=8)
    spec.apply_profile(f, coverage=0.9, n_experts=8, top_k=MODEL.top_k)
    man, _ = build_manual_manifest(spec, MODEL)
    by_layer = {l.layer: l.experts for p in man.nodes for l in p.layers}
    assert 2 in by_layer[4] and len(by_layer[4]) == MODEL.top_k   # 补到 top_k
    assert len(by_layer[5]) == 8


def test_memory_drops_when_the_back_loads_a_subset(tmp_path) -> None:
    """驻留集收窄要真的省下内存 —— 不然这套机制就没有意义。"""
    f = _profile(tmp_path, {"general": {l: _mass(1, 5) for l in range(4, 9)}})
    layout = {"l0": 3, "channels": [{"front": "n1", "back": "n2"}]}
    full = ManualSpec.from_dict(layout, n_layers=8)
    thin = ManualSpec.from_dict(layout, n_layers=8)
    thin.apply_profile(f, coverage=0.9, n_experts=8)
    need = lambda sp: dict((v, g) for v, g, _ in memory_report(sp, MODEL, {}))["n2"]
    assert need(thin) < need(full)


def test_coverage_never_selects_fewer_than_top_k(tmp_path) -> None:
    """质量高度集中时覆盖率规则会只选一个 —— 但 top-k 路由每 token 要 k 个。"""
    f = _profile(tmp_path, {"general": {4: [0.99] + [0.001] * 7}})
    spec = ManualSpec.from_dict(
        {"l0": 3, "channels": [{"front": "n1", "back": "n2"}]}, n_layers=8)
    spec.apply_profile(f, coverage=0.9, n_experts=8, top_k=MODEL.top_k)
    assert len(spec.channels[0].experts[4]) == MODEL.top_k


def test_a_subset_smaller_than_top_k_is_refused(tmp_path) -> None:
    """驻留数 < top_k 时每个 token 必 miss，drop-expert 兜不住这种程度。"""
    with pytest.raises(ValueError, match="top-k"):
        build({"l0": 3, "channels": [{"front": "n1", "back": "n2"}],
               "experts": {"4": [3]}})       # MODEL 的 top_k 是 2


def test_a_missing_task_in_the_profile_says_what_is_there(tmp_path) -> None:
    f = _profile(tmp_path, {"code": {4: _mass(1)}})
    spec = ManualSpec.from_dict(
        {"l0": 3, "channels": [{"front": "n1", "back": "n2", "task": "chat"}]},
        n_layers=8)
    with pytest.raises(ValueError, match=r"\['code'\]"):
        spec.apply_profile(f, coverage=0.9, n_experts=8)
