"""人工指定前后段连接：清单存取、`--wiring` 解析、task 解析、checkpoint 预检。

这一组的共同主题是**可重放**：规划的输入里有一项不可复现（逐对延迟实测），
所以「我要 F0 连 BX1」这种指定必须先把放置固定住才有稳定的所指。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.control import (
    check_model_dirs,
    parse_tasks,
    resolve_wiring,
    wiring_to_json,
)
from p2pmoe.planner.manifest import DeploymentManifest


# --------------------------------------------------------------------------- #
def _plan_dict() -> dict:
    """2 条前段（n1、n2）+ 2 条后段（X:n3、Y:n4），层区间 1–2 / 3–4。"""
    def node(v, role, seg, lo, hi):
        return {
            "node": v, "role": role, "segment": seg, "position": 0,
            "is_head": True, "is_tail": True, "layer_range": [lo, hi],
            "weight_gb": 0.2, "kv_gb": 0.02, "total_gb": 0.22,
            "layers": [{"layer": l, "experts": [0, 1], "weight_gb": 0.1,
                        "kv_gb": 0.01} for l in range(lo, hi + 1)],
        }

    def seg(role, task, v, lo, hi):
        return {"role": role, "task": task, "nodes": [v], "splits": [[lo, hi]],
                "head": v, "tail": v, "hops": 0, "compute_ms": 1.0,
                "hop_ms": 0.0, "delay_ms": 1.0}

    return {
        "model": "t", "l0": 2, "band": {"w_lo": 10.0, "w_hi": 20.0},
        "nodes": [node("n1", "front", "F0", 1, 2), node("n2", "front", "F1", 1, 2),
                  node("n3", "back:X", "BX0", 3, 4), node("n4", "back:Y", "BY0", 3, 4)],
        "segments": {"F0": seg("front", None, "n1", 1, 2),
                     "F1": seg("front", None, "n2", 1, 2),
                     "BX0": seg("back:X", "X", "n3", 3, 4),
                     "BY0": seg("back:Y", "Y", "n4", 3, 4)},
        "pairings": [
            {"front": f, "back": b, "task": u, "forward": [fv, bv], "loop": [bv, fv],
             "w_p50": 10.0, "w_p95": 12.0, "d_loop_p50": 10.0, "t50": 22.0}
            for f, fv in (("F0", "n1"), ("F1", "n2"))
            for b, bv, u in (("BX0", "n3", "X"), ("BY0", "n4", "Y"))
        ],
        "standby_fronts": [], "violations": [],
    }


@pytest.fixture
def man() -> DeploymentManifest:
    return DeploymentManifest.from_dict(_plan_dict())


def write(tmp_path: Path, pairs: list[dict]) -> Path:
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")
    return f


# --------------------------------------------------------------------------- #
# 1. 清单存取 —— 放置能被原样固定住，才谈得上指定连接
# --------------------------------------------------------------------------- #
def test_manifest_round_trips_exactly() -> None:
    d = _plan_dict()
    assert json.loads(DeploymentManifest.from_dict(d).to_json()) == d


def test_round_trip_preserves_the_memory_ledger() -> None:
    """逐节点的 total_gb 是由逐层数据算出来的 —— 少存一项 KV 就对不上账。"""
    m = DeploymentManifest.from_dict(_plan_dict())
    p = m.plan_for("n1")
    assert p.total_gb == pytest.approx(0.22)
    assert p.kv_gb == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# 2. --wiring 解析
# --------------------------------------------------------------------------- #
def test_pin_by_segment_id(man, tmp_path) -> None:
    w = resolve_wiring(write(tmp_path, [{"front": "F1", "back": "BX0"}]), man, ["F0", "F1"])
    assert w == {"F1": ("BX0", "X")}


def test_pin_by_node_id(man, tmp_path) -> None:
    """节点 id 更实用：段 id 是每次规划现编的，机器名是你自己写在 hosts.txt 里的。"""
    w = resolve_wiring(write(tmp_path, [{"front": "n2", "back": "n3"}]), man, ["F0", "F1"])
    assert w == {"F1": ("BX0", "X")}


def test_task_is_taken_from_the_back_segment(man, tmp_path) -> None:
    w = resolve_wiring(write(tmp_path, [{"front": "F0", "back": "BY0"}]), man, ["F0", "F1"])
    assert w["F0"] == ("BY0", "Y")


def test_declaring_the_wrong_task_is_caught(man, tmp_path) -> None:
    """后段的 task 由它装了哪些专家决定，不是配置项 —— 写错要当场报。"""
    f = write(tmp_path, [{"front": "F0", "back": "BY0", "task": "X"}])
    with pytest.raises(SystemExit, match="task"):
        resolve_wiring(f, man, ["F0", "F1"])


def test_unknown_name_lists_what_is_available(man, tmp_path) -> None:
    f = write(tmp_path, [{"front": "F9", "back": "BX0"}])
    with pytest.raises(SystemExit, match="save-wiring"):
        resolve_wiring(f, man, ["F0", "F1"])


def test_roles_must_match(man, tmp_path) -> None:
    f = write(tmp_path, [{"front": "BX0", "back": "F0"}])
    with pytest.raises(SystemExit, match="不是前段"):
        resolve_wiring(f, man, ["F0", "F1"])


def test_one_front_cannot_serve_two_backs(man, tmp_path) -> None:
    """一条前段同时只服务一条请求（I.2.4 排他独占），一对多是配置错误。"""
    f = write(tmp_path, [{"front": "F0", "back": "BX0"}, {"front": "F0", "back": "BY0"}])
    with pytest.raises(SystemExit, match="被指了两次"):
        resolve_wiring(f, man, ["F0", "F1"])


def test_one_back_cannot_be_shared(man, tmp_path) -> None:
    f = write(tmp_path, [{"front": "F0", "back": "BX0"}, {"front": "F1", "back": "BX0"}])
    with pytest.raises(SystemExit, match="被指了两次"):
        resolve_wiring(f, man, ["F0", "F1"])


def test_export_can_be_fed_back_in(man, tmp_path) -> None:
    """--save-wiring 导出的东西，--wiring 必须原样吃得回去。"""
    wired = {"F0": ("BX0", "X"), "F1": ("BY0", "Y")}
    f = tmp_path / "out.json"
    f.write_text(wiring_to_json(wired, man), encoding="utf-8")
    assert resolve_wiring(f, man, ["F0", "F1"]) == wired


# --------------------------------------------------------------------------- #
# 3. --tasks
# --------------------------------------------------------------------------- #
def test_single_task() -> None:
    assert parse_tasks("general") == [("general", 1.0)]


def test_weights_are_normalised() -> None:
    out = dict(parse_tasks("code=3,chat=1"))
    assert out == {"code": 0.75, "chat": 0.25}


def test_bare_names_split_evenly() -> None:
    assert dict(parse_tasks("a,b,c")) == {"a": pytest.approx(1 / 3)} | {
        "b": pytest.approx(1 / 3), "c": pytest.approx(1 / 3)}


# --------------------------------------------------------------------------- #
# 4. checkpoint 预检 —— 真机上最常见的翻车点
# --------------------------------------------------------------------------- #
def _server(tmp_path):
    """一个未配置的 NodeServer，只用来问 check_model。"""
    from p2pmoe.runtime.node import NodeServer

    n = object.__new__(NodeServer)
    n.me = "n1"
    return n


def test_missing_dir_is_reported(tmp_path) -> None:
    r = _server(tmp_path).check_model(str(tmp_path / "nope"))
    assert not r["ok"] and "不存在" in r["why"]


def test_missing_config_is_reported(tmp_path) -> None:
    r = _server(tmp_path).check_model(str(tmp_path))
    assert not r["ok"] and "config.json" in r["why"]


def test_pickle_checkpoint_is_rejected_early(tmp_path) -> None:
    """.bin 做不到「只读部分张量」—— 那是整套方案的前提，要在下发前就拦住。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    r = _server(tmp_path).check_model(str(tmp_path))
    assert not r["ok"] and "safetensors" in r["why"]


def test_a_complete_checkpoint_passes(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x" * 16)
    r = _server(tmp_path).check_model(str(tmp_path))
    assert r["ok"] and r["shards"] == 1


def test_unreachable_agents_are_named_not_swallowed() -> None:
    """预检问不到的节点也要点名 —— 它跟「读不到权重」一样会挡住部署。"""
    bad = check_model_dirs({"dead": ("127.0.0.1", 1)}, "/whatever")
    assert len(bad) == 1 and bad[0].startswith("dead:")
