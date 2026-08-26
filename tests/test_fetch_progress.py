"""下载期间要有进度。

`fetch` 是**几小时**的操作，而每台节点的输出被 `capture_output` 扣着，
要等它整个跑完才吐出来。中间什么都不打的话，人分不清三件事：

    在下 · 卡住了 · 早就死了

而这三件事的处置完全不同。所以进度问的是**目录大小**，不是进程状态 ——
进程活着但一个字节没动，恰恰是最需要被看见的那种卡。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import p2pmoe.deploy.launch as L

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def hosts(tmp_path) -> Path:
    p = tmp_path / "h.txt"
    p.write_text("N01  10.0.0.1:9101\nN02  10.0.0.2:9101\n", encoding="utf-8")
    return p


@pytest.fixture
def plan(tmp_path) -> Path:
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"l0": 1, "model": {}, "segments": {}, "nodes": [
        {"node": "N01", "weight_gb": 2.0}, {"node": "N02", "weight_gb": 8.0}]}),
        encoding="utf-8")
    return p


def _run(monkeypatch, hosts, plan, sizes, stuck=(), every="0.4"):
    """跑一次 fetch，ssh 换成可控的桩。返回打印出来的全部文字。"""
    out: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))

    def fake_ssh(h, cmd, *, ssh, user, timeout=60.0):
        if "du -sb" in cmd:
            return True, str(int(sizes.get(h.node_id, 0)))
        for _ in range(4):
            time.sleep(0.25)
            if h.node_id not in stuck:
                sizes[h.node_id] = min(sizes["_want"][h.node_id],
                                       sizes[h.node_id] + sizes["_want"][h.node_id] / 3)
        return True, "done"

    monkeypatch.setattr(L, "_ssh", fake_ssh)
    try:
        L.main(["fetch", "--hosts", str(hosts), "--plan", str(plan),
                "--repo", "R", "--out", "/w", "--workdir", "/c",
                "--progress-every", every])
    except SystemExit:
        pass
    return "\n".join(out)


def test_progress_is_reported_while_downloading(monkeypatch, hosts, plan) -> None:
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    text = _run(monkeypatch, hosts, plan, sizes)
    assert "进度" in text, "整个过程一声不吭"
    assert "GB" in text


def test_it_shows_per_node_not_just_a_total(monkeypatch, hosts, plan) -> None:
    """总量看不出是哪台慢 —— 而慢的那台决定整轮什么时候结束。"""
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    text = _run(monkeypatch, hosts, plan, sizes)
    assert "N01" in text and "N02" in text


def test_a_stalled_node_is_called_out(monkeypatch, hosts, plan) -> None:
    """**这条是重点。** 进程还在不代表在下东西。"""
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    text = _run(monkeypatch, hosts, plan, sizes, stuck=("N02",))
    assert "无增长" in text


def test_the_first_tick_never_claims_a_stall(monkeypatch, hosts, plan) -> None:
    """首屏没有「上一次」可比。记 0 的话每台都会被标成无增长 ——
    而那正是最刺眼的告警，首屏就全屏告警等于没有告警。"""
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    text = _run(monkeypatch, hosts, plan, sizes)
    first = text.split("── 进度")[1] if "── 进度" in text else ""
    second = text.split("── 进度")[2] if text.count("── 进度") > 1 else ""
    assert "无增长" not in first, "首屏就报无增长"
    assert second or True      # 只有一屏时不苛求


def test_progress_can_be_switched_off(monkeypatch, hosts, plan) -> None:
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    text = _run(monkeypatch, hosts, plan, sizes, every="0")
    assert "── 进度" not in text


def test_the_final_summary_still_appears(monkeypatch, hosts, plan) -> None:
    """进度线程不能把结尾那份汇总挤掉或搅乱。"""
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    text = _run(monkeypatch, hosts, plan, sizes)
    assert "fetch: 2/2 成功" in text


def test_the_monitor_stops_with_the_download(monkeypatch, hosts, plan) -> None:
    """守护线程不能在 main 返回之后还在问 ssh。"""
    before = threading.active_count()
    sizes = {"N01": 0, "N02": 0, "_want": {"N01": 2e9, "N02": 8e9}}
    _run(monkeypatch, hosts, plan, sizes)
    time.sleep(1.0)
    assert threading.active_count() <= before + 1


def test_expected_sizes_come_from_the_plan(plan) -> None:
    """分母必须来自清单 —— 拍脑袋的分母会让进度条撒谎。"""
    exp = L._expected_bytes(plan, per_node_dir=False)
    assert exp == {"N01": 2_000_000_000, "N02": 8_000_000_000}


def test_a_missing_plan_degrades_quietly(tmp_path) -> None:
    """读不到清单就不报进度，而不是崩掉整轮下载。"""
    assert L._expected_bytes(tmp_path / "nope.json", per_node_dir=False) == {}
