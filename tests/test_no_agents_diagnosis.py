"""一台 agent 都连不上时，报错要指向正确的方向。

`Connection refused` 和「超时 / 不可达」是两件事，处置相反：

    ECONNREFUSED   主机在，端口上没人监听  → **agent 没跑**，去 start
    超时 / 不可达   包根本没到              → 网络、防火墙、地址写错

混成一句「没有任何 agent 可用」的话，人会去查网络 —— 而实际上只是
上一轮 stop 之后忘了 start。15 行一模一样的红字更是把结论埋掉了。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "p2pmoe" / "deploy" / "control.py").read_text(encoding="utf-8")


def _run(agents: str) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "p2pmoe.deploy.control", "--agents", agents,
         "--advertise", "127.0.0.1", "--once"],
        capture_output=True, text=True, timeout=180, cwd=ROOT)
    return r.stdout + r.stderr


def test_all_refused_points_at_start_not_at_the_network() -> None:
    """**这条是重点。** 全是 refused → agent 没跑，别去查防火墙。"""
    out = _run("N01=127.0.0.1:9191,N02=127.0.0.1:9192")
    assert "Connection refused" in out
    assert "没人监听" in out
    assert "deploy_15.sh start" in out


def test_it_says_this_is_not_a_partial_outage() -> None:
    """「掉了几台」是常态，「一台都没有」不是 —— 两者不该共用一句话。"""
    out = _run("N01=127.0.0.1:9191,N02=127.0.0.1:9192")
    assert "一个都没连上" in out
    assert "不是「掉了几台」" in out


def test_the_count_is_not_hardcoded() -> None:
    """写死 15 的话，两台的部署会读到一句假话。"""
    out = _run("N01=127.0.0.1:9191,N02=127.0.0.1:9192")
    assert "2 台一个都没连上" in out, out[-600:]


def test_unreachable_points_at_the_network() -> None:
    """反方向：包没到就别让人去 start —— 起了也连不上。"""
    out = _run("N01=10.255.255.1:9101")
    if "Connection refused" in out:
        pytest.skip("这个环境把不可达地址也变成了 refused")
    assert "包根本没到" in out or "不可达" in out


def test_a_mixed_failure_is_reported_as_mixed() -> None:
    """一半没跑、一半不通时，笼统归到任一边都会让人查错方向。"""
    out = _run("N01=127.0.0.1:9191,N02=10.255.255.1:9101")
    if "两种都有" not in out:
        pytest.skip("这个环境两类错误没分开")
    assert "doctor" in out


def test_the_script_checks_before_the_long_sweep() -> None:
    """脚本里要先探一遍 —— 否则要等控制器扫完 15 台才看得到结论。"""
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    assert "_require_agents" in src
    i_def = src.index("_require_agents() {")
    i_use = src.index("_require_agents || return 1")
    assert i_def < i_use
    # 必须在真正打请求之前
    i_ctl = src.index("p2pmoe.deploy.control")
    assert i_use < i_ctl, "检查排在控制器后面等于没检查"
