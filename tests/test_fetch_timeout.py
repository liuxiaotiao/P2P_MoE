"""下载不能用控制命令那套 60 秒的超时。

单台最多拉 22GB。`_ssh` 原来硬编码 `timeout=60`，于是 **fetch 永远不可能成功**：
下得快的节点在 60 秒内没下完就被杀，而 `TimeoutExpired` 的消息是一整串 argv，
读起来完全不像「超时了」—— 看起来像 ssh 命令本身出了问题。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import p2pmoe.deploy.launch as L

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def hosts(tmp_path) -> Path:
    p = tmp_path / "hosts.txt"
    p.write_text("n1  127.0.0.1:9101\n", encoding="utf-8")
    return p


def test_fetch_gets_hours_not_a_minute(monkeypatch, hosts) -> None:
    seen: list[float] = []

    def stub(h, cmd, *, ssh, user, timeout=60.0):
        seen.append(timeout)
        return True, "ok"

    monkeypatch.setattr(L, "_ssh", stub)
    try:
        L.main(["fetch", "--hosts", str(hosts),
                "--plan", str(ROOT / "task/plan_deploy.json"),
                "--repo", "R", "--out", "/w", "--workdir", "/c"])
    except SystemExit:
        pass
    assert seen, "没发出去"
    assert seen[0] >= 3600, f"fetch 的超时只有 {seen[0]}s —— 22GB 下不完"


def test_control_commands_keep_the_short_timeout(monkeypatch, hosts) -> None:
    """start/stop 卡住时要快速失败，不能也等 4 小时。"""
    seen: list[float] = []

    def stub(h, cmd, *, ssh, user, timeout=60.0):
        seen.append(timeout)
        return True, "pid 1"

    monkeypatch.setattr(L, "_ssh", stub)
    try:
        L.main(["stop", "--hosts", str(hosts)])
    except SystemExit:
        pass
    assert seen and seen[0] <= 120, f"控制命令等了 {seen[0]}s"


def test_a_timeout_says_it_timed_out(monkeypatch) -> None:
    """`TimeoutExpired` 的原文里是一整串 argv —— 那读起来像 ssh 坏了，
    而不是像「给的时间不够」。两者的修法完全不同。"""
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["ssh", "-o", "X", "host", "cmd"],
                                        timeout=60)

    monkeypatch.setattr(L.subprocess, "run", boom)
    ok, msg = L._ssh(L.Host("n1", "h", 9101), "x", ssh="ssh", user=None, timeout=60)
    assert not ok
    assert "超过" in msg and "--timeout" in msg
    assert "argv" not in msg and "'-o'" not in msg


# --------------------------------------------------------------------------- #
# 报错摘要要保留头部
# --------------------------------------------------------------------------- #
def test_the_summary_keeps_the_head_not_the_tail() -> None:
    """失败原因几乎总在第一行。留尾巴会把「读不到 https://…」切成
    「ingface.co/…」—— 读的人看到半截 URL，看不出这是个网络错误。"""
    msg = ("读不到 https://huggingface.co/Qwen/X/resolve/main 的 config.json：SSL EOF\n"
           "  · 私有仓库要 HF_TOKEN\n"
           "  · 用 --src 从本地取")
    out = L._brief(msg)
    assert out.startswith("读不到 https://huggingface.co")
    assert "+2 行" in out


def test_a_single_line_is_passed_through() -> None:
    assert L._brief("就一行") == "就一行"


def test_empty_output_says_so() -> None:
    """空输出不该显示成空白 —— 那看起来像成功了。"""
    assert "没有输出" in L._brief("   \n \n")


def test_a_very_long_first_line_is_cut_at_the_end() -> None:
    out = L._brief("x" * 500)
    assert len(out) <= 200 and out.endswith("…")
