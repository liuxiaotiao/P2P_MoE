"""远端命令必须 `cd` 到代码目录。

`--workdir` 默认是 `"."`，而远端的 `cd .` 落在 **ssh 登录目录**（通常 $HOME），
不是代码目录。症状是 15 台齐刷刷报 `No module named 'p2pmoe'` —— 看起来像
没装依赖，其实是跑在了错的目录里，而这两件事的修法完全不同。

真踩过：`start` 传了 `--workdir`，`fetch` 漏了。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import p2pmoe.deploy.launch as L

ROOT = Path(__file__).resolve().parent.parent
HOSTS = "n1  127.0.0.1:9101\nn2  127.0.0.1:9102\n"


@pytest.fixture
def hosts(tmp_path) -> Path:
    p = tmp_path / "hosts.txt"
    p.write_text(HOSTS, encoding="utf-8")
    return p


def _sent(monkeypatch, argv: list[str]) -> list[str]:
    """截下真正要发出去的 ssh 命令。"""
    out: list[str] = []

    def stub(h, cmd, **kw):
        out.append(cmd)
        return False, "stub"

    monkeypatch.setattr(L, "_ssh", stub)
    try:
        L.main(argv)
    except SystemExit:
        pass
    return out


def test_fetch_cds_into_the_workdir(monkeypatch, hosts) -> None:
    sent = _sent(monkeypatch, [
        "fetch", "--hosts", str(hosts), "--plan", str(ROOT / "task/plan_deploy.json"),
        "--repo", "R", "--out", "/w", "--workdir", "/opt/code"])
    assert sent, "一条都没发出去"
    assert all(c.startswith("cd /opt/code && ") for c in sent), sent[0]


def test_start_cds_into_the_workdir(monkeypatch, hosts) -> None:
    sent = _sent(monkeypatch, ["start", "--hosts", str(hosts), "--workdir", "/opt/code"])
    assert sent
    assert all("cd /opt/code" in c for c in sent), sent[0]


def test_a_workdir_with_spaces_is_quoted(monkeypatch, hosts) -> None:
    """没引号的话 `cd /opt/my code` 会被拆成两个参数，cd 到 /opt/my。"""
    sent = _sent(monkeypatch, [
        "fetch", "--hosts", str(hosts), "--plan", str(ROOT / "task/plan_deploy.json"),
        "--repo", "R", "--out", "/w", "--workdir", "/opt/my code"])
    assert "'/opt/my code'" in sent[0], sent[0]


def test_the_default_workdir_is_called_out(monkeypatch, hosts, capsys) -> None:
    """默认值本身没错（本机模式要它），错的是**没人告诉你远端 `.` 是哪儿**。"""
    _sent(monkeypatch, [
        "fetch", "--hosts", str(hosts), "--plan", str(ROOT / "task/plan_deploy.json"),
        "--repo", "R", "--out", "/w"])
    assert "ssh 登录目录" in capsys.readouterr().out


def test_local_mode_is_not_warned_about(monkeypatch, hosts, capsys) -> None:
    """`--local` 下 `.` 就是当前工作目录，没有歧义 —— 别制造假警报。"""
    monkeypatch.setattr(L, "_local_start", lambda *a, **k: (True, "pid 1"))
    try:
        L.main(["start", "--hosts", str(hosts), "--local"])
    except SystemExit:
        pass
    assert "ssh 登录目录" not in capsys.readouterr().out


def test_dry_run_shows_the_command_that_will_actually_run(hosts, capsys) -> None:
    """**这条是这个 bug 藏得住的原因。**

    dry-run 少打了 `cd`，于是 workdir 配错时它看起来完全正常，
    直到真跑才报错。所见即所跑，否则 dry-run 是在骗人。
    """
    try:
        L.main(["fetch", "--hosts", str(hosts),
                "--plan", str(ROOT / "task/plan_deploy.json"),
                "--repo", "R", "--out", "/w", "--workdir", "/opt/code", "--dry-run"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "cd /opt/code &&" in out, "dry-run 没显示 cd —— 所见非所跑"


def test_the_script_passes_workdir_to_fetch_too() -> None:
    """start 传了、fetch 漏了 —— 就是这次的 bug。盯住两处都在。"""
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    launches = re.findall(r'p2pmoe\.deploy\.launch (start|fetch)(.*?)(?=\n\s*(?:read|\$PY|\}|#|$))',
                          src, re.S)
    assert launches, "找不到 launch 调用"
    for action, tail in launches:
        assert "--workdir" in tail, f"launch {action} 没传 --workdir"
