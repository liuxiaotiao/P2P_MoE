"""`deploy_15.sh` 传给 `launch` 的每一个参数，argparse 都得认。

这一类错很贵：脚本前面的检查全过了、y 也按了、ssh 也连上了，然后在最后一步
被 `unrecognized arguments: --transport auto` 顶回来。而它本可以在写代码的那一刻
就被发现 —— 两侧的参数表是同一份契约，只是分散在两个文件里。

真踩过：`--transport` 加在了 fetch.py 上，忘了 launch.py 也要透传；
同一次还发现 `--base-url` / `--src` 两条路根本没接上。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy_15.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="没有 deploy_15.sh")


def _help_flags(module: str, argv: list[str]) -> set[str]:
    """从 `--help` 的输出里把长选项抠出来。

    不猴补 argparse —— 它内部自己也会造解析器与参数组，替换类会递归。
    `--help` 是同一份定义的权威渲染，读它最稳。
    """
    r = subprocess.run([sys.executable, "-m", module, *argv, "--help"],
                       capture_output=True, text=True, timeout=60, cwd=ROOT)
    text = r.stdout + r.stderr
    assert text.strip(), f"{module} --help 没有输出"
    return set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", text))


def _launch_flags() -> set[str]:
    # 子命令是位置参数，任选一个都能渲染出全部选项
    return _help_flags("p2pmoe.deploy.launch", ["fetch"])


def _fetch_flags() -> set[str]:
    return _help_flags("p2pmoe.deploy.fetch", [])


def _flags_after(marker: str) -> set[str]:
    """抠出脚本里某个模块调用之后跟着的长选项。

    要在**下一个模块调用处截断** —— `cmds` / `bootstrap` 打印的那一长串里，
    `fetch` 后面用 `&&` 紧跟着 `agent`，不截断的话会把 `--id` `--bind`
    也算成 fetch 的参数。
    """
    src = SCRIPT.read_text(encoding="utf-8")
    found: set[str] = set()
    for m in re.finditer(re.escape(marker), src):
        tail: list[str] = []
        for line in src[m.end():].splitlines():
            nxt = re.search(r"p2pmoe\.deploy\.\w+", line)
            if nxt:
                tail.append(line[:nxt.start()])
                break
            tail.append(line)
            if not line.rstrip().rstrip("\\").rstrip().endswith("\\") \
                    and not line.rstrip().endswith("\\"):
                break
        found |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", " ".join(tail)))
    return found


def test_every_launch_flag_exists() -> None:
    used = _flags_after("p2pmoe.deploy.launch fetch") \
        | _flags_after("p2pmoe.deploy.launch start") \
        | _flags_after("p2pmoe.deploy.launch stop") \
        | _flags_after("p2pmoe.deploy.launch status") \
        | _flags_after("p2pmoe.deploy.launch probe")
    assert used, "没从脚本里抠到任何参数 —— 匹配规则失效了？"
    missing = used - _launch_flags()
    assert not missing, f"deploy_15.sh 传了 launch 不认识的参数：{sorted(missing)}"


def test_every_fetch_flag_exists() -> None:
    used = _flags_after("p2pmoe.deploy.fetch")
    assert used
    missing = used - _fetch_flags()
    assert not missing, f"deploy_15.sh 传了 fetch 不认识的参数：{sorted(missing)}"


def test_launch_can_forward_all_three_weight_sources() -> None:
    """`SRC_DIR` / `SRC_URL` / `--repo` 三条路都要能透传到各节点。

    少一条的症状是「换个源就崩」，而换源恰恰是上游拉不动时的第一反应。
    """
    known = _launch_flags()
    for flag in ("--src", "--base-url", "--repo", "--endpoint", "--transport"):
        assert flag in known, f"launch 不能透传 {flag}"


@pytest.mark.parametrize("source", [
    ["--repo", "Q/X", "--endpoint", "https://mirror", "--transport", "curl"],
    ["--base-url", "http://10.0.0.9:9400"],
    ["--src", "/mnt/shared/ckpt"],
])
def test_each_source_produces_a_sane_remote_command(source, capsys, tmp_path) -> None:
    """dry-run 打出来的就是真正会跑的那条 —— 逐条源验一遍。"""
    import p2pmoe.deploy.launch as L

    hosts = tmp_path / "h.txt"
    hosts.write_text("N01  10.0.0.1:9101\n", encoding="utf-8")
    try:
        L.main(["fetch", "--hosts", str(hosts),
                "--plan", str(ROOT / "task/plan_deploy.json"),
                "--out", "/w", "--workdir", "/c", "--dry-run", *source])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "cd /c &&" in out
    assert "p2pmoe.deploy.fetch" in out
    assert source[0] in out, f"{source[0]} 没进到远端命令里"
