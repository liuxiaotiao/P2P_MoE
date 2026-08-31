"""`stop` 必须真的停掉 agent —— 而不是停掉自己。

**真踩过。** 原来是 `pkill -f 'p2pmoe.deploy.agent --id N01'`。ssh 远端执行的是
`bash -c '<整条命令>'`，而这条命令里就写着那个模式 —— 于是 `pkill -f` 第一个
匹配到的是**执行它的这个 shell 自己**，把自己杀掉，后面一句都跑不到。

症状极其安静：`stop` 报成功，端口却还占着，下一轮 `start` 报
`Address already in use`，而 `start` 只报 fork 成功，要翻日志才看得见。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import p2pmoe.deploy.launch as L

ROOT = Path(__file__).resolve().parent.parent


def _cmd(node="N01", port=9101, by_port=False) -> str:
    return L._stop_cmd(L.Host(node, "127.0.0.1", port), by_port=by_port)


# --------------------------------------------------------------------------- #
# 1. 不能用裸 pkill -f
# --------------------------------------------------------------------------- #
def test_it_does_not_use_bare_pkill_f() -> None:
    """`pkill -f <pat>` 会杀掉命令行里含 <pat> 的那个 shell —— 也就是它自己。"""
    assert "pkill -f" not in _cmd(), "又用回 pkill -f 了"


def test_it_filters_by_process_name() -> None:
    """agent 是 python，包裹它的是 bash。按 comm 过滤一刀切干净，
    比去猜 $$ / $PPID 有几层可靠。"""
    c = _cmd()
    assert "pgrep -f" in c
    assert "comm=" in c
    assert "python" in c


def test_the_node_id_is_matched_exactly() -> None:
    """同机上可能跑着别的节点 —— N1 的模式不能误伤 N10。"""
    c = _cmd("N01")
    assert "--id N01" in c


# --------------------------------------------------------------------------- #
# 2. 真跑一次：假 agent 死、bash 活
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(sys.platform.startswith("win"), reason="需要 POSIX")
def test_a_python_agent_dies_and_the_wrapping_shell_survives(tmp_path) -> None:
    """**这个文件存在的理由。**

    构造一个命令行长得像 agent 的 python 进程，再用一层命令行同样含该模式的
    bash 去执行清理 —— python 该死，bash 该活。
    """
    marker = f"p2pmoe.deploy.agent --id TESTNODE"
    victim = tmp_path / "fake_agent.py"
    victim.write_text(
        "import sys, time\n"
        "open(sys.argv[-1], 'w').write(str(__import__('os').getpid()))\n"
        "time.sleep(60)\n", encoding="utf-8")
    pidf = tmp_path / "pid"

    proc = subprocess.Popen(
        [sys.executable, str(victim), *marker.split(), str(pidf)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if pidf.exists():
            break
        time.sleep(0.1)
    assert pidf.exists(), "假 agent 没起来"

    cmd = L._stop_cmd(L.Host("TESTNODE", "127.0.0.1", 9101))
    # 外面这层 bash 的命令行里也含那个模式 —— 正是原来会自杀的情形
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr

    for _ in range(50):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    assert proc.poll() is not None, "假 agent 没被杀掉"
    proc.wait(timeout=5)


# --------------------------------------------------------------------------- #
# 3. --by-port：端口那条兜底
# --------------------------------------------------------------------------- #
def test_by_port_tries_three_lookups() -> None:
    """没有哪一条到处都有：ss 容器里常缺，fuser 属于 psmisc，lsof 也不总在。"""
    c = _cmd(by_port=True)
    for tool in ("ss ", "fuser ", "lsof "):
        assert tool in c, f"少了 {tool.strip()} 这条查法"


def test_by_port_prints_the_victim_before_killing() -> None:
    """按端口杀比按命令行狠 —— 这台上可能跑着别人的东西，先看清楚再动手。"""
    c = _cmd(by_port=True)
    i_ps, i_kill = c.index("ps -o pid="), c.index("kill $P")
    assert i_ps < i_kill, "先杀后打印等于没打印"


def test_by_port_is_opt_in() -> None:
    assert "fuser" not in _cmd(by_port=False)


def test_start_clears_stale_processes_first() -> None:
    """`start` 之前不清一次的话，Address already in use 会周期性地回来 ——
    而 start 只报 fork 成功，那个错要翻日志才看得见。"""
    src = (ROOT / "deploy_15.sh").read_text(encoding="utf-8")
    i = src.index("cmd_start()")
    tail = src[i:i + 1200]
    assert "--by-port" in tail, "start 之前没清端口"
