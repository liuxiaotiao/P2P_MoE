"""configure 超时之后，要分清「还在装」和「卡死了」。

**只做 TCP connect 不够。** 监听 socket 还在，内核就把连接收进 backlog ——
进程卡死在一个永不返回的系统调用里（挂掉的 NFS、坏掉的块设备）照样连得上。
那样得到的「进程活着」是个假信号，把人往「再等等」引，而等到天荒地老也不会好。

agent 是每连接一个线程，装载模型的同时仍然应答得了 `capabilities`。所以：

    连得上 + 答得出  → 真的只是在装
    连得上 + 不答    → 卡死
    连不上          → 进程没了
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.control import _probe_agent

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_no_listener_is_gone() -> None:
    assert _probe_agent("N", ("127.0.0.1", _free_port()), timeout=2) == "gone"


def test_listening_but_never_answering_is_wedged() -> None:
    """**这条是重点。**

    一个 bind + listen 但从不 accept 的 socket —— TCP 连接照样成功，
    因为内核把它放进 backlog。只看「连得上」就会把这判成健康。
    """
    w = socket.socket()
    w.bind(("127.0.0.1", 0))
    w.listen(8)
    try:
        assert _probe_agent("N", w.getsockname(), timeout=3) == "wedged"
    finally:
        w.close()


def test_a_real_agent_is_alive() -> None:
    port = _free_port()
    pr = subprocess.Popen(
        [sys.executable, "-m", "p2pmoe.deploy.agent", "--id", "NP",
         "--bind", f"127.0.0.1:{port}", "--mem-mb", "4000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT)
    try:
        for _ in range(40):
            if _probe_agent("NP", ("127.0.0.1", port), timeout=2) == "alive":
                break
            time.sleep(0.25)
        assert _probe_agent("NP", ("127.0.0.1", port), timeout=5) == "alive"
    finally:
        pr.terminate()
        pr.wait(timeout=10)


def test_an_agent_answers_while_another_thread_is_busy() -> None:
    """前提校验：agent 必须是每连接一个线程，否则上面那条判据不成立 ——
    单线程的话「装载中」和「卡死」都表现为不应答，分不开。"""
    src = (ROOT / "p2pmoe" / "runtime" / "node.py").read_text(encoding="utf-8")
    i = src.index("def serve_forever")
    blk = src[i:i + 800]
    assert "threading.Thread" in blk, "agent 不是每连接一个线程了 —— 探活判据要重想"


def test_the_timeout_message_distinguishes_all_three() -> None:
    src = (ROOT / "p2pmoe" / "deploy" / "control.py").read_text(encoding="utf-8")
    i = src.index("配置 %s 超时")
    blk = src[i:i + 3000]
    assert 'state == "alive"' in blk
    assert 'state == "wedged"' in blk
    assert "卡死了" in blk or "卡死" in blk
    # 卡死那支要指向存储，不是「再等等」
    j = blk.index('state == "wedged"')
    tail = blk[j:j + 1500]
    assert "nfs" in tail.lower() or "网络盘" in tail
    assert "dmesg" in tail
