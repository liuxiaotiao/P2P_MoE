"""15 个节点 id 必须落在 15 处**不同的**目录上。

症状很迷惑：某台反复重下，拿到的却总是另一台那份，重下多少次都一样。
最容易想到的解释（下载出错、驻留集变了）全是错的 —— 真正的原因是
**两个节点 id 指向同一处**：同一台机器的两个 IP、NAT 后面的同一台、
或者 `$WEIGHTS` 落在共享盘上。那样两条 fetch 互相覆盖，谁后完成谁留下。

判据不能靠猜：让每台往自己的 `$WEIGHTS` 里写一个写着自己 id 的标记，
再回头读一遍。读出别人的 id，就实锤了。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy_15.sh"

pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="没有 deploy_15.sh")
SRC = SCRIPT.read_text(encoding="utf-8")


def _blk() -> str:
    i = SRC.index("cmd_identity")
    return SRC[i:i + 4000]


def test_it_writes_a_marker_then_reads_it_back() -> None:
    """两趟是必须的 —— 写完立刻读的话，覆盖还没发生。"""
    b = _blk()
    assert ".whoami" in b
    i_w = b.index("> $(printf %q \"$WEIGHTS\")/.whoami")
    i_r = b.index("cat $(printf %q \"$WEIGHTS\")/.whoami")
    assert i_w < i_r, "得先全部写完，再回头读"


def test_the_two_passes_are_separated_by_a_wait() -> None:
    """第一趟没全部落盘就去读，会读到半截状态。"""
    b = _blk()
    i_w = b.index(".whoami")
    tail = b[i_w:b.index("cat $(printf %q \"$WEIGHTS\")/.whoami")]
    assert "wait" in tail


def test_it_also_compares_machine_ids() -> None:
    """同一台机器配了两个 IP 是常见的配错 —— machine-id 一撞就知道。"""
    b = _blk()
    assert "machine-id" in b
    assert "uniq -d" in b, "没有找重复的 machine-id"


def test_it_resolves_the_real_path() -> None:
    """`$WEIGHTS` 可能是软链或共享挂载 —— 字面路径相同不代表是同一处，
    字面不同也不代表不是。"""
    assert "readlink -f" in _blk()


def test_the_header_is_ascii() -> None:
    """`printf %-Ns` 按字符数补宽，中文占两列会把整张表拧歪。"""
    b = _blk()
    m = re.search(r"printf '  %-6s%-16s%-11s%-8s%s\\n' (.+)", b)
    assert m, "找不到表头"
    assert all(ord(c) < 128 for c in m.group(1)), f"表头有非 ASCII：{m.group(1)}"


def test_a_clash_is_an_error_not_a_note() -> None:
    """撞车意味着这个部署放不下 15 条独立通道 —— 那是结论，不是提示。"""
    b = _blk()
    i = b.index("clash")
    assert "return 1" in b[i:], "撞车时没有非零返回"


def test_the_advice_covers_both_causes() -> None:
    """同一台机器 与 共享盘，两者的处置完全不同。"""
    b = _blk()
    assert "同一台机器" in b and ("共享盘" in b or "共享" in b)
    assert "WEIGHTS=" in b, "没给出改路径的具体写法"


def test_it_is_wired_into_the_dispatch() -> None:
    """接进 case 分派，并且出现在**总用法**那一行里。

    找的是带花括号列表的那句 —— 脚本里还有几处子命令各自的用法提示
    （`用法: $0 refetch <节点id>` 之类），抓错了会验了个寂寞。
    """
    assert "identity)" in SRC, "case 分派里没有"
    m = [l for l in SRC.splitlines() if "用法: $0 {" in l]
    assert m, "找不到总用法那一行"
    assert "identity" in m[0], f"总用法里没列出来：{m[0][:120]}"
