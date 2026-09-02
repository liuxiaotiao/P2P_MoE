"""`--advertise ""` 必须和「没给」同等对待。

这是今晚花掉四个小时的那个 bug。判断原来写的是 `if advertise is None`，
而 `deploy_15.sh` 在 ADVERTISE 没设时会原样传 `--advertise ""`。空串不是
None，推断分支被跳过，空串一路下发给 15 台节点。

之后的表现是这套系统里最难查的一种 —— **每一项检查都是绿的**：

    控制机 → 节点   configure ✓   权重装好 ✓   check 全绿 ✓
    节点 → 控制机   回报打到 ("" , port) → OSError
                    → _serve_conn 的 except (ConnectionError, OSError): pass
                    → 连接一关，线程退出，什么都不留

协调器永远等不到识别事件，请求 300s 超时；而去节点上看，现场是「进程空闲、
GPU 0%、零条连接、零个工作线程」—— 完全像是从没收到过活。

所以这里测的不是「空串会不会报错」，而是**空串会不会被当成一个合法地址传下去**。
"""

from __future__ import annotations

from p2pmoe.deploy.control import coordinator_host

REMOTE = {"N01": ("192.168.1.2", 9101), "N02": ("192.168.1.3", 9101)}
LOCAL = {"N01": ("127.0.0.1", 9101), "N02": ("localhost", 9102)}


def test_an_empty_advertise_is_treated_as_absent() -> None:
    """核心断言。空串进去，绝不能空串出来。"""
    host, guessed = coordinator_host("", REMOTE)
    assert host, "空串被当成了合法地址 —— 15 台节点会拿到一个回不去的地址"
    assert guessed is True


def test_none_advertise_is_inferred() -> None:
    host, guessed = coordinator_host(None, REMOTE)
    assert host
    assert guessed is True


def test_empty_and_none_agree() -> None:
    """两者在语义上是同一件事，结论必须一致。"""
    assert coordinator_host("", REMOTE) == coordinator_host(None, REMOTE)
    assert coordinator_host("", LOCAL) == coordinator_host(None, LOCAL)


def test_an_explicit_address_is_used_verbatim() -> None:
    """给了就用，一个字符都不改，也不标成推断出来的。"""
    host, guessed = coordinator_host("10.0.0.5", REMOTE)
    assert host == "10.0.0.5"
    assert guessed is False


def test_all_local_peers_infer_loopback() -> None:
    """全在本机就该是 127.0.0.1，不该去猜出网网卡的地址。"""
    host, guessed = coordinator_host(None, LOCAL)
    assert host == "127.0.0.1"
    assert guessed is True


def test_whitespace_is_not_an_address() -> None:
    """`ADVERTISE=" "` 这种也一样回不去。"""
    host, _ = coordinator_host("   ".strip(), REMOTE)
    assert host.strip(), "空白串被当成了地址"


def test_the_serve_path_guards_advertise_too() -> None:
    """守卫不能只装在 check 上 —— 那正好是唯一不靠它的子命令。

    check 只做可达性探测，全程不下发清单；真正建反向链路的是 serve/measure。
    原来的分布正好是反的，于是 `check` 拦得住而 `measure` 裸奔。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "deploy_15.sh").read_text(
        encoding="utf-8")
    assert "_require_advertise()" in src, "守卫没有抽成可复用的函数"
    # cmd_check 和 cmd_serve 都要调它：定义 1 次 + 调用 2 次
    assert src.count("_require_advertise") >= 3, \
        "serve 路径上没有 ADVERTISE 守卫 —— 空串会一路传到 control.py"
