"""真机部署路径的测试。

这里跑的是**真实子进程 + 真实 socket**，不是 fork。测的是三件只在多机场景下
才存在的东西：agent 的两阶段启动、由节点自己发起的探测、清单下发。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from p2pmoe.deploy.agent import parse_bind
from p2pmoe.deploy.launch import SYSTEMD_UNIT, Host, read_hosts
from p2pmoe.deploy.control import parse_agents
from p2pmoe.deploy.probe import RemoteNetworkOracle
from p2pmoe.runtime.node import NodeConfig, NodeServer
from p2pmoe.runtime.wire import LinkTable, rpc


# --------------------------------------------------------------------------- #
def test_parse_agents() -> None:
    got = parse_agents("v1=10.0.0.11:9101, g1=10.0.0.21:9102")
    assert got == {"v1": ("10.0.0.11", 9101), "g1": ("10.0.0.21", 9102)}
    with pytest.raises(ValueError):
        parse_agents("")


def test_parse_bind() -> None:
    assert parse_bind("0.0.0.0:9101") == ("0.0.0.0", 9101)
    assert parse_bind("9101") == ("0.0.0.0", 9101)
    assert parse_bind(":9101") == ("0.0.0.0", 9101)


def test_read_hosts(tmp_path) -> None:
    f = tmp_path / "hosts.txt"
    f.write_text(
        "# 注释行\n"
        "v1  10.0.0.11\n"
        "v2  10.0.0.12  --mem-mb 32000\n"
        "g1  10.0.0.21:9102  --mem-mb 47000\n"
        "\n",
        encoding="utf-8",
    )
    hosts = read_hosts(f)
    assert [h.node_id for h in hosts] == ["v1", "v2", "g1"]
    assert hosts[0].addr == ("10.0.0.11", 9101)      # 默认端口
    assert hosts[2].addr == ("10.0.0.21", 9102)
    assert hosts[1].extra == ["--mem-mb", "32000"]


def test_read_hosts_rejects_duplicate_ids(tmp_path) -> None:
    f = tmp_path / "h.txt"
    f.write_text("v1 10.0.0.11\nv1 10.0.0.12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="唯一"):
        read_hosts(f)


def test_read_hosts_rejects_equals_form(tmp_path) -> None:
    """hosts 文件是空格分隔；`v1=10.0.0.11` 是 --agents 的格式，别混用。"""
    f = tmp_path / "h.txt"
    f.write_text("v1 v1=10.0.0.11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="格式"):
        read_hosts(f)


def test_probe_matrix_verdicts(agents) -> None:
    """`launch probe` 的判定：本地延迟 < 1ms → 明确告知这套方法没有对象。

    这条不是形式主义。整套方案的前提是「网络占单 token 延迟九成」，池子若在
    LAN 上，均匀性机制无事可做 —— 与其让规划在 Step 4 神秘失败，不如上真机
    第一步就量出来并说清楚。
    """
    from p2pmoe.deploy.launch import Host, _probe_matrix

    hosts = [Host(s_.me, "127.0.0.1", s_.port) for s_ in agents]
    rc = _probe_matrix(hosts, k=4, parallel=4)
    assert rc == 2, "本机延迟应被判为 LAN 量级"


def test_systemd_unit_renders() -> None:
    u = SYSTEMD_UNIT.format(node_id="v1", port=9101, user="p2pmoe",
                            workdir="/opt/p2pmoe", python="python3", extra="")
    assert "--id v1" in u and "0.0.0.0:9101" in u
    assert "Restart=always" in u          # agent 无状态，崩了自动重来
    assert "WantedBy=multi-user.target" in u


# --------------------------------------------------------------------------- #
@pytest.fixture
def agents():
    """起两个真实的 agent 服务（线程内，但走真 socket）。"""
    servers = [NodeServer(f"a{i}", host="127.0.0.1", port=0) for i in (1, 2)]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for t in threads:
        t.start()
    time.sleep(0.2)
    yield servers
    for s in servers:
        s._stop.set()
    time.sleep(0.2)


def test_agent_starts_unconfigured(agents) -> None:
    """两阶段启动：先起服务，此时还没有模型。"""
    for s in agents:
        assert s.model is None and s.cfg is None
        caps = s.capabilities()
        assert caps["configured"] is False
        assert caps["mem_mb"] > 0 and caps["ms_per_layer"] > 0


def test_capabilities_over_the_wire(agents) -> None:
    a = agents[0]
    r = rpc(("127.0.0.1", a.port), {"type": "capabilities"}, timeout=10)
    assert r["type"] == "capabilities_ack"
    assert r["node"] == "a1" and r["configured"] is False


def test_unconfigured_agent_refuses_data_plane(agents) -> None:
    """没配置就收到数据面消息 → 明确报错，而不是静默吞掉。"""
    a = agents[0]
    r = rpc(("127.0.0.1", a.port), {"type": "prefill", "req": "x", "ids": [1, 2]}, timeout=10)
    assert r["type"] == "error" and "还没配置" in r["trace"]


def test_probe_is_initiated_by_the_node(agents) -> None:
    """探测由 a1 去量 a2 —— 控制器只下指令，不自己 ping。"""
    a, b = agents
    r = rpc(("127.0.0.1", a.port),
            {"type": "probe", "peer": "a2", "addr": ["127.0.0.1", b.port], "k": 5},
            timeout=15)
    assert r["type"] == "probe_ack" and r["ok"] is True
    assert r["src"] == "a1" and r["peer"] == "a2"
    assert r["p50"] >= 0 and r["p95"] >= r["p50"]


def test_probe_reports_unreachable_as_infinite(agents) -> None:
    """A 连不上 B 是真实的拓扑事实（NAT / 防火墙），不是错误 ——
    返回不可达，让规划器自然绕开这条链路。"""
    a = agents[0]
    dead = ("127.0.0.1", 1)   # 几乎不可能有人监听
    r = rpc(("127.0.0.1", a.port),
            {"type": "probe", "peer": "ghost", "addr": list(dead), "k": 2}, timeout=15)
    assert r["ok"] is False

    oracle = RemoteNetworkOracle({"a1": ("127.0.0.1", a.port), "ghost": dead})
    pr = oracle.probe("a1", "ghost", 2)
    assert pr.p50 == float("inf")
    assert oracle.failures


def test_remote_oracle_caches_and_is_symmetric(agents) -> None:
    a, b = agents
    addrs = {"a1": ("127.0.0.1", a.port), "a2": ("127.0.0.1", b.port)}
    o = RemoteNetworkOracle(addrs, k_default=4, symmetric=True)
    p1 = o.probe("a1", "a2", 4)
    n_after_first = o.n_rpc
    p2 = o.probe("a2", "a1", 4)     # 对称 ⇒ 命中缓存，不再发 RPC
    assert o.n_rpc == n_after_first
    assert p1.p50 == p2.p50
    assert o.reachability(["a1", "a2"]) == {"a1": 1, "a2": 1}


def test_configure_loads_only_named_experts(agents) -> None:
    """下发清单后，节点只装点名的那几层的那些专家。"""
    a, b = agents
    peers = {"a1": ["127.0.0.1", a.port], "a2": ["127.0.0.1", b.port]}
    cfg = NodeConfig(
        node_id="a1", role="back:X", segment="BX0",
        layer_experts={1: [0, 3, 5], 2: [1, 2]},
        next_hop=None, seg_head="a1", is_head=True, is_tail=True,
        peers=peers, links=LinkTable().to_dict(),
        coordinator=["127.0.0.1", 1], model={},
    )
    r = rpc(("127.0.0.1", a.port), {"type": "configure", "config": cfg.to_dict()}, timeout=30)
    assert r["type"] == "configure_ack"
    assert r["layers"] == [1, 2]
    assert r["n_experts"] == 5
    assert r["resident_mb"] < r["full_mb"]

    assert a.model is not None
    assert a.model.blocks[1].resident == frozenset({0, 3, 5})
    assert a.model.blocks[2].resident == frozenset({1, 2})
    # 非驻留专家的张量根本不存在
    assert set(a.model.blocks[1]._w1) == {0, 3, 5}

    caps = rpc(("127.0.0.1", a.port), {"type": "capabilities"}, timeout=10)
    assert caps["configured"] is True


# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_multinode_15_nodes_end_to_end() -> None:
    """15 台节点的完整多机流程 —— 真实子进程起 agent，真实控制器跑五步。

    一次跑完覆盖三件事：
      1. 部署路径本身走得通（发现 → 探测 → 规划 → 下发 → 服务）；
      2. **不是所有节点都会被用上** —— 规划按需取子集，其余是天然的备用池；
      3. 劣质接入的节点被自然排除 —— 它们既进不了公共带（做不了前段出口），
         也当不了后段入口（端点项会惩罚它）。

    只留这一个重量级用例：起 8 个和起 15 个进程的两个用例连着跑会互相挤，
    偶发超时。15 台这个场景已经把 8 台的都覆盖了。
    """
    import re

    r = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "multinode_local.py"),
         "--nodes", "15", "--requests", "2", "--tokens", "5"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "清单校验 通过" in out, out[-3000:]
    assert "汇总：识别 2/2" in out, out[-2000:]

    bad = set(re.findall(r"(n\d+)=\d+±\d+\[劣\]", out))
    assert len(bad) >= 2, f"没解析到劣质节点: {out[:600]}"

    # 日志行形如 "... INFO   n5   front/F0   层 [...]"
    used = set(re.findall(r"\b(n\d+)\s+(?:front|back:)\S*\s+层", out))
    assert used, "没解析到被使用的节点"
    assert not (used & bad), f"劣质接入节点 {used & bad} 竟被排进方案"
    assert len(used) < 15, "15 台全被用上了？规划本应只取所需的子集"
