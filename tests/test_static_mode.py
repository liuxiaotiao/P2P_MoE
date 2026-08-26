"""静态简化模式：离线定死的前后段链路。

三层分开测：
  * `full_placement` —— 前段装全集这个决定本身；
  * `assign_static_pairs` —— 离线配对的指派语义；
  * `Coordinator(static_wiring=...)` —— 在线只剩「按 task 取通道」。

以及一条回归：环里的在途 token 在完成判定之后回来，不能把通道归还两次。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.experts import ExpertPlacement, full_placement
from p2pmoe.planner.static_pairing import assign_static_pairs
from p2pmoe.planner.types import Segment
from p2pmoe.runtime.coordinator import Coordinator

from test_serving_loop import FakeManifest, FakePool


# --------------------------------------------------------------------------- #
# 1. 前段装全集
# --------------------------------------------------------------------------- #
def test_full_placement_is_every_expert_on_every_layer() -> None:
    p = full_placement(4, 8)
    assert all(p.at(l) == frozenset(range(8)) for l in range(1, 5))
    assert p.achieved_coverage == (1.0,) * 4


def test_full_placement_dominates_any_union() -> None:
    """全集是并集的上界 —— 逐层包含。III.5.4 的支配性因此自动成立。"""
    uni = ExpertPlacement(name="u", n_layers=3,
                          sets=(frozenset({0, 3}), frozenset({1}), frozenset({2, 5, 7})),
                          achieved_coverage=(0.9, 0.9, 0.9))
    full = full_placement(3, 8)
    assert all(full.at(l) >= uni.at(l) for l in range(1, 4))
    assert sum(full.sizes()) > sum(uni.sizes())   # 代价：更多内存


# --------------------------------------------------------------------------- #
# 2. 离线配对
# --------------------------------------------------------------------------- #
@dataclass
class FakeStat:
    p50: float
    p95: float = 0.0
    jitter: float = 0.0


class FakeNet:
    """延迟 = 两端 id 的数字之和 —— 便于手算出唯一的最优指派。"""

    def get(self, a: str, b: str, k: int) -> FakeStat:
        return FakeStat(p50=float(int(a[1:]) + int(b[1:])))


def seg(nodes: list[str], task: str | None = None, delay: float = 0.0) -> Segment:
    return Segment(kind="back" if task else "front", task=task,
                   nodes=tuple(nodes), splits=((1, 2),) * len(nodes),
                   compute_ms=delay, hop_ms=0.0)


def test_each_side_used_at_most_once() -> None:
    fronts = [seg([f"a{i}"]) for i in range(3)]
    backs = {"X": [seg(["b0"], "X"), seg(["b1"], "X")], "Y": [seg(["b2"], "Y")]}
    w = assign_static_pairs(fronts, backs, FakeNet())
    assert len(w.pairs) == 3
    assert len({p.front_id for p in w.pairs}) == 3
    assert len({p.back_id for p in w.pairs}) == 3


def test_leftover_fronts_become_spares() -> None:
    """前段多于后段 —— 多的那些是备胎，不是错误。"""
    fronts = [seg([f"a{i}"]) for i in range(4)]
    backs = {"X": [seg(["b0"], "X")]}
    w = assign_static_pairs(fronts, backs, FakeNet())
    assert len(w.pairs) == 1
    assert len(w.spare_fronts) == 3
    assert not w.unpaired_backs


def test_unpaired_backs_are_reported_not_swallowed() -> None:
    """后段多于前段 —— 有后段没通道，必须说出来（否则那个 task 静默无服务）。"""
    fronts = [seg(["a0"])]
    backs = {"X": [seg(["b0"], "X")], "Y": [seg(["b1"], "Y")]}
    w = assign_static_pairs(fronts, backs, FakeNet())
    assert len(w.pairs) == 1
    assert len(w.unpaired_backs) == 1
    assert "没配上" in w.summary()


def test_greedy_takes_the_cheapest_pair_first() -> None:
    fronts = [seg(["a9"]), seg(["a1"])]          # F0=a9（贵）、F1=a1（便宜）
    backs = {"X": [seg(["b1"], "X"), seg(["b9"], "X")]}
    w = assign_static_pairs(fronts, backs, FakeNet())
    cheapest = min(w.pairs, key=lambda p: p.t50)
    assert (cheapest.front_id, cheapest.back_id) == ("F1", "BX0")   # a1 × b1


def test_as_map_is_what_the_coordinator_eats() -> None:
    fronts = [seg(["a0"]), seg(["a1"])]
    backs = {"X": [seg(["b0"], "X")], "Y": [seg(["b1"], "Y")]}
    m = assign_static_pairs(fronts, backs, FakeNet()).as_map()
    assert set(m) <= {"F0", "F1"}
    assert {t for _, t in m.values()} == {"X", "Y"}


def test_spread_is_over_chosen_pairs_only() -> None:
    """静态模式的极差口径：只管选中的那几对，不管全组合矩阵。"""
    fronts = [seg(["a0"]), seg(["a8"])]
    backs = {"X": [seg(["b0"], "X")], "Y": [seg(["b1"], "Y")]}
    w = assign_static_pairs(fronts, backs, FakeNet())
    assert w.spread_ms == pytest.approx(max(p.t50 for p in w.pairs)
                                        - min(p.t50 for p in w.pairs))


# --------------------------------------------------------------------------- #
# 3. 在线：只剩「按 task 取通道」
# --------------------------------------------------------------------------- #
WIRING = {"F0": ("BX0", "X"), "F1": ("BY0", "Y")}


def make_static(**kw) -> tuple[Coordinator, FakePool]:
    man = FakeManifest(n_front=2, backs={"X": 1, "Y": 1})
    c = Coordinator(man, baselines={}, priors={}, static_wiring=WIRING, **kw)
    c.pool = FakePool()
    c.max_tokens = 2
    return c, c.pool


def token(c: Coordinator, req: str, phase: str = "decode") -> None:
    c._on({"type": "token", "req": req, "node": "b", "segment": "B",
           "phase": phase, "token": 1,
           "back_stats": {"hist": [], "ntl": 3, "miss": 3, "mass": 1.0}})


def test_fronts_are_partitioned_by_task() -> None:
    c, _ = make_static()
    assert not c.free_fronts                      # 全局池不再存在
    assert {u: list(q) for u, q in c.front_pools.items()} == {"X": ["F0"], "Y": ["F1"]}


def test_submit_without_task_is_refused() -> None:
    """静态模式下 task 不是推断出来的，是调用方给的 —— 没给就是用法错误。"""
    c, _ = make_static()
    with pytest.raises(ValueError, match="必须给 task"):
        c.submit("r0", [1, 2])


def test_unknown_task_names_what_is_available() -> None:
    c, _ = make_static()
    with pytest.raises(ValueError, match=r"\['X', 'Y'\]"):
        c.submit("r0", [1], task="Z")


def test_binding_happens_at_submit_with_no_control_rtt() -> None:
    """到达即成对：没有识别、没有向协调器要后段这一跳。"""
    c, pool = make_static()
    r = c.submit("r0", [1, 2], task="X")
    assert (r.front, r.back, r.task) == ("F0", "BX0", "X")
    assert len(pool.of_type("prefill")) == 1
    assert not pool.of_type("bind")               # 动态模式才有的派发消息


def test_second_request_on_a_busy_channel_queues() -> None:
    c, pool = make_static()
    c.submit("r0", [1], task="X")
    r1 = c.submit("r1", [2], task="X")
    assert r1.front == ""                          # X 通道占着
    assert len(pool.of_type("prefill")) == 1
    r2 = c.submit("r2", [3], task="Y")             # 另一条通道不受影响
    assert r2.front == "F1"


def test_queued_request_starts_when_its_channel_returns() -> None:
    c, pool = make_static()
    c.submit("r0", [1], task="X")
    r1 = c.submit("r1", [2], task="X")
    for _ in range(c.max_tokens):
        token(c, "r0")
    assert c.records["r0"].done.is_set()
    assert r1.front == "F0" and r1.back == "BX0"   # 同一条通道，配对不变
    assert len(pool.of_type("prefill")) == 2


def test_miss_never_triggers_rebind() -> None:
    """miss 率照统计，但静态模式下它不是误绑的证据 —— 不存在误绑。"""
    c, _ = make_static(alarm_factor=1.0)
    r = c.submit("r0", [1], task="X")
    c.max_tokens = 50
    for _ in range(12):
        token(c, "r0")                             # back_stats 的 miss 率是 100%
    assert r.rebinds == 0
    assert r.back == "BX0"


def test_channel_returns_exactly_once_after_a_late_token() -> None:
    """回归：完成判定之后环里还有一个在途 token 绕回来。

    早先的版本会收下它 → tokens 超过 max_tokens → 二次 _finish → 同一条通道
    被 append 两次，池深凭空长大。环是异步的，「停」必然要在某个已发出的计算
    之后生效，所以迟到的 token 只能丢弃。
    """
    c, _ = make_static()
    c.submit("r0", [1], task="X")
    for _ in range(c.max_tokens + 3):              # 多喂 3 个「迟到」的
        token(c, "r0")
    assert len(c.records["r0"].tokens) == c.max_tokens
    assert list(c.front_pools["X"]) == ["F0"]      # 恰好一条，不是三条


def test_queue_depths_reports_channels_not_two_pools() -> None:
    c, _ = make_static()
    c.submit("r0", [1], task="X")
    d = c.queue_depths()
    assert d["mode"] == "static"
    assert d["free_channels"] == {"X": 0, "Y": 1}


# --------------------------------------------------------------------------- #
# 4. 端到端：真 checkpoint 格式 + 定死的链路，真的绕出 token
# --------------------------------------------------------------------------- #
def test_static_ring_produces_tokens_on_a_real_checkpoint(tmp_path) -> None:
    """把整条路走一遍：合成 checkpoint → 选择性加载 → 静态链路 → 出 token。

    权重是随机的，所以 token 没有语义 —— 这里断言的是**机制**：
    前段装全集、后段只装子集、配对在配置期就定死、环真的绕回来、
    跑完通道恰好归还一次。
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    import json

    from p2pmoe.planner.experts import build_placement
    from p2pmoe.planner.hf_config import model_spec_from_hf
    from p2pmoe.planner.network import MeasurementCache
    from p2pmoe.planner.pipeline import plan
    from p2pmoe.planner.types import Node, PlannerConfig, TaskProfile
    from p2pmoe.runtime.coordinator import LocalCluster
    from p2pmoe.runtime.wire import LinkTable
    from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint
    from p2pmoe.sim.network import SimNetwork
    from p2pmoe.sim.replay import make_activation_profiles

    mdir = write_fake_checkpoint(tmp_path / "ckpt", TINY_QWEN3_MOE, seed=1)
    hf = json.loads((mdir / "config.json").read_text(encoding="utf-8"))
    spec, info = model_spec_from_hf(hf, name="tiny", ctx_max=128, dtype_bytes=4)

    names = ["X", "Y"]
    profiles = make_activation_profiles(names, {u: 3 for u in names},
                                        n_layers=info.n_layers,
                                        n_experts=info.n_experts, seed=1)
    plcs = {u: build_placement(p, 0.95) for u, p in profiles.items()}
    tasks = [TaskProfile(name=u, lam=0.5,
                         experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u in names]
    nodes = [Node(id=f"n{i+1}", tier="1GB", mem_gb=1.0, ms_per_layer=0.35,
                  reserve_gb=0.0, avail=0.95) for i in range(6)]
    sim = SimNetwork([n.id for n in nodes], seed=1, good_access=(12.0, 16.0),
                     bad_access=(28.0, 33.0), bad_frac=0.2, backbone=(2.0, 5.0),
                     jitter=(4.0, 9.0))
    cfg = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                        kappa_over=0.3, n_standby=0, seed=1)
    net = MeasurementCache(sim, k=cfg.k_probe, j_cap_ms=cfg.j_cap_ms, k_gate=cfg.k_gate)
    res = plan(nodes, spec, tasks, full_placement(info.n_layers, info.n_experts),
               net, cfg, {l: 0.75 + 0.05 * l for l in range(1, info.n_layers)},
               p_min=0.75)
    wiring = assign_static_pairs(res.fronts_final, res.backs, net, k=cfg.k_audit)
    assert wiring.pairs, "微型池子应该至少配出一条通道"
    wired = wiring.as_map()

    links = LinkTable(
        p50={(a, b): sim.true_p50(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        jitter={(a, b): sim.true_jitter(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        scale=0.05,
    )
    served = sorted({t for _, t in wired.values()})

    # 文本层只活在控制机侧 —— 节点的配置一个字都不用变
    from p2pmoe.runtime.text import TextIO
    textio = TextIO.from_model_dir(mdir)

    with LocalCluster(res.manifest, None, links, static_wiring=wired,
                      backend="torch", model_dir=str(mdir), model_hf=hf,
                      textio=textio) as cl:
        cl.coord.max_tokens = 3
        done = []
        for i, u in enumerate(served * 2):
            rec = cl.coord.submit(f"r{i}", text="hello 世界", task=u)
            assert rec.done.wait(timeout=120), "\n".join(rec.events)
            done.append(rec)

        assert not cl.coord.errors, cl.coord.errors[:1]
        assert all(len(r.tokens) == 3 for r in done)
        assert all(0 <= t < info.vocab for r in done for t in r.tokens)
        # 文本真的走通了：prompt 编码进去，生成的 id 又拼回了字符
        assert all(r.ids == textio.tok.encode("hello 世界") for r in done)
        assert all(r.text == textio.tok.decode(r.tokens) for r in done), \
            "流式增量拼出来的必须和一次性 decode 逐字相同"
        # 配对是定死的：同一个 task 的两条请求必定落在同一对段上
        for u in served:
            got = {(r.front, r.back) for r in done if r.task == u}
            assert len(got) == 1, f"task {u} 的组合变了：{got}"
        assert all(r.rebinds == 0 for r in done)
        # 每条通道恰好归还一次
        depths = cl.coord.queue_depths()
        assert depths["mode"] == "static"
        assert all(v == 1 for v in depths["free_channels"].values()), depths

    # 前段装全集、后段装子集 —— 清单层面复核一次
    fronts = [p for p in res.manifest.nodes if p.role == "front"]
    backs = [p for p in res.manifest.nodes if p.role.startswith("back:")]
    assert all(len(l.experts) == info.n_experts for p in fronts for l in p.layers)
    assert any(len(l.experts) < info.n_experts for p in backs for l in p.layers)
