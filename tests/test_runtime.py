"""运行时测试：执行层语义、KV 生命周期、通信层、端到端协议。

重点在**可判定的语义**，不在数值：
  * 只驻留子集时，非驻留专家的张量确实不存在；
  * drop-expert 的门控确实重归一（和为 1）；
  * 换绑时前段 KV 确实原地不动、后段 KV 确实作废（命题 III.7.1）；
  * 端到端：注入一个误绑，通道二能自己发现并纠正。
"""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from p2pmoe.planner.experts import build_placement, union_placement
from p2pmoe.runtime.corpus import (
    make_corpus,
    measure_baseline_miss,
    measure_front_refs,
    profile_from_corpus,
    sample_prompt,
)
from p2pmoe.runtime.identify import HistogramClassifier
from p2pmoe.runtime.model import (
    PartialExpertMoEBlock,
    SegmentModel,
    ToyMoEConfig,
    embed_tokens,
)
from p2pmoe.runtime.wire import LinkTable, recv_msg, send_msg

CFG = ToyMoEConfig(n_layers=6, d_model=64, n_experts=16, d_ff=64, vocab=128)


# --------------------------------------------------------------------------- #
# 执行层：选择性加载
# --------------------------------------------------------------------------- #
def test_only_resident_experts_are_materialised() -> None:
    b = PartialExpertMoEBlock(CFG, layer=1, resident=[0, 3, 7])
    assert set(b._w1) == {0, 3, 7} and set(b._w2) == {0, 3, 7}
    # 非驻留专家的张量根本不存在 —— 不是「加载后不用」
    for e in range(CFG.n_experts):
        if e not in b.resident:
            assert e not in b._w1
    assert b.resident_bytes < b.full_bytes


@pytest.mark.parametrize("n", [1, 4, 8, 16])
def test_resident_bytes_scales_with_subset(n: int) -> None:
    b = PartialExpertMoEBlock(CFG, layer=2, resident=range(n))
    per = CFG.expert_params * 8
    assert b.resident_bytes == pytest.approx(b.full_bytes - (CFG.n_experts - n) * per)


def test_empty_resident_set_is_rejected() -> None:
    with pytest.raises(ValueError):
        PartialExpertMoEBlock(CFG, layer=1, resident=[])


# --------------------------------------------------------------------------- #
# 执行层：drop-expert 与 miss 统计
# --------------------------------------------------------------------------- #
def test_full_residency_never_misses() -> None:
    m = SegmentModel(CFG, {l: range(CFG.n_experts) for l in range(1, CFG.n_layers + 1)})
    _, st = m.forward("r", embed_tokens(CFG, [1, 2, 3, 4, 5]))
    assert st.miss_token_layer == 0
    assert st.miss_mass == pytest.approx(0.0)


def test_drop_expert_renormalises_gates() -> None:
    """门控重归一：跳过缺失专家后，剩余权重之和必须是 1。

    直接验证 —— 用只驻留 1 个专家的层，此时若该专家被选中，输出必然等于
    「门控权重 1.0 × 该专家」，因为重归一后只剩它一个。
    """
    x = embed_tokens(CFG, [11, 12, 13])
    one = PartialExpertMoEBlock(CFG, layer=1, resident=[0])
    y, st = one.forward(x, {})

    # 手工复算：对每个 top-k 里选中专家 0 的 token，输出增量应等于该专家的完整输出
    h = x + one._attend(x, {})
    hn = h / np.linalg.norm(h, axis=-1, keepdims=True)
    logits = (hn @ one.wr) * CFG.router_temp
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    topk = np.argpartition(-p, CFG.top_k - 1, axis=-1)[:, : CFG.top_k]

    for t in range(x.shape[0]):
        if 0 in topk[t]:
            want = np.maximum(h[t] @ one._w1[0], 0.0) @ one._w2[0]   # 门控 = 1.0
            assert np.allclose(y[t] - h[t], want, atol=1e-9)
    assert st.miss_token_layer == x.shape[0]  # 每个 token 都至少缺一个


def test_all_topk_missing_falls_back_to_residual() -> None:
    """top-k 全缺时，本层对该 token 只走 attention 残差（不能凭空造 FFN 输出）。"""
    x = embed_tokens(CFG, [5])
    b = PartialExpertMoEBlock(CFG, layer=3, resident=range(CFG.n_experts))
    h_full, _ = b.forward(x, {})
    # 找出实际被选中的专家，然后构造一个不含它们的驻留集
    hn = x + b._attend(x, {})
    hh = hn / np.linalg.norm(hn, axis=-1, keepdims=True)
    logits = (hh @ b.wr) * CFG.router_temp
    picks = set(np.argsort(-logits[0])[: CFG.top_k].tolist())
    other = [e for e in range(CFG.n_experts) if e not in picks]
    b2 = PartialExpertMoEBlock(CFG, layer=3, resident=other)
    y, st = b2.forward(x, {})
    assert st.miss_token_layer == 1
    assert np.allclose(y[0], hn[0])  # 只剩残差


def test_miss_mass_counts_dropped_gate_mass() -> None:
    b = PartialExpertMoEBlock(CFG, layer=1, resident=[0])
    _, st = b.forward(embed_tokens(CFG, [7, 8]), {})
    assert st.miss_mass > 0
    # 丢掉的质量不可能超过总门控质量（每 token 每层 top-k 之和 ≤ 1）
    assert st.miss_mass <= st.n_token_layer


# --------------------------------------------------------------------------- #
# KV 生命周期（命题 III.7.1）
# --------------------------------------------------------------------------- #
def test_kv_accumulates_across_decode_steps() -> None:
    m = SegmentModel(CFG, {l: range(CFG.n_experts) for l in range(1, CFG.n_layers + 1)})
    m.forward("r", embed_tokens(CFG, [1, 2, 3]))
    assert m.kv_tokens("r") == 3
    m.forward("r", embed_tokens(CFG, [4]))
    assert m.kv_tokens("r") == 4


def test_rebind_keeps_front_kv_and_drops_back_kv() -> None:
    """命题 III.7.1 的直接验证：换绑时前段 KV 原地不动、后段 KV 作废。

    前段的计算是对输入的精确 forward（专家路由由输入决定），识别是旁路统计、
    不进计算图，所以前段 KV 对任何事后的 task 结论都有效。
    """
    l0 = 3
    front = SegmentModel(CFG, {l: range(CFG.n_experts) for l in range(1, l0 + 1)})
    back_a = SegmentModel(CFG, {l: range(CFG.n_experts) for l in range(l0 + 1, CFG.n_layers + 1)})
    back_b = SegmentModel(CFG, {l: range(CFG.n_experts) for l in range(l0 + 1, CFG.n_layers + 1)})

    h, _ = front.forward("r", embed_tokens(CFG, [1, 2, 3, 4]))
    back_a.forward("r", h)
    assert front.kv_tokens("r") == 4 and back_a.kv_tokens("r") == 4

    # 换绑：旧后段丢 KV，前段**不调用** drop_kv
    assert back_a.drop_kv("r") is True
    assert not back_a.has_kv("r")
    assert front.kv_tokens("r") == 4, "前段 KV 不该被换绑影响"

    # 用缓存的 L₀ 输出重放给新后段 —— 前段一层都不重算
    back_b.forward("r", h)
    assert back_b.kv_tokens("r") == 4
    assert front.kv_tokens("r") == 4


def test_kv_is_per_request() -> None:
    m = SegmentModel(CFG, {l: range(CFG.n_experts) for l in range(1, CFG.n_layers + 1)})
    m.forward("a", embed_tokens(CFG, [1, 2]))
    m.forward("b", embed_tokens(CFG, [3]))
    assert m.kv_tokens("a") == 2 and m.kv_tokens("b") == 1
    m.drop_kv("a")
    assert not m.has_kv("a") and m.kv_tokens("b") == 1


# --------------------------------------------------------------------------- #
# 通信层
# --------------------------------------------------------------------------- #
def test_wire_roundtrip_with_array() -> None:
    a, b = socket.socketpair()
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = threading.Thread(target=send_msg, args=(a, {"type": "hop", "req": "r"}, arr))
    t.start()
    h, got = recv_msg(b)
    t.join()
    assert h["type"] == "hop" and h["req"] == "r"
    assert np.array_equal(got, arr)
    a.close()
    b.close()


def test_wire_roundtrip_without_array() -> None:
    a, b = socket.socketpair()
    t = threading.Thread(target=send_msg, args=(a, {"type": "ping"}, None))
    t.start()
    h, got = recv_msg(b)
    t.join()
    assert h["type"] == "ping" and got is None
    a.close()
    b.close()


def test_link_table_sampling_matches_target_quantiles() -> None:
    lt = LinkTable(p50={("a", "b"): 40.0}, jitter={("a", "b"): 12.0})
    rng = np.random.default_rng(0)
    s = np.array([lt.sample_ms("a", "b", rng) for _ in range(20000)])
    assert np.median(s) == pytest.approx(40.0, rel=0.05)
    assert np.quantile(s, 0.95) - np.median(s) == pytest.approx(12.0, rel=0.15)
    assert lt.sample_ms("a", "a", rng) == 0.0


def test_link_scale_applies() -> None:
    lt = LinkTable(p50={("a", "b"): 40.0}, jitter={("a", "b"): 10.0}, scale=0.5)
    rng = np.random.default_rng(1)
    s = np.array([lt.sample_ms("a", "b", rng) for _ in range(5000)])
    assert np.median(s) == pytest.approx(20.0, rel=0.08)


# --------------------------------------------------------------------------- #
# 识别
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def calibrated():
    cfg = ToyMoEConfig()
    corpus = make_corpus(cfg, ["X", "Y", "Z"], seed=3, shared_clusters=0)
    profiles = profile_from_corpus(cfg, corpus)
    plcs = {u: build_placement(p, 0.95) for u, p in profiles.items()}
    uni = union_placement(list(plcs.values()))
    return cfg, corpus, profiles, plcs, uni


def test_classifier_separates_tasks(calibrated) -> None:
    cfg, corpus, profiles, plcs, uni = calibrated
    l0 = 3
    refs = measure_front_refs(cfg, corpus, uni, l0)
    clf = HistogramClassifier(refs, {"X": 0.5, "Y": 0.3, "Z": 0.2})
    front = SegmentModel(cfg, {l: uni.at(l) for l in range(1, l0 + 1)})
    ok = 0
    for u in ("X", "Y", "Z"):
        for j in range(4):
            ids = sample_prompt(corpus, u, 12, seed=500 + j)
            _, st = front.forward(f"{u}{j}", embed_tokens(cfg, ids))
            ok += clf.predict(st.hist).task == u
            front.drop_kv(f"{u}{j}")
    assert ok >= 10, f"12 条里只认对 {ok} 条"


def test_confidence_zones() -> None:
    refs = {"A": np.array([1.0, 0.0, 0.0]), "B": np.array([0.0, 1.0, 0.0])}
    clf = HistogramClassifier(refs, {"A": 0.6, "B": 0.4}, tau_hi=0.9, tau_lo=0.6, temp=6.0)
    assert clf.predict([1.0, 0.0, 0.0]).zone == "commit"
    v = clf.predict([1.0, 1.0, 0.0])   # 完全对称 ⇒ 置信最低
    assert v.zone == "prior" and v.task == "A"   # 绑最大先验池（III.7.5）
    assert v.keep_cache is True


# --------------------------------------------------------------------------- #
# 基线校准 —— 与文档口径的偏差
# --------------------------------------------------------------------------- #
def test_measured_baseline_exceeds_nominal(calibrated) -> None:
    """文档 II.5 的「基线 = 1−覆盖率」在 top-k 路由下系统性偏低。

    三条原因（见 corpus.measure_baseline_miss）：口径差一个 k、前段 miss 沿层
    放大、decode 阶段分布漂移。这条测试把差距固定下来，防止有人「优化」回去。
    """
    cfg, corpus, profiles, plcs, uni = calibrated
    l0 = 3
    measured = measure_baseline_miss(cfg, corpus, uni, plcs, l0, n_seq=4, n_decode=8)
    for u, p in plcs.items():
        nominal = p.baseline_miss(l0 + 1, cfg.n_layers)
        assert measured[u] > nominal * 1.5, (
            f"task {u}: 实测 {measured[u]:.3f} vs 名义 {nominal:.3f}"
        )


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_end_to_end_ring_and_rebind() -> None:
    """起真实进程，跑完整一圈，并验证注入的误绑能被自己发现并纠正。"""
    import examples.e2e as e2e  # noqa: F401  —— 仅确认可导入

    from p2pmoe.planner.network import MeasurementCache
    from p2pmoe.planner.pipeline import plan
    from p2pmoe.planner.types import PlannerConfig, TaskProfile
    from p2pmoe.runtime.coordinator import LocalCluster
    from p2pmoe.sim.network import SimNetwork

    cfg = ToyMoEConfig()
    corpus = make_corpus(cfg, ["X", "Y", "Z"], seed=3, shared_clusters=0)
    profiles = profile_from_corpus(cfg, corpus)
    plcs = {u: build_placement(p, 0.95) for u, p in profiles.items()}
    uni = union_placement(list(plcs.values()))

    spec = e2e.toy_model_spec(cfg)
    nodes = e2e.build_pool()
    tasks = [TaskProfile(name=u, lam=l, experts_per_layer=plcs[u].as_experts_per_layer(),
                         placement=plcs[u]) for u, l in e2e.TASKS]
    sim = SimNetwork([n.id for n in nodes], seed=3, good_access=(12.0, 16.0),
                     bad_access=(28.0, 33.0), bad_frac=0.2, backbone=(2.0, 5.0),
                     jitter=(4.0, 9.0))
    pc = PlannerConfig(eta=0.15, beta=1.3, j_cap_ms=30.0, theta=0.8,
                       kappa_over=0.3, n_standby=0, seed=3)
    net = MeasurementCache(sim, k=pc.k_probe, j_cap_ms=pc.j_cap_ms, k_gate=pc.k_gate)
    res = plan(nodes, spec, tasks, uni, net, pc,
               {l: 0.70 + 0.05 * l for l in range(2, cfg.n_layers)}, p_min=0.80)
    assert res.manifest is not None and res.manifest.ok

    links = LinkTable(
        p50={(a, b): sim.true_p50(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        jitter={(a, b): sim.true_jitter(a, b) for a in sim.node_ids for b in sim.node_ids if a != b},
        scale=0.1,
    )
    refs = measure_front_refs(cfg, corpus, uni, res.l0, n_seq=8)
    clf = HistogramClassifier(refs, {u: l for u, l in e2e.TASKS})
    base = measure_baseline_miss(cfg, corpus, uni, plcs, res.l0, n_seq=4, n_decode=8)

    with LocalCluster(res.manifest, cfg, links, clf, baselines=base,
                      priors={u: l for u, l in e2e.TASKS}, alarm_factor=3.0) as cl:
        cl.coord.max_tokens = 10

        clean = cl.coord.submit("t-clean", sample_prompt(corpus, "X", 12, seed=1),
                                true_task="X")
        assert clean.done.wait(timeout=120), "干净请求没跑完"
        assert clean.task == "X", f"识别错了: {clean.task}"
        assert clean.rebinds == 0, "绑对的池不该报警 —— 基线校准失效"
        assert len(clean.tokens) >= 10

        wrong = next(t for t, _ in e2e.TASKS if t != "X")
        inj = cl.coord.submit("t-inject", sample_prompt(corpus, "X", 12, seed=2),
                              true_task="X", force_task=wrong)
        assert inj.done.wait(timeout=120), "注入请求没跑完"
        assert inj.rebinds >= 1, "注入的误绑没有被通道二发现"
        assert inj.task == "X", f"换绑没纠正回来: {inj.task}"

        assert not cl.coord.errors, cl.coord.errors[:2]
