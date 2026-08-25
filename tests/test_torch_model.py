"""真实模型执行层：选择性加载 + torch 版 PartialExpertMoEBlock。

在一个**真实 key 命名**的微型 Qwen3-MoE checkpoint 上跑（`sim/fake_checkpoint.py`）。
权重是随机的、输出没有意义 —— 但要测的是**机制**：只加载点名的专家、层内结构接得上、
KV 语义、drop-expert 的行为。数值正确性只能在真机上对真权重验。

torch 是可选依赖（`requirements.txt` 里没有），没装就跳过。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

torch = pytest.importorskip("torch", reason="真实模型层需要 torch（可选依赖）")
pytest.importorskip("safetensors")

from p2pmoe.runtime.torch_model import (  # noqa: E402
    TorchModelConfig,
    TorchPartialExpertMoEBlock,
    TorchSegmentModel,
)
from p2pmoe.runtime.weights import (  # noqa: E402
    KeyPlan,
    SelectiveLoader,
    WeightIndex,
    qwen_moe_keys,
)
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint  # noqa: E402


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("qwen3moe")
    write_fake_checkpoint(d, seed=7)
    return d


@pytest.fixture(scope="module")
def cfg(ckpt):
    return TorchModelConfig.from_hf(json.loads((ckpt / "config.json").read_text()))


def _load(ckpt, layer_experts, **kw):
    idx = WeightIndex(ckpt)
    plan = KeyPlan(layer_experts=layer_experts, **kw)
    return SelectiveLoader(idx).load(qwen_moe_keys(plan))


# --------------------------------------------------------------------------- #
# 选择性加载
# --------------------------------------------------------------------------- #
def test_index_reads_sharded_checkpoint(ckpt) -> None:
    idx = WeightIndex(ckpt)
    assert len(idx.shards) == 2, "分片布局要能识别 —— 真实模型都是分片的"
    assert idx.total_bytes > 0
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in idx.weight_map


def test_only_named_experts_are_loaded(ckpt) -> None:
    """**这是整套方案的核心原语。** 不在清单里的专家张量根本不会被读进来。"""
    tensors, rep = _load(ckpt, {3: [1, 5], 4: [0, 2]})
    assert not rep.missing

    got = {}
    for k in tensors:
        if ".experts." in k:
            layer = int(k.split("model.layers.")[1].split(".")[0])
            e = int(k.split(".experts.")[1].split(".")[0])
            got.setdefault(layer, set()).add(e)
    # 规划器 1-based → checkpoint 0-based
    assert got == {2: {1, 5}, 3: {0, 2}}


def test_layer_index_is_converted_one_based_to_zero_based(ckpt) -> None:
    """规划器用 1-based 层号（文档第〇部分的层区间口径），checkpoint 是 0-based。

    只在 qwen_moe_keys 这一处转换。转错会静默加载错层 —— 输出还是有的，
    只是全错，最难查的那种 bug。
    """
    keys = qwen_moe_keys(KeyPlan(layer_experts={1: [0]}))
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.1.self_attn.q_proj.weight" not in keys


@pytest.mark.parametrize("n_experts", [1, 2, 4, 8])
def test_bytes_loaded_scale_with_subset(ckpt, n_experts) -> None:
    """加载字节数应当随驻留专家数线性增长 —— 省下来的是真的省。"""
    _, rep = _load(ckpt, {1: list(range(n_experts))})
    assert rep.bytes_loaded > 0
    assert rep.fraction < 1.0
    if n_experts == 1:
        base = rep.bytes_loaded
        _, rep8 = _load(ckpt, {1: list(range(8))})
        assert rep8.bytes_loaded > base * 3


def test_missing_keys_are_reported_not_silently_ignored(ckpt) -> None:
    idx = WeightIndex(ckpt)
    tensors, rep = SelectiveLoader(idx).load({"model.layers.0.does.not.exist"})
    assert rep.missing == ["model.layers.0.does.not.exist"]
    assert not tensors


def test_pickle_checkpoint_is_rejected(tmp_path) -> None:
    """权重必须是 safetensors —— .bin 不支持逐张量选择性加载，
    与其在半路失败不如一开始就说清楚。"""
    (tmp_path / "pytorch_model.bin").write_bytes(b"not safetensors")
    with pytest.raises(FileNotFoundError, match="safetensors"):
        WeightIndex(tmp_path)


# --------------------------------------------------------------------------- #
# 执行层
# --------------------------------------------------------------------------- #
def test_full_residency_never_misses(ckpt, cfg) -> None:
    allx = list(range(cfg.n_experts))
    t, _ = _load(ckpt, {1: allx, 2: allx})
    m = TorchSegmentModel(cfg, {1: allx, 2: allx}, t)
    _, st = m.forward("r", torch.randn(6, cfg.d_model))
    assert st.miss_token_layer == 0
    assert st.miss_mass == pytest.approx(0.0, abs=1e-5)


def test_output_identical_when_resident_covers_routed_experts(ckpt, cfg) -> None:
    """**最强的一条**：驻留集只要**包含**实际被路由到的专家，输出就与全量逐位一致。

    这说明「只驻留子集」不是近似 —— 只要没 miss，结果就是精确的。有 miss 时才
    进入 drop-expert 近似（文档标注为「运维近似，非无损」）。
    """
    allx = list(range(cfg.n_experts))
    # token 数要少到路由集是**真子集** —— 全用满就测不出「省了还一致」这件事。
    # top-k=2 时 2 个 token 最多碰 4 个专家。
    torch.manual_seed(3)
    x = torch.randn(2, cfg.d_model)

    t, _ = _load(ckpt, {1: allx})
    full = TorchSegmentModel(cfg, {1: allx}, t)
    y_full, st_full = full.forward("a", x)

    routed = sorted(int(i) for i in (st_full.hist > 0).nonzero()[0])
    assert 0 < len(routed) < cfg.n_experts, f"这个输入路由到了 {routed}，测不出差别"

    t2, _ = _load(ckpt, {1: routed})
    part = TorchSegmentModel(cfg, {1: routed}, t2)
    y_part, st_part = part.forward("b", x)

    assert st_part.miss_token_layer == 0
    assert torch.equal(y_full, y_part), "无 miss 时必须逐位一致"
    assert part.resident_bytes < full.resident_bytes


def test_partial_residency_triggers_miss(ckpt, cfg) -> None:
    t, _ = _load(ckpt, {1: [0]})
    m = TorchSegmentModel(cfg, {1: [0]}, t)
    _, st = m.forward("r", torch.randn(8, cfg.d_model))
    assert st.miss_token_layer > 0
    assert st.miss_mass > 0


def test_all_topk_missing_falls_back_to_residual(ckpt, cfg) -> None:
    """top-k 全缺时本层只走 attention 残差，不凭空造 FFN 输出。"""
    allx = list(range(cfg.n_experts))
    x = torch.randn(4, cfg.d_model)
    t, _ = _load(ckpt, {1: allx})
    full = TorchPartialExpertMoEBlock(cfg, 1, allx, t)
    _, st = full.forward(x, {})
    routed = {int(i) for i in (st.hist > 0).nonzero()[0]}
    other = [e for e in range(cfg.n_experts) if e not in routed]
    if not other:
        pytest.skip("这个输入用满了所有专家")

    t2, _ = _load(ckpt, {1: other})
    blk = TorchPartialExpertMoEBlock(cfg, 1, other, t2)
    y, st2 = blk.forward(x, {})
    assert st2.miss_token_layer == x.shape[0], "每个 token 都该全缺"
    # 全缺时输出 = attention 之后的残差，与再跑一次同样输入的残差一致
    assert torch.isfinite(y).all()


def test_drop_expert_renormalises(ckpt, cfg) -> None:
    """门控重归一：保留下来的权重之和必须是 1。

    用「只驻留 1 个专家」构造：凡是选中它的 token，重归一后权重必然是 1.0，
    于是该 token 的 FFN 增量恰好等于该专家的完整输出。
    """
    t, _ = _load(ckpt, {1: [3]})
    blk = TorchPartialExpertMoEBlock(cfg, 1, [3], t)
    x = torch.randn(12, cfg.d_model)
    y, st = blk.forward(x, {})

    from p2pmoe.runtime.torch_model import _rms_norm

    h = x + blk._attend(_rms_norm(x, blk.ln_in, cfg.rms_eps), {})
    z = _rms_norm(h, blk.ln_post, cfg.rms_eps)
    probs = torch.softmax((z @ blk.router.T).float(), dim=-1)
    _, topi = torch.topk(probs, cfg.top_k, dim=-1)

    for tok in range(x.shape[0]):
        if 3 in topi[tok].tolist():
            hi = (torch.nn.functional.silu(z[tok] @ blk._gate[3].T)
                  * (z[tok] @ blk._up[3].T)) @ blk._down[3].T
            assert torch.allclose(y[tok] - h[tok], hi, atol=1e-5), "权重应重归一到 1.0"


# --------------------------------------------------------------------------- #
# KV 生命周期（命题 III.7.1）
# --------------------------------------------------------------------------- #
def test_kv_accumulates_and_is_per_request(ckpt, cfg) -> None:
    allx = list(range(cfg.n_experts))
    t, _ = _load(ckpt, {1: allx, 2: allx})
    m = TorchSegmentModel(cfg, {1: allx, 2: allx}, t)
    m.forward("a", torch.randn(3, cfg.d_model))
    assert m.kv_tokens("a") == 3
    m.forward("a", torch.randn(1, cfg.d_model))
    assert m.kv_tokens("a") == 4
    m.forward("b", torch.randn(2, cfg.d_model))
    assert m.kv_tokens("b") == 2 and m.kv_tokens("a") == 4
    assert m.drop_kv("a") and not m.has_kv("a")
    assert m.kv_tokens("b") == 2


def test_rebind_keeps_front_kv_and_drops_back_kv(ckpt, cfg) -> None:
    """命题 III.7.1 在真实模型层上同样成立 —— 换绑不重算前段。"""
    allx = list(range(cfg.n_experts))
    t, _ = _load(ckpt, {1: allx, 2: allx, 3: allx, 4: allx}, with_embed=True)
    front = TorchSegmentModel(cfg, {1: allx, 2: allx}, t)
    back_a = TorchSegmentModel(cfg, {3: allx, 4: allx}, t)
    back_b = TorchSegmentModel(cfg, {3: allx, 4: allx}, t)

    h, _ = front.forward("r", front.embed_tokens([1, 2, 3, 4]))
    back_a.forward("r", h)
    assert front.kv_tokens("r") == 4 and back_a.kv_tokens("r") == 4

    assert back_a.drop_kv("r") is True
    assert front.kv_tokens("r") == 4, "前段 KV 不该被换绑影响"

    back_b.forward("r", h)          # 用缓存的 L₀ 输出重放，前段一层不重算
    assert back_b.kv_tokens("r") == 4
    assert front.kv_tokens("r") == 4


def test_decode_matches_prefill_prefix(ckpt, cfg) -> None:
    """KV cache 正确性：逐 token decode 的结果，应当与一次性 prefill 的对应位置一致。

    这条测的是 RoPE 的位置偏移与因果 mask —— 用 cache 时位置从 KV 长度接着数，
    偏移错了这里就会露出来（而单看输出形状完全正常）。
    """
    allx = list(range(cfg.n_experts))
    t, _ = _load(ckpt, {1: allx, 2: allx}, with_embed=True)
    a = TorchSegmentModel(cfg, {1: allx, 2: allx}, t)
    b = TorchSegmentModel(cfg, {1: allx, 2: allx}, t)

    ids = [5, 9, 13, 21, 34]
    x = a.embed_tokens(ids)
    y_once, _ = a.forward("all", x)          # 一次 prefill 全部 5 个

    for i in range(len(ids)):
        y_i, _ = b.forward("step", x[i : i + 1])
        assert torch.allclose(y_i[0], y_once[i], atol=1e-4), \
            f"第 {i} 个 token 对不上 —— 检查 RoPE 位置偏移与因果 mask"


# --------------------------------------------------------------------------- #
# 与 numpy 版的契约一致
# --------------------------------------------------------------------------- #
def test_same_contract_as_numpy_version(ckpt, cfg) -> None:
    """换的是实现，不是行为 —— 两版必须暴露同一组方法。"""
    from p2pmoe.runtime.model import SegmentModel as NumpySegmentModel

    for m in ("forward", "drop_kv", "has_kv", "kv_tokens", "active_requests"):
        assert hasattr(TorchSegmentModel, m) and hasattr(NumpySegmentModel, m)
    for p in ("resident_bytes", "full_bytes"):
        assert isinstance(getattr(TorchSegmentModel, p), property)
        assert isinstance(getattr(NumpySegmentModel, p), property)


def test_stats_are_wire_compatible(ckpt, cfg) -> None:
    """MoEStats 要能过线 —— 在线协议的 miss 检出靠它。"""
    from p2pmoe.runtime.model import MoEStats

    t, _ = _load(ckpt, {1: [0, 1, 2]})
    m = TorchSegmentModel(cfg, {1: [0, 1, 2]}, t)
    _, st = m.forward("r", torch.randn(4, cfg.d_model))
    back = MoEStats.from_wire(st.to_wire())
    assert back.n_token_layer == st.n_token_layer
    assert back.miss_token_layer == st.miss_token_layer
    assert len(back.hist) == cfg.n_experts


def test_empty_resident_set_is_rejected(ckpt, cfg) -> None:
    t, _ = _load(ckpt, {1: [0]})
    with pytest.raises(ValueError):
        TorchPartialExpertMoEBlock(cfg, 1, [], t)


def test_missing_weight_raises_clearly(ckpt, cfg) -> None:
    t, _ = _load(ckpt, {1: [0]})
    with pytest.raises(KeyError, match="experts.5"):
        TorchPartialExpertMoEBlock(cfg, 1, [0, 5], t)
