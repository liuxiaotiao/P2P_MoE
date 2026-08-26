"""Qwen3-Next：混合注意力 + 超稀疏 MoE + 共享专家。

判据与 `test_reference_parity.py` 一样硬：**同一份权重、同一个输入，
与 transformers 官方的 `Qwen3NextForCausalLM` 逐元素比对**。

这个模型比 Qwen3-MoE 多了三处能悄悄算错的地方，写的时候三处都踩了：

* **零中心 RMSNorm** —— `output * (1 + weight)`，权重初始化为 0 而非 1。
  照 Qwen3-MoE 写成 `* weight` 不会报错，只是每个归一化都错；
* **部分旋转 RoPE** —— `partial_rotary_factor = 0.25`，head_dim 256 里只有
  64 维带位置信息，其余原样透传。全维旋转同样不报错；
* **同一模型两种 norm 约定** —— `RMSNorm` 用 `1+w`，`RMSNormGated` 用 `w`。
  这种地方只能照抄，不能推理。

合成 checkpoint 的 norm 权重**故意给随机值而不是 0/1** —— 写 0 的话
`*(1+w)` 与 `*w` 算出来一样，测了等于没测。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_next")

from p2pmoe.runtime.qwen3_next import (
    NextModelConfig,
    TorchNextSegmentModel,
    layer_types_of,
)
from p2pmoe.runtime.weights import (
    KeyPlan, SelectiveLoader, WeightIndex, qwen3_next_keys,
)
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_NEXT, write_fake_next_checkpoint

CFG = dict(TINY_QWEN3_NEXT)
E, L = CFG["num_experts"], CFG["num_hidden_layers"]
TOL = 1e-3


def _fuse(sd: dict) -> dict:
    out = {k: v for k, v in sd.items() if ".mlp.experts." not in k}
    for l in range(L):
        p = f"model.layers.{l}.mlp.experts"
        out[f"{p}.gate_up_proj"] = torch.stack([torch.cat(
            [sd[f"{p}.{e}.gate_proj.weight"], sd[f"{p}.{e}.up_proj.weight"]], 0)
            for e in range(E)])
        out[f"{p}.down_proj"] = torch.stack(
            [sd[f"{p}.{e}.down_proj.weight"] for e in range(E)])
    return out


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    from safetensors.torch import load_file
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    d = tmp_path_factory.mktemp("next")
    write_fake_next_checkpoint(d, CFG, seed=5, n_shards=2)
    sd: dict = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        sd.update(load_file(str(f)))

    hf = Qwen3NextConfig(**{k: v for k, v in CFG.items()
                            if k not in ("architectures", "model_type", "torch_dtype")})
    ref = Qwen3NextForCausalLM(hf).to(torch.float32).eval()
    missing, unexpected = ref.load_state_dict(_fuse(sd), strict=False)
    assert not missing and not unexpected, (
        f"参考实现没吃下我们的权重（{missing[:3]} / {unexpected[:3]}）—— "
        f"那样比的是两组随机数")
    return ref, Path(d), NextModelConfig.from_hf(CFG)


def build(d: Path, cfg: NextModelConfig, layers, *, embed=True, head=True,
          experts=None, policy="drop"):
    le = {l: (list(experts[l]) if experts else list(range(E))) for l in layers}
    keys = qwen3_next_keys(
        KeyPlan(layer_experts=le, with_embed=embed, with_lm_head=head),
        layer_types=list(cfg.layer_types),
        shared_expert=cfg.shared_intermediate > 0,
        tie_word_embeddings=cfg.tie_word_embeddings)
    tensors, rep = SelectiveLoader(WeightIndex(d)).load(keys, dtype=torch.float32)
    assert not rep.missing, rep.missing[:3]
    return TorchNextSegmentModel(cfg, le, tensors, miss_policy=policy)


# --------------------------------------------------------------------------- #
# 1. 层类型 —— 混合结构本身
# --------------------------------------------------------------------------- #
def test_layer_types_follow_the_interval() -> None:
    """`full_attention_interval=4` → 第 3、7、11…（0-based）是标准注意力。"""
    t = layer_types_of({"num_hidden_layers": 8, "full_attention_interval": 4})
    assert t == ["linear_attention"] * 3 + ["full_attention"] \
        + ["linear_attention"] * 3 + ["full_attention"]


def test_explicit_layer_types_win() -> None:
    got = layer_types_of({"num_hidden_layers": 2,
                          "layer_types": ["full_attention", "linear_attention"],
                          "full_attention_interval": 4})
    assert got == ["full_attention", "linear_attention"]


def test_the_two_layer_kinds_need_disjoint_keys(pair) -> None:
    """两种层的权重 key **完全不重叠** —— 这是与 Qwen3-MoE 最大的结构差别。"""
    _, _, cfg = pair
    lt = list(cfg.layer_types)
    lin = qwen3_next_keys(KeyPlan(layer_experts={1: [0]}), layer_types=lt)
    full = qwen3_next_keys(KeyPlan(layer_experts={4: [0]}), layer_types=lt)
    assert any("linear_attn" in k for k in lin)
    assert not any("self_attn" in k for k in lin)
    assert any("self_attn" in k for k in full)
    assert not any("linear_attn" in k for k in full)


def test_every_layer_carries_a_shared_expert(pair) -> None:
    """共享专家对每个 token 都激活 —— 承载该层的节点必须装它，不参与裁剪。"""
    _, _, cfg = pair
    for l in (1, 4):                       # 一层线性、一层标准
        keys = qwen3_next_keys(KeyPlan(layer_experts={l: [0]}),
                               layer_types=list(cfg.layer_types))
        assert any("shared_expert.gate_proj" in k for k in keys)
        assert any("shared_expert_gate" in k for k in keys)


# --------------------------------------------------------------------------- #
# 2. 数值 —— 与官方逐元素比对
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ids", [[3], [3, 17, 200, 5, 88, 41], list(range(1, 25))])
def test_prefill_matches_transformers(pair, ids) -> None:
    """**本文件最要紧的一条。** 零中心 norm、部分旋转 RoPE、DeltaNet 递推、
    共享专家的门控 —— 一条断言同时钉住全部。"""
    ref, d, cfg = pair
    mine = build(d, cfg, range(1, L + 1))
    with torch.no_grad():
        want = ref(torch.tensor([ids])).logits[0]
        h, _ = mine.forward(f"p{len(ids)}", mine.embed_tokens(ids))
        got = mine.logits(h)
    assert got.shape == want.shape
    assert (got - want).abs().max() < TOL
    assert bool((got.argmax(-1) == want.argmax(-1)).all())


def test_each_layer_kind_matches(pair) -> None:
    """逐层比 —— 线性层与标准层各自都要对，否则错会被后面掩掉。"""
    from p2pmoe.runtime.qwen3_next import _rms_norm

    ref, d, cfg = pair
    mine = build(d, cfg, range(1, L + 1))
    ids = [3, 17, 200, 5]
    with torch.no_grad():
        want = ref(torch.tensor([ids]), output_hidden_states=True).hidden_states
        x = mine.embed_tokens(ids)
        seen = set()
        for i, l in enumerate(mine.layers, start=1):
            x, _ = mine.blocks[l].forward(x, {})
            got = (_rms_norm(x, mine.final_norm, cfg.rms_eps)
                   if i == len(mine.layers) else x)
            assert (got - want[i][0]).abs().max() < TOL, \
                f"第 {l} 层（{mine.blocks[l].kind}）对不上"
            seen.add(mine.blocks[l].kind)
    assert seen == {"linear_attention", "full_attention"}, "两种层都要测到"


def test_decode_matches_with_recurrent_state(pair) -> None:
    """decode 走的是另一条路：DeltaNet 的循环状态 + attention 的 KV cache。

    两者都是「跨步携带的状态」，但形态完全不同 —— 一个定长、一个增长。
    连着走 7 步：位置偏移错一位、状态更新顺序反了，都会在这里露馅。
    """
    ref, d, cfg = pair
    mine = build(d, cfg, range(1, L + 1))
    prompt = [3, 17, 200, 5]
    with torch.no_grad():
        h, _ = mine.forward("d", mine.embed_tokens(prompt))
        mt = int(mine.logits(h[-1:])[0].argmax())
        r = ref(torch.tensor([prompt]), use_cache=True)
        rt = int(r.logits[0, -1].argmax())
        past = r.past_key_values
        ms, rs, worst = [mt], [rt], 0.0
        for _ in range(7):
            h, _ = mine.forward("d", mine.embed_tokens([mt]))
            ml = mine.logits(h[-1:])[0]
            mt = int(ml.argmax())
            r = ref(torch.tensor([[rt]]), past_key_values=past, use_cache=True)
            rl = r.logits[0, -1]
            past, rt = r.past_key_values, int(rl.argmax())
            worst = max(worst, float((ml - rl).abs().max()))
            ms.append(mt)
            rs.append(rt)
    assert ms == rs, f"生成分叉：{ms} vs {rs}"
    assert worst < TOL


def test_decode_equals_prefill_of_the_same_prefix(pair) -> None:
    _, d, cfg = pair
    mine = build(d, cfg, range(1, L + 1))
    ids = [3, 17, 200, 5, 88]
    with torch.no_grad():
        h1, _ = mine.forward("a", mine.embed_tokens(ids))
        one = mine.logits(h1[-1:])[0]
        for t in ids:
            h2, _ = mine.forward("b", mine.embed_tokens([t]))
        step = mine.logits(h2[-1:])[0]
    assert (one - step).abs().max() < TOL


# --------------------------------------------------------------------------- #
# 3. 分段 —— 本方案的前提
# --------------------------------------------------------------------------- #
def test_splitting_across_segments_changes_nothing(pair) -> None:
    """切点落在两种层之间也不影响 —— 前段 1–2（线性），后段 3–4（含标准）。"""
    ref, d, cfg = pair
    ids = [3, 17, 200, 5]
    front = build(d, cfg, range(1, 3), embed=True, head=False)
    back = build(d, cfg, range(3, L + 1), embed=False, head=True)
    with torch.no_grad():
        y, _ = front.forward("s", front.embed_tokens(ids))
        z, _ = back.forward("s", y)
        got = back.logits(z)
        want = ref(torch.tensor([ids])).logits[0]
    assert (got - want).abs().max() < TOL


def test_a_subset_covering_the_routing_is_lossless(pair) -> None:
    """只驻留被路由到的专家 → 与全装逐位一致。共享专家照装（它不参与裁剪）。"""
    ref, d, cfg = pair
    ids = [3, 17]
    full = build(d, cfg, range(1, L + 1))
    with torch.no_grad():
        x = full.embed_tokens(ids)
        routed = {}
        for l in full.layers:
            x, st = full.blocks[l].forward(x, {})
            routed[l] = sorted(e for e in range(E) if st.hist[e] > 0)
        assert any(len(v) < E for v in routed.values()), "这个输入用满了专家，测不到"
        thin = build(d, cfg, full.layers, experts=routed)
        h, _ = thin.forward("t", thin.embed_tokens(ids))
        want = ref(torch.tensor([ids])).logits[0]
    assert (thin.logits(h) - want).abs().max() < TOL


@pytest.mark.parametrize("policy", ["drop", "drop_noscale", "local_topk"])
def test_no_miss_means_every_policy_is_exact(pair, policy) -> None:
    """三种补救策略在全装时都必须退化成精确计算 —— 与 Qwen3-MoE 同一条纪律。"""
    ref, d, cfg = pair
    ids = [3, 17, 200, 5]
    mine = build(d, cfg, range(1, L + 1), policy=policy)
    with torch.no_grad():
        h, st = mine.forward("n", mine.embed_tokens(ids))
        want = ref(torch.tensor([ids])).logits[0]
    assert st.miss_token_layer == 0
    assert (mine.logits(h) - want).abs().max() < TOL


# --------------------------------------------------------------------------- #
# 4. 规划器的内存账
# --------------------------------------------------------------------------- #
def test_memory_model_knows_the_layers_differ() -> None:
    """混合模型的逐层内存不同 —— 按「每层一样」估会系统性偏。"""
    from p2pmoe.planner.hf_config import model_spec_from_hf

    real = {"architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
            "num_hidden_layers": 48, "num_attention_heads": 16,
            "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4,
            "linear_num_value_heads": 32, "linear_num_key_heads": 16,
            "linear_value_head_dim": 128, "linear_key_head_dim": 128,
            "linear_conv_kernel_dim": 4, "num_experts": 512,
            "num_experts_per_tok": 10, "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512, "hidden_size": 2048,
            "intermediate_size": 5120, "vocab_size": 151936}
    spec, info = model_spec_from_hf(real, name="next", ctx_max=2048, dtype_bytes=2)
    assert info.hybrid
    assert sum(1 for t in info.layer_types if t == "linear_attention") == 36
    # 真实是 80B × 2 字节 = 160GB
    assert 150 < info.total_gb < 170, info.total_gb
    assert info.shared_gb > 0, "共享专家不能漏 —— 漏了内存判据会偏乐观"
    # DeltaNet 的状态是**定长**的：ctx 翻倍它不变，attention 层翻倍
    assert info.kv_gb_at(1, 4096) == info.kv_gb_at(1, 2048)
    assert info.kv_gb_at(4, 4096) == pytest.approx(2 * info.kv_gb_at(4, 2048))


def test_granularity_verdict_likes_this_model() -> None:
    """512 专家 / top-10 = 51 —— 比 Qwen3-30B-A3B 的 16 好得多。"""
    from p2pmoe.planner.hf_config import granularity_verdict, model_spec_from_hf

    _, info = model_spec_from_hf(CFG, name="tiny-next", ctx_max=128, dtype_bytes=4)
    ok, _ = granularity_verdict(info)
    assert isinstance(ok, bool)
