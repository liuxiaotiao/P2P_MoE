"""HF config → ModelSpec 的换算，以及「这个模型适不适合本方案」的判据。

数值对照各模型 config.json 的公开字段（见 hf_config.PRESETS）。这些是外部事实，
测试把它们钉住 —— 换算错了会直接把可行性判断带偏。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.planner.hf_config import (
    PRESETS,
    granularity_verdict,
    model_spec_from_hf,
)


# --------------------------------------------------------------------------- #
def test_mixtral_total_size_matches_known_figure() -> None:
    """Mixtral 8x7B fp16 约 93GB —— 换算对不对，拿这个已知值校。"""
    spec, info = model_spec_from_hf(PRESETS["mixtral-8x7b"], name="mixtral")
    assert info.n_layers == 32 and info.n_experts == 8 and info.top_k == 2
    assert info.total_gb == pytest.approx(93.4, rel=0.02)
    # 每个专家 3 × 4096 × 14336 × 2B
    assert info.expert_gb == pytest.approx(3 * 4096 * 14336 * 2 / 1e9, rel=1e-6)


def test_qwen3_moe_total_size_matches_known_figure() -> None:
    """Qwen3-30B-A3B：30B 参数 × 2B ≈ 60GB。"""
    spec, info = model_spec_from_hf(PRESETS["qwen3-30b-a3b"], name="qwen3")
    assert info.n_layers == 48 and info.n_experts == 128 and info.top_k == 8
    assert info.total_gb == pytest.approx(61.0, rel=0.03)


def test_olmoe_total_size_matches_known_figure() -> None:
    """OLMoE-1B-7B：7B × 2B ≈ 14GB。"""
    spec, info = model_spec_from_hf(PRESETS["olmoe-1b-7b"], name="olmoe")
    assert info.total_gb == pytest.approx(13.8, rel=0.03)


# --------------------------------------------------------------------------- #
def test_gqa_kv_dim_is_not_d_model() -> None:
    """GQA 下 KV 维远小于 d_model —— 按 d_model 算会把 KV 高估好几倍。

    Qwen3-30B-A3B：4 个 KV 头 × 128 = 512，而 d_model 是 2048，差 4 倍。
    文档 I.2.2 的公式隐含 MHA，真实模型上必须修正。
    """
    spec, info = model_spec_from_hf(PRESETS["qwen3-30b-a3b"])
    assert info.kv_dim == 4 * 128 == 512
    assert spec.kv_dim == 512
    naive = spec.ctx_max * 2 * spec.d_model * spec.kv_bytes_per_elem / 1e9
    assert spec.kv_gb_per_layer == pytest.approx(naive / 4, rel=1e-6)


def test_mha_falls_back_to_d_model() -> None:
    """OLMoE 是 MHA（kv 头数 = 头数），KV 维就等于 d_model。"""
    spec, info = model_spec_from_hf(PRESETS["olmoe-1b-7b"])
    assert info.kv_dim == spec.d_model
    assert spec.kv_gb_per_layer == pytest.approx(
        spec.ctx_max * 2 * spec.d_model * 2 / 1e9, rel=1e-6
    )


# --------------------------------------------------------------------------- #
def test_granularity_rejects_mixtral() -> None:
    """**这条是选模型的核心判据。**

    Mixtral 只有 8 个专家、top-2：单 token 就用掉 25%，一个 task 的驻留集会接近
    全集，「只驻留子集」省不下内存 —— 本方案的前提不成立。这与显存多少无关，
    是模型自身的性质。
    """
    _, info = model_spec_from_hf(PRESETS["mixtral-8x7b"])
    assert info.granularity == 4.0
    ok, why = granularity_verdict(info)
    assert not ok and "前提不成立" in why


def test_granularity_accepts_fine_grained_moe() -> None:
    _, info = model_spec_from_hf(PRESETS["qwen3-30b-a3b"])
    assert info.granularity == 16.0
    ok, why = granularity_verdict(info)
    assert ok and "适合本方案" in why


def test_granularity_is_monotone() -> None:
    """比值越大越适用 —— 判据本身要单调，不能出现反转。"""
    infos = [model_spec_from_hf(c, name=n)[1] for n, c in PRESETS.items()]
    infos.sort(key=lambda i: i.granularity)
    verdicts = [granularity_verdict(i)[0] for i in infos]
    assert verdicts == sorted(verdicts), f"判据不单调: {[i.name for i in infos]}"


# --------------------------------------------------------------------------- #
def test_resident_subset_actually_saves_memory_on_fine_grained() -> None:
    """细粒度模型上，驻留 20% 的专家应当省掉大部分内存。"""
    _, info = model_spec_from_hf(PRESETS["qwen3-30b-a3b"])
    n = round(info.n_experts * 0.20)
    assert info.layer_gb(n) / info.layer_full_gb < 0.25


def test_field_aliases_across_families() -> None:
    """各家字段名不统一：Mixtral 用 num_local_experts，Qwen3 用 num_experts。"""
    a, _ = model_spec_from_hf({
        "num_hidden_layers": 4, "num_local_experts": 8, "num_experts_per_tok": 2,
        "hidden_size": 128, "intermediate_size": 256,
        "num_attention_heads": 4, "vocab_size": 100,
    })
    b, _ = model_spec_from_hf({
        "num_hidden_layers": 4, "num_experts": 8, "moe_top_k": 2,
        "hidden_size": 128, "moe_intermediate_size": 256,
        "num_attention_heads": 4, "vocab_size": 100,
    })
    assert a.n_experts == b.n_experts == 8
    assert a.top_k == b.top_k == 2
    assert a.expert_gb == b.expert_gb


def test_missing_required_field_raises() -> None:
    with pytest.raises(KeyError):
        model_spec_from_hf({"num_hidden_layers": 4})
