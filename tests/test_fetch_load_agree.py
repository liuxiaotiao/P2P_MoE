"""下载侧要的 key 与加载侧要的 key 必须**一模一样**。

这条线断掉的症状很贵：`fetch` 把 141GB 下完，`start` 才报「checkpoint 缺 N 个
key」—— 反馈来得太晚，而且错的方向不明显（看起来像 checkpoint 坏了）。

真出过一次：`keys_for_node` 无条件走 Qwen3-MoE 的命名，而 Qwen3-Next 的
key 集合逐层不同（36 层 `linear_attn.*` / 12 层 `self_attn.*`），还多一路
共享专家。于是下载的和加载要的是两套东西。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import keys_for_node
from p2pmoe.planner.manifest import DeploymentManifest
from p2pmoe.runtime.qwen3_next import NextModelConfig
from p2pmoe.runtime.torch_model import TorchModelConfig
from p2pmoe.runtime.weights import KeyPlan, qwen3_next_keys, qwen_moe_keys
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, TINY_QWEN3_NEXT


def _load_side_keys(cfg: dict, man: DeploymentManifest, node: str) -> set[str]:
    """复刻 `NodeServer._build_torch` 的分派 —— 加载侧真正会要的那套 key。"""
    p = man.plan_for(node)
    plan = KeyPlan(
        layer_experts={l.layer: list(l.experts) for l in p.layers},
        with_embed=(p.role == "front" and p.is_head),
        with_lm_head=(p.role.startswith("back:") and p.is_tail),
    )
    arch = str(cfg.get("model_type", "")).lower()
    archs = [str(a).lower() for a in cfg.get("architectures", [])]
    if arch == "qwen3_next" or any("qwen3next" in a for a in archs):
        m = NextModelConfig.from_hf(cfg)
        return qwen3_next_keys(plan, layer_types=list(m.layer_types),
                               shared_expert=m.shared_intermediate > 0,
                               tie_word_embeddings=m.tie_word_embeddings)
    m = TorchModelConfig.from_hf(cfg)
    return qwen_moe_keys(plan, tie_word_embeddings=m.tie_word_embeddings)


def _manifest(n_layers: int, n_experts: int) -> DeploymentManifest:
    l0 = n_layers // 3
    def layers(a, b):
        return [{"layer": l, "experts": list(range(0, n_experts, 2)),
                 "weight_gb": 0.1, "kv_gb": 0.0} for l in range(a, b + 1)]
    return DeploymentManifest.from_dict({
        "l0": l0, "model": {}, "segments": {
            "F0": {"role": "front", "task": None, "nodes": ["nf"]},
            "B0": {"role": "back:u", "task": "u", "nodes": ["nb"]},
        },
        "nodes": [
            {"node": "nf", "role": "front", "segment": "F0", "position": 0,
             "is_head": True, "is_tail": True, "layer_range": [1, l0],
             "weight_gb": 1.0, "kv_gb": 0.0, "total_gb": 1.0,
             "layers": layers(1, l0)},
            {"node": "nb", "role": "back:u", "segment": "B0", "position": 0,
             "is_head": True, "is_tail": True, "layer_range": [l0 + 1, n_layers],
             "weight_gb": 1.0, "kv_gb": 0.0, "total_gb": 1.0,
             "layers": layers(l0 + 1, n_layers)},
        ],
    })


CASES = [
    pytest.param(dict(TINY_QWEN3_MOE), id="qwen3_moe"),
    pytest.param(dict(TINY_QWEN3_NEXT), id="qwen3_next"),
]


@pytest.mark.parametrize("cfg", CASES)
@pytest.mark.parametrize("node", ["nf", "nb"])
def test_the_two_sides_ask_for_exactly_the_same_keys(cfg, node) -> None:
    man = _manifest(cfg["num_hidden_layers"], cfg["num_experts"])
    want = _load_side_keys(cfg, man, node)
    got = keys_for_node(man, node, config=cfg)
    assert got == want, (
        f"下载 {len(got)} 个、加载要 {len(want)} 个；"
        f"只下不加载 {sorted(got - want)[:3]}，加载要但没下 {sorted(want - got)[:3]}")


def test_next_gets_its_shared_expert() -> None:
    """共享专家对每个 token 都激活、不参与裁剪。

    漏了它**不会报错**（key 缺失才报），只会让承载该层的节点少算一路输出 ——
    这种错最难发现，因为模型照样出 token。
    """
    cfg = dict(TINY_QWEN3_NEXT)
    man = _manifest(cfg["num_hidden_layers"], cfg["num_experts"])
    keys = keys_for_node(man, "nb", config=cfg)
    n_layers = len(man.plan_for("nb").layers)
    for suffix in ("gate_proj", "up_proj", "down_proj"):
        n = sum(1 for k in keys if k.endswith(f"mlp.shared_expert.{suffix}.weight"))
        assert n == n_layers, f"shared_expert.{suffix} 只有 {n}/{n_layers} 层"
    assert sum(1 for k in keys if "shared_expert_gate" in k) == n_layers


def test_next_uses_linear_attn_where_the_config_says_so() -> None:
    """36 层 DeltaNet 没有 self_attn.*。拿 MoE 的方案去要，要的是不存在的 key。"""
    cfg = dict(TINY_QWEN3_NEXT)
    man = _manifest(cfg["num_hidden_layers"], cfg["num_experts"])
    keys = keys_for_node(man, "nb", config=cfg)
    types = list(NextModelConfig.from_hf(cfg).layer_types)
    for l in (x.layer for x in man.plan_for("nb").layers):
        pre = f"model.layers.{l-1}."
        has_lin = any(k.startswith(pre + "linear_attn.") for k in keys)
        has_att = any(k.startswith(pre + "self_attn.") for k in keys)
        if types[l - 1] == "linear_attention":
            assert has_lin and not has_att, f"层 {l} 该是 DeltaNet"
        else:
            assert has_att and not has_lin, f"层 {l} 该是全注意力"


def test_moe_is_untouched_by_the_next_branch() -> None:
    """Qwen3-MoE 的行为必须一个字节都没变 —— 修 Next 不能碰它。"""
    cfg = dict(TINY_QWEN3_MOE)
    man = _manifest(cfg["num_hidden_layers"], cfg["num_experts"])
    assert keys_for_node(man, "nb", config=cfg) == keys_for_node(man, "nb")


def test_no_config_falls_back_to_moe() -> None:
    """拿不到 config 的调用方保持原行为，而不是崩掉。"""
    cfg = dict(TINY_QWEN3_MOE)
    man = _manifest(cfg["num_hidden_layers"], cfg["num_experts"])
    assert keys_for_node(man, "nf", config=None)
