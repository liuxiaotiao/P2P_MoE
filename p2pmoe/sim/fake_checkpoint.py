"""生成一个微型的 Qwen3-MoE 格式 checkpoint —— 用来在不下载 61GB 的前提下验证机制。

**key 命名与真实 checkpoint 一字不差**（取自 Qwen/Qwen3-30B-A3B 的
`model.safetensors.index.json`），分片布局也一样。所以选择性加载器在这上面能跑通，
换成真权重就只是文件更大。

它验证得了的：加载器只打开需要的 key、字节数确实按比例下降、层内结构接得上、
KV 语义、drop-expert 的行为。

它验证不了的：真实权重的数值正确性（合成权重是随机的，输出没有意义），以及
真实规模下的显存与带宽压力。那两件事只能在真机上对真模型做。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

__all__ = ["TINY_QWEN3_MOE", "write_fake_checkpoint", "write_fake_tokenizer"]

# 微型 checkpoint 的对话模板 —— 就是 Qwen3 的 ChatML 骨架，去掉 tools/thinking
# 那些分支。留着它是为了让 `--chat` 这条路在测试里真的被走到：模板渲染错了
# （比如多一个 BOS、少一个角色标记）在真模型上表现为「输出像坏了」，很难查。
_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

TINY_QWEN3_MOE: dict = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "num_hidden_layers": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "hidden_size": 64,
    "moe_intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "norm_topk_prob": True,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


def write_fake_checkpoint(
    out_dir: str | Path,
    cfg: Mapping = TINY_QWEN3_MOE,
    *,
    seed: int = 0,
    n_shards: int = 2,
) -> Path:
    """写出 config.json + 分片 safetensors + index.json。

    分成多个分片是有意的：真实模型都是分片的，而选择性加载的一个要点就是
    「只打开含有目标 key 的那几个分片」。单文件测不出这条。
    """
    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg), indent=2), encoding="utf-8")

    g = torch.Generator().manual_seed(seed)
    d = int(cfg["hidden_size"])
    f = int(cfg["moe_intermediate_size"])
    h, kvh, hd = (int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]),
                  int(cfg["head_dim"]))
    E, L, V = int(cfg["num_experts"]), int(cfg["num_hidden_layers"]), int(cfg["vocab_size"])

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * (1.0 / max(shape[-1], 1) ** 0.5)).float()

    tensors: dict[str, "torch.Tensor"] = {
        "model.embed_tokens.weight": rnd(V, d),
        "model.norm.weight": torch.ones(d),
        "lm_head.weight": rnd(V, d),
    }
    for i in range(L):
        p = f"model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = rnd(h * hd, d)
        tensors[f"{p}.self_attn.k_proj.weight"] = rnd(kvh * hd, d)
        tensors[f"{p}.self_attn.v_proj.weight"] = rnd(kvh * hd, d)
        tensors[f"{p}.self_attn.o_proj.weight"] = rnd(d, h * hd)
        tensors[f"{p}.self_attn.q_norm.weight"] = torch.ones(hd)
        tensors[f"{p}.self_attn.k_norm.weight"] = torch.ones(hd)
        tensors[f"{p}.input_layernorm.weight"] = torch.ones(d)
        tensors[f"{p}.post_attention_layernorm.weight"] = torch.ones(d)
        tensors[f"{p}.mlp.gate.weight"] = rnd(E, d)
        for e in range(E):
            tensors[f"{p}.mlp.experts.{e}.gate_proj.weight"] = rnd(f, d)
            tensors[f"{p}.mlp.experts.{e}.up_proj.weight"] = rnd(f, d)
            tensors[f"{p}.mlp.experts.{e}.down_proj.weight"] = rnd(d, f)

    keys = sorted(tensors)
    per = max(1, (len(keys) + n_shards - 1) // n_shards)
    weight_map: dict[str, str] = {}
    for s in range(n_shards):
        chunk = keys[s * per : (s + 1) * per]
        if not chunk:
            continue
        name = f"model-{s+1:05d}-of-{n_shards:05d}.safetensors"
        save_file({k: tensors[k].contiguous() for k in chunk}, str(out / name))
        for k in chunk:
            weight_map[k] = name

    (out / "generation_config.json").write_text(
        # 与真 checkpoint 同一套语义：generation_config 的 eos 才是生成时该用的，
        # 它和 config.json 里那个架构层面的 eos 可以不是同一个（Qwen3 就不是）。
        json.dumps({"eos_token_id": _EOS_ID, "do_sample": False}, indent=2),
        encoding="utf-8",
    )
    write_fake_tokenizer(out, vocab_size=V)

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(
            t.numel() * t.element_size() for t in tensors.values())},
            "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    return out


# --------------------------------------------------------------------------- #
_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
_EOS_ID = 2      # <|im_end|> —— 与 _SPECIALS 的下标一致

_TRAIN_TEXT = [
    "hello world", "the quick brown fox jumps over the lazy dog",
    "把请求打进前段，绕环出 token", "分散环境下网络占九成",
    "front segment back segment expert routing",
    "你好，世界。这是一段用来训练微型分词器的中文。",
]


def write_fake_tokenizer(out_dir: str | Path, *, vocab_size: int = 512) -> Path:
    """写一份能用的 byte-level BPE `tokenizer.json`。

    **必须是 byte-level 的**，不能图省事用 word-level：整套增量解码的难点就在
    「一个 token 可能是半个 UTF-8 字符」（中文一个字 3 字节，常被切成 2+1）。
    word-level 分词器每个 token 都是完整字符，`Detokenizer` 那条 U+FFFD 回退
    分支永远走不到 —— 测了等于没测。所以训练语料里特意有中文。

    byte-level 的 256 个字节 token 是底座，vocab 至少要 256 + 特殊 token；
    这也是 `TINY_QWEN3_MOE` 的 vocab_size 是 512 而不是 128 的原因。
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        _TRAIN_TEXT,
        trainers.BpeTrainer(vocab_size=int(vocab_size), special_tokens=list(_SPECIALS),
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                            show_progress=False),
    )
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(
        json.dumps({
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": _CHAT_TEMPLATE,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out / "tokenizer.json"
