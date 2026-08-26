"""**对齐官方实现**：我们自己写的 Qwen3-MoE 前向 vs transformers 的。

为什么必须有这一组
------------------
`torch_model.py` 是从头写的：attention、GQA、RoPE、QK-RMSNorm、top-k 路由、
逐专家前向。理由充分（fused kernel 拿不到 routing、也不支持专家子集，见该文件
开头），但代价是**没有任何东西替我们保证它算对了**。

而这类 bug 的症状极其隐蔽：RoPE 的实部虚部拆反、norm 放在残差前还是后、
门控权重乘在 down 之前还是之后 —— 每一个都不会抛异常，只会让输出**看起来像
模型很差**。合成 checkpoint 的权重本来就是随机的，输出本来就是乱码，
肉眼永远看不出区别。

所以判据只能是：**同一份权重、同一个输入，与官方实现逐元素比对。**

其它测试验的是别的东西
----------------------
* `test_torch_model.py`：机制 —— 只驻留子集、drop-expert 重归一、KV 生命周期。
  它保证「我们的实现自洽」，不保证「我们的实现是 Qwen3」；
* 本文件：数值 —— 我们的实现**就是** Qwen3。

transformers 只在开发/测试时需要（`requirements-dev.txt`），
推理节点上不装 —— 它的 fused MoE block 正是我们要替换掉的东西。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel
from p2pmoe.runtime.weights import KeyPlan, SelectiveLoader, WeightIndex, qwen_moe_keys
from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_MOE, write_fake_checkpoint

CFG = dict(TINY_QWEN3_MOE, num_hidden_layers=4)
TOL = 1e-3          # float32 累加噪声在 1e-6 量级，1e-3 已经很松


def _fuse_experts(sd: dict, n_layers: int, n_experts: int) -> dict:
    """逐专家张量 → transformers 5.x 的融合布局。

    `Qwen3MoeExperts` 把每层的专家打成 `[E, 2*inter, hidden]` 与 `[E, hidden, inter]`。
    **这正是我们不能直接用它的原因**：融合之后就没有「只加载第 20 层的 3、7、12 号
    专家」这个概念了。这里做反向转换，只是为了让参考实现能吃我们的 checkpoint。
    """
    out = {k: v for k, v in sd.items() if ".mlp.experts." not in k}
    for l in range(n_layers):
        p = f"model.layers.{l}.mlp.experts"
        out[f"{p}.gate_up_proj"] = torch.stack([
            torch.cat([sd[f"{p}.{e}.gate_proj.weight"], sd[f"{p}.{e}.up_proj.weight"]], 0)
            for e in range(n_experts)
        ])
        out[f"{p}.down_proj"] = torch.stack(
            [sd[f"{p}.{e}.down_proj.weight"] for e in range(n_experts)])
    return out


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """(参考实现, 我们的实现, checkpoint 目录) —— 同一份随机权重。"""
    from safetensors.torch import load_file
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    d = tmp_path_factory.mktemp("ckpt")
    write_fake_checkpoint(d, CFG, seed=7, n_shards=2)
    sd: dict = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        sd.update(load_file(str(f)))

    hf = Qwen3MoeConfig(
        num_hidden_layers=CFG["num_hidden_layers"], num_experts=CFG["num_experts"],
        num_experts_per_tok=CFG["num_experts_per_tok"], hidden_size=CFG["hidden_size"],
        moe_intermediate_size=CFG["moe_intermediate_size"],
        intermediate_size=CFG["moe_intermediate_size"],
        num_attention_heads=CFG["num_attention_heads"],
        num_key_value_heads=CFG["num_key_value_heads"], head_dim=CFG["head_dim"],
        vocab_size=CFG["vocab_size"], rms_norm_eps=CFG["rms_norm_eps"],
        rope_theta=CFG["rope_theta"], norm_topk_prob=CFG["norm_topk_prob"],
        tie_word_embeddings=False, decoder_sparse_step=1, mlp_only_layers=[],
        attention_dropout=0.0,
    )
    ref = Qwen3MoeForCausalLM(hf).to(torch.float32).eval()
    missing, unexpected = ref.load_state_dict(
        _fuse_experts(sd, CFG["num_hidden_layers"], CFG["num_experts"]), strict=False)
    assert not missing and not unexpected, (
        f"参考实现没吃下我们的权重（missing {missing[:3]}, unexpected {unexpected[:3]}）"
        f"—— 那样比的是两组随机数，测了等于没测"
    )

    le = {l: list(range(CFG["num_experts"]))
          for l in range(1, CFG["num_hidden_layers"] + 1)}
    tensors, _ = SelectiveLoader(WeightIndex(d)).load(
        qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=True, with_lm_head=True)),
        dtype=torch.float32)
    mine = TorchSegmentModel(TorchModelConfig.from_hf(CFG), le, tensors)
    return ref, mine, d


# --------------------------------------------------------------------------- #
# 1. prefill
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ids", [
    [3],                                  # 单 token
    [3, 17, 200, 5, 88, 41],
    list(range(1, 33)),                   # 长一点，因果 mask 更容易露馅
])
def test_prefill_matches_transformers(pair, ids) -> None:
    """**本仓库最要紧的一条数值断言。**

    attention 缩放、因果 mask、GQA 的 head 复制、RoPE 的位置与旋转约定、
    QK-norm 的位置、路由的 softmax→topk→重归一顺序、门控权重乘的位置 ——
    这一条同时钉住上面全部。
    """
    ref, mine, _ = pair
    with torch.no_grad():
        want = ref(torch.tensor([ids])).logits[0]
        h, _ = mine.forward(f"p{len(ids)}", mine.embed_tokens(ids))
        got = mine.logits(h)
    assert got.shape == want.shape
    assert (got - want).abs().max() < TOL
    assert bool((got.argmax(-1) == want.argmax(-1)).all())


def test_hidden_states_match_layer_by_layer(pair) -> None:
    """逐层比中间结果 —— 只比最终 logits 的话，某一层的错可能被后面掩掉。

    口径注意：HF 的 `hidden_states` 有 n_layers+1 项，**最后一项是过了
    `model.norm` 的**（前面几项都是层的原始输出）。不知道这一点会看到
    「前几层全对、最后一层差一大截」，然后跑去改一个没坏的实现。
    """
    from p2pmoe.runtime.torch_model import _rms_norm

    ref, mine, _ = pair
    ids = [3, 17, 200, 5]
    with torch.no_grad():
        want = ref(torch.tensor([ids]), output_hidden_states=True).hidden_states
        assert len(want) == len(mine.layers) + 1
        x = mine.embed_tokens(ids)
        assert (x - want[0][0]).abs().max() < TOL, "词嵌入就对不上"
        for i, l in enumerate(mine.layers, start=1):
            x, _ = mine.blocks[l].forward(x, {})
            got = (_rms_norm(x, mine.final_norm, mine.cfg.rms_eps)
                   if i == len(mine.layers) else x)
            assert (got - want[i][0]).abs().max() < TOL, f"第 {l} 层输出对不上"


# --------------------------------------------------------------------------- #
# 2. decode（KV cache 是另一条代码路径）
# --------------------------------------------------------------------------- #
def test_decode_with_kv_cache_matches(pair) -> None:
    """贪心生成 8 步，两边各用各的 KV cache，token 序列必须一模一样。

    decode 的 bug 有个特征：prefill 全对、第一个 token 也对，从第二个开始漂 ——
    位置偏移错一位、cache 拼接顺序反了都会这样。所以要连着走好几步。
    """
    ref, mine, _ = pair
    prompt = [3, 17, 200, 5]
    with torch.no_grad():
        h, _ = mine.forward("d", mine.embed_tokens(prompt))
        my_tok = int(mine.logits(h[-1:])[0].argmax())
        r = ref(torch.tensor([prompt]), use_cache=True)
        ref_tok = int(r.logits[0, -1].argmax())
        past = r.past_key_values

        mine_seq, ref_seq, worst = [my_tok], [ref_tok], 0.0
        for _ in range(7):
            h, _ = mine.forward("d", mine.embed_tokens([my_tok]))
            my_lg = mine.logits(h[-1:])[0]
            my_tok = int(my_lg.argmax())
            r = ref(torch.tensor([[ref_tok]]), past_key_values=past, use_cache=True)
            ref_lg = r.logits[0, -1]
            past, ref_tok = r.past_key_values, int(ref_lg.argmax())
            worst = max(worst, float((my_lg - ref_lg).abs().max()))
            mine_seq.append(my_tok)
            ref_seq.append(ref_tok)

    assert mine_seq == ref_seq, f"生成分叉了：{mine_seq} vs {ref_seq}"
    assert worst < TOL


def test_decode_equals_prefill_of_the_same_prefix(pair) -> None:
    """自洽性：逐 token 喂进去，与一次性 prefill 整段，末位必须相同。"""
    _, mine, _ = pair
    ids = [3, 17, 200, 5, 88]
    with torch.no_grad():
        h1, _ = mine.forward("a", mine.embed_tokens(ids))
        one_shot = mine.logits(h1[-1:])[0]
        for t in ids:
            h2, _ = mine.forward("b", mine.embed_tokens([t]))
        step = mine.logits(h2[-1:])[0]
    assert (one_shot - step).abs().max() < TOL


# --------------------------------------------------------------------------- #
# 3. 分段跑与整段跑等价 —— 这条才是本方案的前提
# --------------------------------------------------------------------------- #
def test_splitting_across_segments_changes_nothing(pair) -> None:
    """把层切成前段/后段两截分别跑，结果必须和一整段跑一样。

    整套方案建立在这条上：层区间连续切分之后，把 hidden state 交给下一段
    继续算 —— 与在一台机器上从头跑到尾**在数值上是同一件事**。
    """
    ref, _, d = pair
    n_layers, n_experts = CFG["num_hidden_layers"], CFG["num_experts"]
    ids = [3, 17, 200, 5]
    l0 = 2

    def build(layers, embed, head):
        le = {l: list(range(n_experts)) for l in layers}
        tensors, _ = SelectiveLoader(WeightIndex(d)).load(
            qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=embed,
                                  with_lm_head=head)), dtype=torch.float32)
        return TorchSegmentModel(TorchModelConfig.from_hf(CFG), le, tensors)

    front = build(range(1, l0 + 1), True, False)
    back = build(range(l0 + 1, n_layers + 1), False, True)
    with torch.no_grad():
        y, _ = front.forward("s", front.embed_tokens(ids))
        z, _ = back.forward("s", y)
        got = back.logits(z)
        want = ref(torch.tensor([ids])).logits[0]
    assert (got - want).abs().max() < TOL


def test_the_wire_roundtrip_does_not_break_parity(pair) -> None:
    """跨段的 hidden state 会被降到 float32 走线 —— 确认这不影响结论。

    线上 payload 恒为 float32（见 wire._as_wire_array）。fp32 跑的时候这是恒等的；
    真机用 bf16 时这里会有真实的精度损失，但那是**接口定义**，不是 bug。
    """
    import numpy as np

    from p2pmoe.runtime.wire import _as_wire_array

    ref, _, d = pair
    n_layers, n_experts = CFG["num_hidden_layers"], CFG["num_experts"]
    ids = [3, 17, 200, 5]
    l0 = 2

    def build(layers, embed, head):
        le = {l: list(range(n_experts)) for l in layers}
        tensors, _ = SelectiveLoader(WeightIndex(d)).load(
            qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=embed,
                                  with_lm_head=head)), dtype=torch.float32)
        return TorchSegmentModel(TorchModelConfig.from_hf(CFG), le, tensors)

    front = build(range(1, l0 + 1), True, False)
    back = build(range(l0 + 1, n_layers + 1), False, True)
    with torch.no_grad():
        y, _ = front.forward("w", front.embed_tokens(ids))
        wire = np.frombuffer(_as_wire_array(y).tobytes(),
                             dtype=np.float32).reshape(y.shape)   # 过一遍线
        z, _ = back.forward("w", wire)
        got = back.logits(z)
        want = ref(torch.tensor([ids])).logits[0]
    assert (got - want).abs().max() < TOL


# --------------------------------------------------------------------------- #
# 4. 只驻留子集时的偏差有多大 —— 这是近似，不是 bug
# --------------------------------------------------------------------------- #
def test_a_subset_covering_the_routed_experts_is_lossless(pair) -> None:
    """驻留集只要包含实际被路由到的专家，输出与全量**逐位一致**。

    「只驻留子集」本身不是近似 —— 近似发生在被路由到的专家不在本地时（drop-expert）。
    这条把两者分开。
    """
    ref, mine, d = pair
    ids = [3, 17]
    with torch.no_grad():
        # 先用全量跑一遍，记下每层实际被路由到的专家
        x = mine.embed_tokens(ids)
        routed: dict[int, set[int]] = {}
        for l in mine.layers:
            x_prev = x
            x, st = mine.blocks[l].forward(x, {})
            routed[l] = {e for e in range(CFG["num_experts"]) if st.hist[e] > 0}
        le = {l: sorted(routed[l]) for l in mine.layers}
        assert any(len(v) < CFG["num_experts"] for v in le.values()), (
            "这个输入把所有专家都用上了，测不出「子集」")

        tensors, _ = SelectiveLoader(WeightIndex(d)).load(
            qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=True,
                                  with_lm_head=True)), dtype=torch.float32)
        thin = TorchSegmentModel(TorchModelConfig.from_hf(CFG), le, tensors)
        h, _ = thin.forward("t", thin.embed_tokens(ids))
        got = thin.logits(h)
        want = ref(torch.tensor([ids])).logits[0]
    assert (got - want).abs().max() < TOL


# --------------------------------------------------------------------------- #
# 5. 缺专家时的三种补救策略
# --------------------------------------------------------------------------- #
POLICIES = ("drop", "drop_noscale", "local_topk")


def _build(d, layer_experts, policy):
    tensors, _ = SelectiveLoader(WeightIndex(d)).load(
        qwen_moe_keys(KeyPlan(layer_experts=layer_experts, with_embed=True,
                              with_lm_head=True)), dtype=torch.float32)
    return TorchSegmentModel(TorchModelConfig.from_hf(CFG), layer_experts, tensors,
                             miss_policy=policy)


@pytest.mark.parametrize("policy", POLICIES)
def test_no_miss_means_all_policies_equal_the_reference(pair, policy) -> None:
    """**最要紧的一条**：没有缺失时，三种策略必须都退化成精确计算。

    补救策略只该在「路由要的专家不在本地」时起作用。要是它在全装时也改变了
    结果，那就不是补救，是 bug —— 而且会把「驻留集覆盖实际路由时逐位一致」
    这条性质毁掉，整套方案的可行性论证就没了。
    """
    ref, mine, d = pair
    ids = [3, 17, 200, 5]
    le = {l: list(range(CFG["num_experts"])) for l in mine.layers}
    m = _build(d, le, policy)
    with torch.no_grad():
        h, st = m.forward("n", m.embed_tokens(ids))
        got = m.logits(h)
        want = ref(torch.tensor([ids])).logits[0]
    assert st.miss_token_layer == 0
    assert (got - want).abs().max() < TOL


@pytest.mark.parametrize("policy", POLICIES)
def test_a_subset_covering_the_routing_is_still_lossless(pair, policy) -> None:
    """驻留集包含实际被路由到的专家时，三种策略也都必须精确。"""
    ref, mine, d = pair
    ids = [3, 17]
    with torch.no_grad():
        x = mine.embed_tokens(ids)
        routed = {}
        for l in mine.layers:
            x, st = mine.blocks[l].forward(x, {})
            routed[l] = sorted(e for e in range(CFG["num_experts"]) if st.hist[e] > 0)
        m = _build(d, routed, policy)
        h, _ = m.forward("c", m.embed_tokens(ids))
        want = ref(torch.tensor([ids])).logits[0]
    assert (m.logits(h) - want).abs().max() < TOL


def test_stats_always_report_the_true_routing(pair) -> None:
    """**统计口径与计算口径必须分开。**

    直方图与 miss 率要反映「路由真正想要谁」，而不是「我们最后用了谁」。
    否则画像只会不断确认已有的驻留集（「换一批专家会不会更好」就问不出来了），
    通道二也会因为看不见缺失而永远不报警。

    口径注意：只能**逐层、同一输入**比。整段前向里第 2 层的输入已经被第 1 层的
    策略改过了，那时路由本就该不同 —— 那不是观测口径被污染，是下游真的变了。
    """
    _, mine, d = pair
    ids = [3, 17, 200, 5]
    le = {l: [0, 1] for l in mine.layers}      # 故意只留 2 个，制造大量缺失
    x = mine.embed_tokens(ids)                 # 同一个输入喂给三种策略的同一层
    layer = mine.layers[0]
    stats = {}
    for p in POLICIES:
        m = _build(d, le, p)
        with torch.no_grad():
            _, stats[p] = m.blocks[layer].forward(x, {})
    base = stats["drop"]
    assert base.miss_token_layer > 0, "这个驻留集应该造成缺失，否则测不到东西"
    for p in POLICIES[1:]:
        assert stats[p].miss_token_layer == base.miss_token_layer
        assert stats[p].miss_mass == pytest.approx(base.miss_mass)
        assert (stats[p].hist == base.hist).all()


def test_local_topk_always_uses_k_experts(pair) -> None:
    """`local_topk` 的定义特征：永远凑够 k 个，不会退化成「只走残差」。

    drop 在 top-k 全缺时该层 FFN 输出为零 —— 那是对残差流的一个大扰动。
    """
    _, mine, d = pair
    ids = [3, 17, 200, 5]
    le = {l: [0, 1] for l in mine.layers}
    with torch.no_grad():
        drop = _build(d, le, "drop")
        loc = _build(d, le, "local_topk")
        hd, _ = drop.forward("x", drop.embed_tokens(ids))
        hl, _ = loc.forward("y", loc.embed_tokens(ids))
    assert not torch.allclose(hd, hl), "缺失这么多时两种策略不该给出同样的结果"


def test_noscale_shrinks_instead_of_amplifying(pair) -> None:
    """`drop_noscale` 不重归一：丢得越多，FFN 贡献越小，越接近只走残差。

    重归一等于宣称「路由本来就只想要剩下这几个」—— 那不是事实，而且会把
    一个低概率专家的输出放大到权重 1。
    """
    _, mine, d = pair
    ids = [3, 17]
    le = {l: [0, 1] for l in mine.layers}
    with torch.no_grad():
        x = mine.embed_tokens(ids)
        blocks = {}
        for p in ("drop", "drop_noscale"):
            m = _build(d, le, p)
            y, _ = m.blocks[m.layers[0]].forward(x, {})
            blocks[p] = y
        # 同一层：不缩放的那版离「纯残差」更近
        attn_only, _ = _build(d, {mine.layers[0]: [0, 1]}, "drop").blocks[
            mine.layers[0]].forward(x, {})
    d_norm = float((blocks["drop"] - x).norm())
    d_soft = float((blocks["drop_noscale"] - x).norm())
    assert d_soft < d_norm, "不重归一应该给出更保守（更接近残差）的更新"
