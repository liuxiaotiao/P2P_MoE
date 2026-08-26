"""文本层：编解码、增量解码、停止条件、以及它与协调器的接线。

这一层的 bug 有个共同特征 —— **不会抛异常，只会让输出看起来像模型坏了**：
少一个 BOS、漏套对话模板、把半个 UTF-8 字符当成一个 token 解码。所以这里的
断言基本都在比对「跟一次性解码的结果是否逐字相同」。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.runtime.coordinator import Coordinator
from p2pmoe.runtime.text import Detokenizer, StopSpec, TextIO, Tokenizer, load_stop_spec

from test_serving_loop import FakeManifest, FakePool

pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tokdir(tmp_path_factory) -> Path:
    """只造 tokenizer，不造权重 —— 这一层测试不需要 torch。"""
    from p2pmoe.sim.fake_checkpoint import write_fake_tokenizer

    d = tmp_path_factory.mktemp("tok")
    write_fake_tokenizer(d, vocab_size=512)
    (d / "config.json").write_text(json.dumps({"eos_token_id": 0}), encoding="utf-8")
    (d / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 2}), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def tok(tokdir) -> Tokenizer:
    return Tokenizer.from_model_dir(tokdir)


# --------------------------------------------------------------------------- #
# 1. 编解码
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "hello world",
    "the quick brown fox",
    "你好，世界",
    "龍",                      # 训练语料里没有 → 被切成裸字节
    "mixed 中英 123 !@#",
    "",
])
def test_round_trip(tok, text) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_missing_tokenizer_says_what_to_do(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        Tokenizer.from_model_dir(tmp_path)


# --------------------------------------------------------------------------- #
# 2. 增量解码 —— 这一节是整个文件的重点
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "hello world", "你好，世界", "龍龍龍", "a你b好c", "front 前段 back 后段",
])
def test_incremental_equals_one_shot(tok, text) -> None:
    """逐 token 流式吐出来的，必须和一次性 decode 逐字相同。"""
    ids = tok.encode(text)
    d = tok.stream()
    streamed = "".join(d.push(i) for i in ids) + d.flush()
    assert streamed == tok.decode(ids) == text


def test_partial_utf8_is_held_back_not_mangled(tok) -> None:
    """半个汉字不能当成一个字符解码 —— 那会吐出 U+FFFD，而且再也补不回来。"""
    ids = tok.encode("龍")
    assert len(ids) > 1, "这个字应该被切成多个裸字节，否则这条测试测不到东西"
    d = tok.stream()
    out = [d.push(i) for i in ids]
    assert out[:-1] == [""] * (len(ids) - 1)     # 字节没凑齐，一个字都不吐
    assert out[-1] == "龍"                       # 最后一个字节到齐，整字吐出


def test_flush_surfaces_a_truncated_character(tok) -> None:
    """截断在多字节字符中间时，收尾吐一个替换符 —— 比静默吞掉一段文本诚实。"""
    ids = tok.encode("龍")
    d = tok.stream()
    for i in ids[:-1]:
        assert d.push(i) == ""
    assert d.flush() != ""                       # 攒着的半个字符被吐出来了


def test_text_property_tracks_everything_emitted(tok) -> None:
    ids = tok.encode("你好，世界")
    d = tok.stream()
    parts = [d.push(i) for i in ids]
    assert "".join(parts) == d.text


# --------------------------------------------------------------------------- #
# 3. 停止条件
# --------------------------------------------------------------------------- #
def test_generation_config_and_config_are_both_read(tokdir) -> None:
    """两个文件里的 eos 可以不同（Qwen3 就不同），两个都得认。

    只读 config.json 的话，对话模型永远等不到那个它其实已经吐出来的结束符。
    """
    spec = load_stop_spec(tokdir)
    assert spec.ids == frozenset({0, 2})


def test_stop_spec_handles_a_list_of_eos(tmp_path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [11, 22]}), encoding="utf-8")
    assert load_stop_spec(tmp_path).ids == frozenset({11, 22})


def test_missing_files_are_not_an_error(tmp_path) -> None:
    assert not load_stop_spec(tmp_path)


def test_stop_strings_are_substring_matches() -> None:
    s = StopSpec(strings=("\n\n", "<END>"))
    assert s.hit_text("abc\n\ndef") == "\n\n"
    assert s.hit_text("abc") is None


# --------------------------------------------------------------------------- #
# 4. chat template
# --------------------------------------------------------------------------- #
def test_chat_template_renders_roles_and_generation_prompt(tok) -> None:
    out = tok.apply_chat_template([{"role": "user", "content": "hi"}])
    assert "<|im_start|>user" in out and "hi" in out
    assert out.rstrip("\n").endswith("<|im_start|>assistant")


def test_generation_prompt_can_be_turned_off(tok) -> None:
    out = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                  add_generation_prompt=False)
    assert "<|im_start|>assistant" not in out


def test_chat_encoding_does_not_add_a_second_bos(tok) -> None:
    """模板已经把角色标记写全了，encoder 不该再加特殊 token。"""
    msgs = [{"role": "user", "content": "hi"}]
    rendered = tok.apply_chat_template(msgs)
    assert tok.encode_chat(msgs) == tok.encode(rendered, add_special=False)


def test_template_absent_is_an_explicit_error(tokdir, tmp_path) -> None:
    """基座模型没有模板 —— 报错并说清楚，而不是悄悄退化成 completion。"""
    bare = Tokenizer(Tokenizer.from_model_dir(tokdir)._t, config={})
    with pytest.raises(ValueError, match="chat_template"):
        bare.apply_chat_template([{"role": "user", "content": "hi"}])


def test_textio_chat_mode_wraps_the_prompt(tokdir) -> None:
    raw = TextIO.from_model_dir(tokdir, chat=False)
    chat = TextIO.from_model_dir(tokdir, chat=True, system="be brief")
    assert len(chat.encode_prompt("hi")) > len(raw.encode_prompt("hi"))
    assert "be brief" in chat.tok.decode(chat.encode_prompt("hi"), skip_special=False)


# --------------------------------------------------------------------------- #
# 5. 接到协调器上
# --------------------------------------------------------------------------- #
def make_coord(tokdir, **kw) -> tuple[Coordinator, FakePool, list]:
    seen: list[tuple[str, str]] = []
    c = Coordinator(FakeManifest(n_front=1, backs={"X": 1}),
                    baselines={"X": 0.1}, priors={"X": 1.0}, alarm_factor=99.0,
                    textio=TextIO.from_model_dir(tokdir, **kw),
                    on_text=lambda rec, d: seen.append((rec.req, d)))
    c.pool = FakePool()
    c.max_tokens = 50
    return c, c.pool, seen


def feed(c: Coordinator, req: str, token: int) -> None:
    c._on({"type": "token", "req": req, "node": "b", "segment": "BX0",
           "phase": "decode", "token": token,
           "back_stats": {"hist": [], "ntl": 3, "miss": 0, "mass": 0.0}})


def test_submit_takes_text_and_encodes_it(tokdir) -> None:
    c, _, _ = make_coord(tokdir)
    r = c.submit("r0", text="hello world")
    assert r.ids == c.text.tok.encode("hello world")
    assert r.prompt == "hello world"


def test_text_and_ids_are_mutually_exclusive(tokdir) -> None:
    c, _, _ = make_coord(tokdir)
    with pytest.raises(ValueError, match="二选一"):
        c.submit("r0", [1, 2], text="hi")


def test_text_without_a_tokenizer_is_refused() -> None:
    c = Coordinator(FakeManifest(n_front=1, backs={"X": 1}),
                    baselines={}, priors={})
    c.pool = FakePool()
    with pytest.raises(ValueError, match="没配 tokenizer"):
        c.submit("r0", text="hi")


def test_generated_text_streams_and_matches_a_one_shot_decode(tokdir) -> None:
    c, _, seen = make_coord(tokdir)
    r = c.submit("r0", text="hi")
    ids = c.text.tok.encode("你好，世界")
    for i in ids:
        feed(c, "r0", i)
    assert r.text == c.text.tok.decode(ids)
    assert "".join(d for _, d in seen) == r.text      # 流式增量拼起来就是全文


def test_eos_stops_and_is_not_part_of_the_output(tokdir) -> None:
    """EOS 是控制符不是内容 —— 既不进 tokens 也不进 text。"""
    c, _, _ = make_coord(tokdir)
    r = c.submit("r0", text="hi")
    for i in c.text.tok.encode("hello"):
        feed(c, "r0", i)
    n_before, text_before = len(r.tokens), r.text
    feed(c, "r0", 2)                                  # generation_config 的 eos
    assert r.done.is_set() and r.stop_reason == "eos"
    assert len(r.tokens) == n_before and r.text == text_before


def test_stop_string_ends_the_request(tokdir) -> None:
    c = Coordinator(FakeManifest(n_front=1, backs={"X": 1}),
                    baselines={}, priors={},
                    textio=TextIO.from_model_dir(tokdir, stop_strings=("world",)))
    c.pool = FakePool()
    c.max_tokens = 50
    r = c.submit("r0", text="hi")
    for i in c.text.tok.encode("hello world and more"):
        if r.done.is_set():
            break
        feed(c, "r0", i)
    assert r.stop_reason == "stop_string"
    assert "world" in r.text


def test_max_tokens_still_bounds_a_runaway(tokdir) -> None:
    """EOS 是「模型说完了」，max_tokens 是预算 —— 两件事，都要有。"""
    c, _, _ = make_coord(tokdir)
    c.max_tokens = 3
    r = c.submit("r0", text="hi")
    for _ in range(10):
        if r.done.is_set():
            break
        feed(c, "r0", 80)                             # 一个普通 token，永远不是 EOS
    assert r.stop_reason == "max_tokens"
    assert len(r.tokens) == 3


def test_id_mode_still_works_without_a_tokenizer() -> None:
    """不配文本层就是 id 进 id 出 —— toy 模型与协议测试都走这条路。"""
    c = Coordinator(FakeManifest(n_front=1, backs={"X": 1}),
                    baselines={}, priors={})
    c.pool = FakePool()
    c.max_tokens = 2
    r = c.submit("r0", [1, 2, 3])
    for _ in range(2):
        feed(c, "r0", 7)
    assert r.done.is_set() and r.text == "" and r.tokens == [7, 7]


# --------------------------------------------------------------------------- #
# 6. 节点侧：EOS 抄近路，但节点仍然不需要 tokenizer
# --------------------------------------------------------------------------- #
def _bare_back_tail(stop_ids: list[int]):
    """搭一个最小的后段 tail，只为观察它发不发 `loop`。"""
    import threading

    import numpy as np

    from p2pmoe.runtime.model import ToyMoEConfig
    from p2pmoe.runtime.node import NodeConfig, NodeServer

    n = object.__new__(NodeServer)
    n.me = "b0"
    n._lock = threading.Lock()
    n._reqs = {}
    n.pool = FakePool()
    n.reports = []
    n._report = n.reports.append
    n.mcfg = ToyMoEConfig()
    n.cfg = NodeConfig(
        node_id="b0", role="back:X", segment="BX0", layer_experts={},
        next_hop=None, seg_head="b0", is_head=True, is_tail=True,
        peers={}, links={}, coordinator=("127.0.0.1", 1), model={},
        stop_ids=list(stop_ids),
    )
    return n, np.zeros((1, n.mcfg.d_model), dtype=np.float32)


def test_node_skips_the_loop_when_it_samples_a_stop_token() -> None:
    """采到 EOS 就别再绕环了 —— 否则白转一整圈只为算一个马上要丢的 token。"""
    from p2pmoe.runtime.model import MoEStats

    n, y = _bare_back_tail(stop_ids=[])
    n._back_tail("r0", "decode", y, MoEStats.zeros(n.mcfg.n_experts), "f0")
    sampled = n.reports[-1]["token"]
    assert n.pool.of_type("loop"), "普通 token 应该正常绕环"

    n2, y2 = _bare_back_tail(stop_ids=[sampled])
    n2._back_tail("r0", "decode", y2, MoEStats.zeros(n2.mcfg.n_experts), "f0")
    assert not n2.pool.of_type("loop"), "采到停止 token 后不该再发 loop"
    assert n2.reports[-1]["stop"] is True     # 但 token 照报，收尾由协调器做
