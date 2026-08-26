"""文本进出 —— tokenizer、增量解码、停止条件。

**这一层属于控制面，不属于数据面。** 节点自始至终只见 token id：
head(f) 拿 id 查嵌入表，tail(b) 采样出 id，环里传的是 hidden state。
没有任何一台推理节点需要装 tokenizer。

这不是为了省那几 MB，是因为它决定了依赖怎么分层：

    requirements.txt          numpy                两边都要
    requirements-control.txt  + tokenizers、jinja2 控制机（文本进出）
    requirements-node.txt     + torch、safetensors 推理节点（张量）

两份互不包含。控制机不碰张量，节点不碰文本 —— `test_requirements.py` 守着这条线。

为什么不用 transformers
-----------------------
只为了 tokenizer 拉进一个几百 MB、带 torch 依赖的包不划算。HF 的 `tokenizers`
是同一套 Rust 实现的独立包，`AutoTokenizer` 底下跑的就是它。差别只有两处，
这里都自己补上了：

* **chat template** —— transformers 用 jinja2 渲染 `tokenizer_config.json` 里的
  模板字符串，这里做同样的事（`apply_chat_template`）；
* **停止 token** —— transformers 从 `generation_config.json` 读，这里也读。

代价：只支持带 `tokenizer.json` 的模型（HF 新格式，Qwen3 系都有）。老的
sentencepiece-only 模型会明确报错并告诉你装 transformers，而不是悄悄退化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["Tokenizer", "Detokenizer", "StopSpec", "load_stop_spec"]

_REPLACEMENT = "�"


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StopSpec:
    """什么时候停。

    `max_tokens` 是**预算**，EOS 是**模型自己说完了** —— 两件事，都要有。
    只有 max_tokens 的话，短回答会被后面一堆废 token 拖着不放；只有 EOS 的话，
    一次跑飞就能把一条通道占到天荒地老。
    """

    ids: frozenset[int] = frozenset()
    strings: tuple[str, ...] = ()
    """文本层面的停止串（如 "\\n\\n"）。EOS 之外的场景才用得上。"""

    def hit_id(self, token: int) -> bool:
        return token in self.ids

    def hit_text(self, text: str) -> str | None:
        """返回被命中的那个停止串；没命中返回 None。"""
        for s in self.strings:
            if s and s in text:
                return s
        return None

    def __bool__(self) -> bool:
        return bool(self.ids or self.strings)


def load_stop_spec(model_dir: str | Path, *, extra: Sequence[str] = ()) -> StopSpec:
    """从 checkpoint 读停止 token。

    两处都要读，且 **generation_config 优先** —— 这是 HF 的既定语义：
    `config.json` 的 `eos_token_id` 是模型架构层面的（Qwen3 是 151643
    `<|endoftext|>`），`generation_config.json` 的才是生成时该用的
    （Qwen3-Instruct 是 151645 `<|im_end|>`）。只读 config.json 的话，
    对话模型永远等不到那个它其实已经吐出来的结束符。
    """
    d = Path(model_dir)
    ids: set[int] = set()

    def take(obj: Mapping | None) -> None:
        if not obj:
            return
        v = obj.get("eos_token_id")
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v if isinstance(x, int))

    for name in ("generation_config.json", "config.json"):
        f = d / name
        if f.exists():
            try:
                take(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
    return StopSpec(ids=frozenset(ids), strings=tuple(extra))


# --------------------------------------------------------------------------- #
class Detokenizer:
    """增量解码：喂一个 id，吐出**新出现**的那段文本。

    为什么不能逐 token 解码再拼接
    -----------------------------
    byte-level BPE 的一个 token 可以是半个 UTF-8 字符（中文常见：一个汉字
    3 字节，可能被切成 2+1）。单独 decode 那半个字节得到的是 U+FFFD 替换符，
    拼起来就是一串乱码 —— 而且**错误不可逆**，后面补上的那半个字节也救不回来。

    所以正确做法是：整段重解，取增量。检测到结尾是 U+FFFD 就说明字节还没凑齐，
    这一步先不吐字，等下一个 token 补上。

    复杂度是 O(n²)（每步重解全序列）。生产级实现（vLLM 等）会维护一个滑动的
    已提交前缀来摊掉它；这里 n 是几十到几千，重解一次是微秒级，不值得为它引入
    一个容易出错的增量状态机。真要跑长上下文再优化。
    """

    def __init__(self, tok: "Tokenizer", *, skip_special: bool = True):
        self.tok = tok
        self.skip_special = skip_special
        self.ids: list[int] = []
        self.text = ""
        """到目前为止已经吐出去的全文。"""

    def push(self, token: int) -> str:
        self.ids.append(int(token))
        whole = self.tok.decode(self.ids, skip_special=self.skip_special)
        if whole.endswith(_REPLACEMENT):
            return ""          # 字节没凑齐，等下一个 token
        delta, self.text = whole[len(self.text):], whole
        return delta

    def extend(self, tokens: Iterable[int]) -> str:
        return "".join(self.push(t) for t in tokens)

    def flush(self) -> str:
        """收尾：把因为字节没凑齐而攒着的部分吐出来（可能含 U+FFFD）。

        生成被截断在一个多字节字符中间时会走到这里 —— 吐一个替换符，
        比静默吞掉一段文本诚实。
        """
        whole = self.tok.decode(self.ids, skip_special=self.skip_special)
        delta, self.text = whole[len(self.text):], whole
        return delta


# --------------------------------------------------------------------------- #
class Tokenizer:
    """HF `tokenizer.json` 的薄封装 + chat template 渲染。"""

    def __init__(self, backend: Any, *, config: Mapping | None = None,
                 model_dir: str | Path | None = None):
        self._t = backend
        self.config = dict(config or {})
        self.model_dir = Path(model_dir) if model_dir else None

    # -- 构造 -------------------------------------------------------------- #
    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Tokenizer":
        d = Path(model_dir)
        f = d / "tokenizer.json"
        if not f.exists():
            raise FileNotFoundError(
                f"{d} 里没有 tokenizer.json。本模块只支持 HF 新格式的 tokenizer；"
                f"老的 sentencepiece-only 模型请装 transformers 用 AutoTokenizer，"
                f"或先用它转出一份 tokenizer.json"
            )
        try:
            from tokenizers import Tokenizer as _HFTok
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "文本进出需要 tokenizers：pip install -r requirements-control.txt"
            ) from e
        cfg_f = d / "tokenizer_config.json"
        cfg = json.loads(cfg_f.read_text(encoding="utf-8")) if cfg_f.exists() else {}
        return cls(_HFTok.from_file(str(f)), config=cfg, model_dir=d)

    # -- 编解码 ------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return self._t.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        return list(self._t.encode(text, add_special_tokens=add_special).ids)

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        return self._t.decode(list(ids), skip_special_tokens=skip_special)

    def stream(self, *, skip_special: bool = True) -> Detokenizer:
        return Detokenizer(self, skip_special=skip_special)

    def id_of(self, token: str) -> int | None:
        return self._t.token_to_id(token)

    # -- chat template ----------------------------------------------------- #
    @property
    def chat_template(self) -> str | None:
        """checkpoint 自带的模板字符串。没有就是 None（基座模型常见）。"""
        t = self.config.get("chat_template")
        if isinstance(t, str):
            return t
        if isinstance(t, list):        # 新格式：[{"name": ..., "template": ...}]
            for entry in t:
                if isinstance(entry, Mapping) and entry.get("name") == "default":
                    return entry.get("template")
            if t and isinstance(t[0], Mapping):
                return t[0].get("template")
        return None

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool = True,
        **extra: Any,
    ) -> str:
        """渲染对话模板 —— 与 transformers 同一份模板、同一套 jinja 设置。

        指令模型**必须**走这里。直接把用户的话喂进去是 completion 语义，
        Qwen3-Instruct 那种模型会当成一段待续写的文本，输出看起来像是坏了，
        其实只是没有被告知「这是一轮对话」。
        """
        tpl = self.chat_template
        if tpl is None:
            raise ValueError(
                "这个 checkpoint 的 tokenizer_config.json 里没有 chat_template"
                "（基座模型通常就没有）—— 用 completion 模式，别套对话格式"
            )
        try:
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "渲染 chat template 需要 jinja2："
                "pip install -r requirements-control.txt"
            ) from e

        def raise_exception(msg: str):
            raise TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda o, **kw: json.dumps(o, ensure_ascii=False, **kw)
        env.globals["raise_exception"] = raise_exception
        ctx: dict[str, Any] = {
            k: v for k, v in self.config.items()
            if k.endswith("_token") and isinstance(v, str)
        }
        ctx.update(
            messages=[dict(m) for m in messages],
            add_generation_prompt=add_generation_prompt,
            tools=None,
        )
        ctx.update(extra)
        return env.from_string(tpl).render(**ctx)

    def encode_chat(self, messages: Sequence[Mapping[str, str]], **kw) -> list[int]:
        """模板渲染后再编码。

        `add_special=False` 是有意的：模板里已经把 BOS / 角色标记写全了，
        再让 encoder 加一遍会多出一个 BOS，模型看到的就不是它训练时的格式。
        """
        return self.encode(self.apply_chat_template(messages, **kw), add_special=False)


# --------------------------------------------------------------------------- #
@dataclass
class TextIO:
    """把 tokenizer 与停止条件打包 —— 协调器只需要拿着这一个对象。"""

    tok: Tokenizer
    stop: StopSpec = field(default_factory=StopSpec)
    chat: bool = False
    system: str | None = None

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, *, chat: bool = False,
                       system: str | None = None,
                       stop_strings: Sequence[str] = ()) -> "TextIO":
        return cls(
            tok=Tokenizer.from_model_dir(model_dir),
            stop=load_stop_spec(model_dir, extra=tuple(stop_strings)),
            chat=chat,
            system=system,
        )

    def encode_prompt(self, text: str) -> list[int]:
        if not self.chat:
            return self.tok.encode(text)
        msgs = ([{"role": "system", "content": self.system}] if self.system else [])
        msgs.append({"role": "user", "content": text})
        return self.tok.encode_chat(msgs)

    def stream(self) -> Detokenizer:
        return self.tok.stream()
