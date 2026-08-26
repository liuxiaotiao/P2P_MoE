"""测试集文件：`--prompts-file` 怎么解析。

标注**只用来核对识别对不对**，不影响派发 —— 派发靠前段自己识别，
那才是被测的东西。所以「标注被误读」的后果不是跑错，而是**评分错**：
准确率那一栏会拿错误的真值去比。这比跑错更难发现，所以单独测。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.control import load_cases

TASKS = ("mbpp", "gsm8k")


def w(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cases.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_a_tab_separated_label_is_picked_up(tmp_path) -> None:
    got = load_cases(w(tmp_path, "mbpp\t反转链表\ngsm8k\t48 个朋友\n"), TASKS)
    assert got == [("mbpp", "反转链表"), ("gsm8k", "48 个朋友")]


def test_an_unlabelled_line_is_all_prompt(tmp_path) -> None:
    got = load_cases(w(tmp_path, "就是一句话\n"), TASKS)
    assert got == [(None, "就是一句话")]


def test_a_tab_inside_code_is_not_mistaken_for_a_label(tmp_path) -> None:
    """**这条是这个文件存在的理由。**

    代码类 prompt 里制表符是缩进，不是分隔符。把 `def f():\\t...` 的
    `def f():` 当成 task 名，会让整条 prompt 被腰斩 —— 而且悄无声息。
    """
    line = "def f():\treturn 1"
    got = load_cases(w(tmp_path, line + "\n"), TASKS)
    assert got == [(None, line)], "认不出的前缀 → 整行都是 prompt"


def test_an_unknown_task_name_is_not_a_label(tmp_path) -> None:
    got = load_cases(w(tmp_path, "humaneval\t写个函数\n"), TASKS)
    assert got == [(None, "humaneval\t写个函数")]


def test_comments_and_blank_lines_are_skipped(tmp_path) -> None:
    got = load_cases(w(tmp_path, "# 说明\n\n   \nmbpp\tx\n  # 缩进的注释\n"), TASKS)
    assert got == [("mbpp", "x")]


def test_trailing_whitespace_goes_but_leading_stays(tmp_path) -> None:
    """行尾空白是编辑器留的，行首缩进可能是 prompt 的一部分。"""
    got = load_cases(w(tmp_path, "    缩进的 prompt   \n"), TASKS)
    assert got == [(None, "    缩进的 prompt")]


def test_label_whitespace_is_tolerated(tmp_path) -> None:
    got = load_cases(w(tmp_path, " mbpp \t反转链表\n"), TASKS)
    assert got == [("mbpp", "反转链表")]


def test_a_prompt_can_be_empty_after_the_label(tmp_path) -> None:
    """空 prompt 是采集端的 bug，但解析器不该替它做决定 —— 原样传下去，
    让它在跑的时候暴露成「0 token 的请求」，而不是在这里被悄悄吃掉。"""
    got = load_cases(w(tmp_path, "mbpp\t\n"), TASKS)
    assert got == [("mbpp", "")]


def test_an_empty_file_yields_nothing(tmp_path) -> None:
    assert load_cases(w(tmp_path, "# 全是注释\n\n"), TASKS) == []


def test_order_is_preserved(tmp_path) -> None:
    """--requests 少于条数时只跑前几条 —— 顺序变了就不是同一个子集。"""
    src = "\n".join(f"mbpp\tp{i}" for i in range(20))
    got = load_cases(w(tmp_path, src), TASKS)
    assert [t for _, t in got] == [f"p{i}" for i in range(20)]
