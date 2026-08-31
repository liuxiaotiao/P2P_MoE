"""跑的是 toy 模型时，必须说得足够响。

toy 模型**不加载任何下载的权重**。于是「权重根本没下成功」和「跑的是 toy」
在日志里长得一模一样 —— 都顺利跑出 token、都有漂亮的时延数字。

而两者差一个量级：toy 是 8 层 × 32 专家的 numpy 玩具，逐 token 十几毫秒；
真的 80B 跨链路是百毫秒级。拿 toy 的数字当测量结果，会得出完全错误的结论 ——
而且没有任何东西会提醒你。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "p2pmoe" / "deploy" / "control.py").read_text(encoding="utf-8")


def test_the_toy_path_warns() -> None:
    i = SRC.index("if not args.model_dir:")
    head = SRC[i:i + 1200]
    assert "log.warning" in head, "toy 分支没有任何警告"


def test_the_warning_says_it_is_not_real_weights() -> None:
    """光说「toy」不够 —— 要说清它**不加载真实权重**，那才是被误读的点。"""
    i = SRC.index("if not args.model_dir:")
    head = SRC[i:i + 1200]
    assert "真实权重" in head or "不加载任何真实权重" in head


def test_the_warning_says_how_to_run_the_real_thing() -> None:
    i = SRC.index("if not args.model_dir:")
    head = SRC[i:i + 1200]
    assert "--model-dir" in head and "--profile" in head


def test_the_summary_line_carries_the_backend() -> None:
    """**这条最关键。** 汇总行是会被复制进报告的那一行，
    脱离了「跑的是什么」，那些毫秒数就是误导。"""
    for m in re.finditer(r'log\.info\("汇总', SRC):
        line = SRC[m.start():m.start() + 200]
        assert "%s" in line and "who" in SRC[m.start():m.start() + 400], \
            "汇总行没带后端"


def test_a_toy_run_is_labelled_as_a_toy_in_the_summary() -> None:
    i = SRC.index('who = (')
    blk = SRC[i:i + 300]
    assert "玩具" in blk or "toy" in blk
    assert "numpy" in blk, "靠 backend 判断，而不是靠别的间接信号"


def test_saved_results_record_the_backend() -> None:
    """落盘的 JSON 也要带 —— 几周后回看时，那是唯一还在的上下文。"""
    assert '"backend": setup.backend' in SRC
