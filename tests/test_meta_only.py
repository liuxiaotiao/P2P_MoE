"""控制机只要 config 与 tokenizer，不要权重。

控制机与节点共用 `--model-dir` 这一个路径，但要的东西不一样：节点要权重
（几十 GB），控制机一个张量都不碰 —— 只用 config 算模型规格、用 tokenizer
编解码文本。所以控制机上那个目录经常是空的（权重是各节点自己拉的，不经过
控制机），而报错必须指向「取 10MB 元数据」而不是「再下一遍 141GB」。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_NEXT, write_fake_checkpoint

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    return Path(write_fake_checkpoint(
        str(tmp_path_factory.mktemp("ck")), dict(TINY_QWEN3_NEXT), seed=0))


def _fetch(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "p2pmoe.deploy.fetch", *args],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)


def test_meta_only_takes_config_and_tokenizer(ckpt, tmp_path) -> None:
    out = tmp_path / "meta"
    r = _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert (out / "config.json").exists()
    assert (out / "tokenizer.json").exists()


def test_meta_only_takes_no_tensors(ckpt, tmp_path) -> None:
    """**这条是重点。** 拿了张量就等于把 141GB 搬到了控制机上。"""
    out = tmp_path / "meta"
    _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    assert not list(out.glob("*.safetensors")), "元数据模式不该有任何权重文件"
    assert not list(out.glob("*.bin"))


def test_meta_only_is_small(ckpt, tmp_path) -> None:
    """真模型上这一坨约 10MB（几乎全是 tokenizer.json），量级不能漂。"""
    out = tmp_path / "meta"
    _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    total = sum(f.stat().st_size for f in out.iterdir())
    assert total < 50 * 1024 * 1024


def test_meta_only_needs_neither_plan_nor_node(ckpt, tmp_path) -> None:
    """控制机手上可能还没有清单 —— 取元数据不该被它挡住。"""
    r = _fetch("--meta-only", "--src", str(ckpt), "--out", str(tmp_path / "m"))
    assert r.returncode == 0


def test_without_meta_only_plan_and_node_are_required(ckpt, tmp_path) -> None:
    """反过来：拉权重那条路少了清单就该拒绝，而不是下一堆不知道给谁的东西。"""
    r = _fetch("--src", str(ckpt), "--out", str(tmp_path / "m"))
    assert r.returncode != 0
    assert "--meta-only" in (r.stderr + r.stdout)


def test_the_config_is_usable_by_the_control_side(ckpt, tmp_path) -> None:
    """取回来的 config 要真能算出模型规格 —— 不然取了也白取。"""
    from p2pmoe.planner.hf_config import model_spec_from_hf

    out = tmp_path / "meta"
    _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    hf = json.loads((out / "config.json").read_text(encoding="utf-8"))
    spec, info = model_spec_from_hf(hf, name="m", ctx_max=2048, dtype_bytes=2)
    assert spec.n_layers == TINY_QWEN3_NEXT["num_hidden_layers"]
    assert spec.n_experts == TINY_QWEN3_NEXT["num_experts"]


def test_a_missing_model_dir_points_at_meta_not_at_refetching_weights() -> None:
    """报错的方向决定了人接下来干什么。指错方向的代价是重下 141GB。"""
    r = subprocess.run(
        [sys.executable, "-m", "p2pmoe.deploy.control", "--agents", "n=1.2.3.4:9101",
         "--advertise", "1.2.3.4", "--model-dir", "/nonexistent/w", "--static", "--once"],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    msg = r.stdout + r.stderr
    assert "meta" in msg
    assert "不要权重" in msg
