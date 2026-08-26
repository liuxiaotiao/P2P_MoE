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


# --------------------------------------------------------------------------- #
# 「仓库里没有」与「取失败了」是两回事
# --------------------------------------------------------------------------- #
def test_a_missing_tokenizer_is_a_hard_failure(ckpt, tmp_path) -> None:
    """没有 tokenizer.json，控制机编不了 prompt、解不了 token。

    以前这里只是打一句「多数情况无所谓」然后 rc=0 —— 而真正的错要到几步之后
    才发作（--chat 渲染模板时），到时候没人会想到是取元数据那步丢的。
    """
    import shutil

    bad = tmp_path / "ck"
    shutil.copytree(ckpt, bad)
    (bad / "tokenizer.json").unlink()
    r = _fetch("--meta-only", "--src", str(bad), "--out", str(tmp_path / "m"))
    assert r.returncode != 0
    assert "tokenizer.json" in (r.stdout + r.stderr)


def test_a_transient_error_is_retried(ckpt, tmp_path, monkeypatch) -> None:
    """11MB 的 tokenizer.json 是这批里唯一会超时的。重试比让人重跑整条命令便宜。"""
    import urllib.error

    import p2pmoe.deploy.fetch as F

    n = {"c": 0}
    real = F.Source.read

    def flaky(self, name, start=None, end=None):
        if name == "tokenizer.json":
            n["c"] += 1
            if n["c"] < 3:
                raise urllib.error.URLError("timed out")
        return real(self, name, start, end)

    monkeypatch.setattr(F.Source, "read", flaky)
    monkeypatch.setattr(F.time, "sleep", lambda *_: None)
    out = tmp_path / "m"
    assert F.main(["--meta-only", "--src", str(ckpt), "--out", str(out)]) == 0
    assert n["c"] == 3, "该重试到成功"
    assert (out / "tokenizer.json").exists()


def test_a_file_that_truly_is_absent_is_not_retried(ckpt, tmp_path, monkeypatch) -> None:
    """真缺的文件重试只是白等 —— 而等待会被误读成「网络很慢」。"""
    import p2pmoe.deploy.fetch as F

    n = {"c": 0}
    real = F.Source.read

    def count(self, name, start=None, end=None):
        if name == "special_tokens_map.json":
            n["c"] += 1
        return real(self, name, start, end)

    monkeypatch.setattr(F.Source, "read", count)
    monkeypatch.setattr(F.time, "sleep", lambda *_: None)
    F.main(["--meta-only", "--src", str(ckpt), "--out", str(tmp_path / "m")])
    assert n["c"] == 1, f"真缺的文件试了 {n['c']} 次"


# --------------------------------------------------------------------------- #
# 已经在本地的文件：跳过，但要验它真能用
# --------------------------------------------------------------------------- #
def test_an_existing_good_file_is_kept(ckpt, tmp_path) -> None:
    """手工 curl 补上的那一个不能被覆盖 —— 在会掐断的链路上，
    那可能是唯一一次成功。"""
    import shutil

    out = tmp_path / "m"
    out.mkdir()
    shutil.copy(ckpt / "tokenizer.json", out / "tokenizer.json")
    mtime = (out / "tokenizer.json").stat().st_mtime_ns

    r = _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    assert r.returncode == 0
    assert (out / "tokenizer.json").stat().st_mtime_ns == mtime, "被重写了"
    assert "已有" in (r.stdout + r.stderr)


def test_an_html_error_page_is_not_mistaken_for_a_tokenizer(ckpt, tmp_path) -> None:
    """**手工下载最常见的坑**：拿到的是 HTML 错误页（登录墙 / 403 / 镜像提示页）。
    文件存在、大小非零，直到 tokenizer 加载时才炸 —— 那时离原因隔了很远。
    """
    out = tmp_path / "m"
    out.mkdir()
    (out / "tokenizer.json").write_text("<html>403 Forbidden</html>", encoding="utf-8")
    r = _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    assert r.returncode == 0
    assert "读不通" in (r.stdout + r.stderr), "没认出这不是 tokenizer"
    import json as _json
    back = _json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    assert isinstance(back.get("model"), dict), "重取之后应该是真的 tokenizer"


def test_an_empty_file_is_refetched(ckpt, tmp_path) -> None:
    """0 字节通常是上次下载被打断留下的残骸。"""
    out = tmp_path / "m"
    out.mkdir()
    (out / "tokenizer.json").write_bytes(b"")
    assert _fetch("--meta-only", "--src", str(ckpt), "--out", str(out)).returncode == 0
    assert (out / "tokenizer.json").stat().st_size > 0


def test_force_refetches_everything(ckpt, tmp_path) -> None:
    import shutil

    out = tmp_path / "m"
    out.mkdir()
    shutil.copy(ckpt / "tokenizer.json", out / "tokenizer.json")
    r = _fetch("--meta-only", "--src", str(ckpt), "--out", str(out), "--force")
    assert r.returncode == 0
    assert "已有" not in (r.stdout + r.stderr)


def test_a_json_that_is_not_a_tokenizer_is_refetched(ckpt, tmp_path) -> None:
    """合法 JSON 但不是 tokenizer —— 镜像的提示页有时就长这样。"""
    out = tmp_path / "m"
    out.mkdir()
    (out / "tokenizer.json").write_text('{"error":"not found"}', encoding="utf-8")
    r = _fetch("--meta-only", "--src", str(ckpt), "--out", str(out))
    assert "读不通" in (r.stdout + r.stderr)


def test_validation_does_not_import_tokenizers() -> None:
    """节点只见 token id，不该为了校验一个文件多装一套分词器。"""
    src = (ROOT / "p2pmoe" / "deploy" / "fetch.py").read_text(encoding="utf-8")
    assert "from tokenizers" not in src and "import tokenizers" not in src
