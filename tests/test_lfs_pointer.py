"""认出 git-lfs 指针文件。

`GIT_LFS_SKIP_SMUDGE=1 git clone` 只拉仓库结构，每个大文件留一个 130 字节的
指针文本。**目录看起来完全正常** —— 41 个 `.safetensors` 都在，名字对、数量对，
只是每个都是文本。

不认出来的话，错会以 `头长 1936026161 不合理` 的形式出现在解析文件头那一步 ——
离真正的原因隔了三层，而真正要做的只是补一句 `git lfs pull`。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import Source, looks_like_lfs_pointer, read_header

POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
           b"oid sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\n"
           b"size 4294967296\n")

ROOT = Path(__file__).resolve().parent.parent


def test_a_pointer_is_recognised() -> None:
    assert looks_like_lfs_pointer(POINTER[:64])


def test_a_real_safetensors_header_is_not() -> None:
    """safetensors 开头是 8 字节小端 u64 —— 别把真权重误判成指针。"""
    real = (100).to_bytes(8, "little") + b'{"a":{"dtype":"F32"}}'
    assert not looks_like_lfs_pointer(real[:64])


def test_random_bytes_are_not() -> None:
    assert not looks_like_lfs_pointer(bytes(range(64)))


def test_reading_a_pointer_says_what_to_do(tmp_path) -> None:
    """**这条是重点。** 报错要指向 `git lfs pull`，而不是 safetensors 格式。"""
    (tmp_path / "model-00001-of-00041.safetensors").write_bytes(POINTER)
    src = Source(local=str(tmp_path))
    with pytest.raises(ValueError) as e:
        read_header(src, "model-00001-of-00041.safetensors")
    msg = str(e.value)
    assert "git-lfs 指针" in msg
    assert "git lfs pull" in msg
    assert "GIT_LFS_SKIP_SMUDGE" in msg, "要点名是哪条命令造成的"


def test_a_genuinely_broken_file_still_gets_the_old_message(tmp_path) -> None:
    """不是指针的坏文件不该被误导到 lfs 上去。"""
    (tmp_path / "m.safetensors").write_bytes(b"not a safetensors at all, really")
    src = Source(local=str(tmp_path))
    with pytest.raises(ValueError, match="头长"):
        read_header(src, "m.safetensors")


def test_serve_weights_refuses_to_serve_pointers(tmp_path, capsys) -> None:
    """起服务时就拦住 —— 否则 15 台会各自拿到 130 字节文本，
    每台报一次同样费解的错。"""
    from p2pmoe.deploy.serve_weights import main

    for i in (1, 2):
        (tmp_path / f"model-0000{i}-of-00002.safetensors").write_bytes(POINTER)
    rc = main(["--dir", str(tmp_path)])
    assert rc == 2
    out = capsys.readouterr().out + capsys.readouterr().err
    # main 用 logging 输出，pytest 的 caplog 更稳；这里只断返回码与不起服务
    assert rc != 0


def test_serve_weights_accepts_real_shards(tmp_path, monkeypatch) -> None:
    """反向：真权重不能被误拦。"""
    from p2pmoe.deploy import serve_weights as SW

    hdr = b'{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(
        len(hdr).to_bytes(8, "little") + hdr + b"\x00" * 4)

    started = {}

    class FakeSrv:
        port = 9999
        stats = {"bytes": 0, "reqs": 0}

        def __init__(self, *a, **k):
            self.root = tmp_path

        def start(self):
            started["yes"] = True
            return self

        def stop(self):
            started["stopped"] = True

    class Stop:
        def wait(self, *_):
            raise KeyboardInterrupt      # 立刻退出等待循环

    monkeypatch.setattr(SW, "WeightServer", FakeSrv)
    monkeypatch.setattr(SW.threading, "Event", Stop)
    assert SW.main(["--dir", str(tmp_path)]) == 0
    assert started.get("yes"), "真权重被误拦了 —— 服务根本没起来"
    assert started.get("stopped"), "退出时没停服务"
