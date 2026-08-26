"""连接断在中途时，从断点续 —— 不是从头再来。

「小文件过得去、大文件过不去」是个常见形状：TLS 检查型代理掐长连接、
MTU 黑洞丢大包、CDN 重置，都会这样。它们的共同点是**总能传一段**。

一次性 `urlopen().read()` 遇到这种链路是死循环：收了 8MB 断掉 → 全部作废 →
重来 → 大概率断在同一个地方。按已收字节续传就能爬完，而且不必先弄清
到底是哪一种原因。
"""

from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import RangeNotSupported, Source
from p2pmoe.deploy.serve_weights import WeightServer

BLOB = bytes(range(256)) * 1000       # 256000 字节


@pytest.fixture
def src(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(BLOB)
    s = WeightServer(tmp_path, "127.0.0.1", 0).start()
    yield Source(base_url=f"http://127.0.0.1:{s.port}"), s
    s.stop()


class Cutter:
    """把每次响应截断到 n 字节，模拟「传到一半断」。"""

    def __init__(self, monkeypatch, source: Source, n: int, fail_times: int = 99):
        self.n, self.left, self.calls = n, fail_times, 0
        real = Source._get

        def wrapped(s, name, lo, hi, *, ranged):
            self.calls += 1
            data, status = real(s, name, lo, hi, ranged=ranged)
            if self.left > 0 and len(data) > self.n:
                self.left -= 1
                return data[:self.n], status
            return data, status

        monkeypatch.setattr(Source, "_get", wrapped)


def test_a_truncated_whole_file_read_resumes(src, monkeypatch) -> None:
    s, _ = src
    c = Cutter(monkeypatch, s, n=40_000)
    assert s.read("m.safetensors") == BLOB
    assert c.calls > 1, "一次就读完了 —— 没走到续传"


def test_a_truncated_range_read_resumes(src, monkeypatch) -> None:
    """逐张量拉取走的是这条路：要的是 [lo, hi)，断了要接着要剩下的。"""
    s, _ = src
    Cutter(monkeypatch, s, n=300)
    assert s.read("m.safetensors", 1000, 9000) == BLOB[1000:9000]


def test_the_resumed_bytes_are_stitched_in_order(src, monkeypatch) -> None:
    """**错位比失败更糟** —— 拼错顺序不会报错，只会得到一份坏权重。"""
    s, _ = src
    Cutter(monkeypatch, s, n=137)
    got = s.read("m.safetensors", 5000, 25000)
    assert got == BLOB[5000:25000]


def test_it_never_returns_more_than_asked(src, monkeypatch) -> None:
    """续传时的 `bytes=lo-` 可能带回多余的尾巴 —— 必须切掉。"""
    s, _ = src
    Cutter(monkeypatch, s, n=64)
    assert len(s.read("m.safetensors", 100, 200)) == 100


def test_a_transient_connection_error_is_retried(src, monkeypatch) -> None:
    s, _ = src
    n = {"c": 0}
    real = Source._get

    def flaky(self, name, lo, hi, *, ranged):
        n["c"] += 1
        if n["c"] <= 2:
            raise urllib.error.URLError("connection reset by peer")
        return real(self, name, lo, hi, ranged=ranged)

    monkeypatch.setattr(Source, "_get", flaky)
    monkeypatch.setattr("p2pmoe.deploy.fetch.time.sleep", lambda *_: None)
    assert s.read("m.safetensors", 0, 500) == BLOB[:500]


def test_a_permanently_dead_link_still_raises(src, monkeypatch) -> None:
    """续传不是无限重试 —— 真的连不上要报出来，而不是静静返回半份数据。"""
    s, _ = src

    def dead(self, name, lo, hi, *, ranged):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(Source, "_get", dead)
    monkeypatch.setattr("p2pmoe.deploy.fetch.time.sleep", lambda *_: None)
    with pytest.raises(urllib.error.URLError):
        s.read("m.safetensors", 0, 100)


def test_range_unsupported_is_still_detected(src, monkeypatch) -> None:
    """续传建立在 Range 之上 —— 不支持 Range 的源仍然要在第一次就被挡下，
    而不是被当成「断了，续一下」然后无限转圈。"""
    s, _ = src
    real = Source._get

    def ignore_range(self, name, lo, hi, *, ranged):
        data, _ = real(self, name, None, None, ranged=False)
        return data, 200          # 装作无视了 Range

    monkeypatch.setattr(Source, "_get", ignore_range)
    monkeypatch.setattr(Source, "_get", lambda self, name, lo, hi, *, ranged: (
        (_ for _ in ()).throw(RangeNotSupported("不支持 Range 请求"))))
    with pytest.raises(RangeNotSupported):
        s.read("m.safetensors", 0, 100)


def test_a_local_source_is_untouched(tmp_path) -> None:
    """本地目录不走网，续传逻辑不该碰它。"""
    (tmp_path / "m.safetensors").write_bytes(BLOB)
    s = Source(local=str(tmp_path))
    assert s.read("m.safetensors", 10, 20) == BLOB[10:20]
    assert s.read("m.safetensors") == BLOB
