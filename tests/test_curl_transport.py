"""第二条传输通道：curl。

「curl 能下、Python 下不动」是很常见的一对症状 —— 两者用的**不是同一套 TLS**：
conda 环境自带 OpenSSL，系统 curl 用系统的；版本、cipher、ALPN、代理处理都可能
不同。与其查清是哪一处不同，不如换用那个已经证明能跑的。

这里测的是：curl 那条路取回来的字节要和 urllib 那条**逐字节一致**，
Range 语义要一样，且「对端无视 Range」在两条路上都要被抓出来。
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import RangeNotSupported, Source
from p2pmoe.deploy.serve_weights import WeightServer

pytestmark = pytest.mark.skipif(shutil.which("curl") is None, reason="没有 curl")

BLOB = bytes(range(256)) * 500      # 128000 字节


@pytest.fixture
def srv(tmp_path):
    (tmp_path / "m.safetensors").write_bytes(BLOB)
    s = WeightServer(tmp_path, "127.0.0.1", 0).start()
    yield s
    s.stop()


def _src(srv, transport):
    return Source(base_url=f"http://127.0.0.1:{srv.port}", transport=transport,
                  timeout=20)


# --------------------------------------------------------------------------- #
# 1. 两条路结果一致
# --------------------------------------------------------------------------- #
def test_curl_reads_the_whole_file(srv) -> None:
    assert _src(srv, "curl").read("m.safetensors") == BLOB


def test_curl_reads_a_range(srv) -> None:
    assert _src(srv, "curl").read("m.safetensors", 1000, 2000) == BLOB[1000:2000]


def test_both_transports_agree_byte_for_byte(srv) -> None:
    """**这条是重点。** 换传输不能换出不一样的字节。"""
    a = _src(srv, "urllib").read("m.safetensors", 777, 9999)
    b = _src(srv, "curl").read("m.safetensors", 777, 9999)
    assert a == b == BLOB[777:9999]


def test_curl_handles_an_open_ended_range(srv) -> None:
    s = _src(srv, "curl")
    assert s.read("m.safetensors", len(BLOB) - 20, len(BLOB)) == BLOB[-20:]


# --------------------------------------------------------------------------- #
# 2. auto：urllib 失败就切
# --------------------------------------------------------------------------- #
def test_auto_falls_back_to_curl(srv, monkeypatch, caplog) -> None:
    s = _src(srv, "auto")
    monkeypatch.setattr(Source, "_get_urllib", lambda *a, **k: (_ for _ in ()).throw(
        urllib.error.URLError("handshake operation timed out")))
    with caplog.at_level("WARNING", logger="p2pmoe.fetch"):
        got = s.read("m.safetensors", 0, 100)
    assert got == BLOB[:100]
    assert "curl" in caplog.text


def test_the_switch_is_sticky(srv, monkeypatch) -> None:
    """切过去就别再回头试 urllib —— 每个张量区间都失败一次太贵了
    （41 个分片、上万个张量）。"""
    s = _src(srv, "auto")
    n = {"c": 0}

    def boom(*a, **k):
        n["c"] += 1
        raise urllib.error.URLError("x")

    monkeypatch.setattr(Source, "_get_urllib", boom)
    for lo in (0, 100, 200, 300):
        s.read("m.safetensors", lo, lo + 50)
    assert n["c"] == 1, f"urllib 被试了 {n['c']} 次 —— 切换没粘住"
    assert s.transport == "curl"


def test_range_unsupported_does_not_trigger_a_switch(srv, monkeypatch) -> None:
    """对端无视 Range 是**对端行为**，换传输不会变。
    切过去只会把同一个错再撞一遍，还把真正的原因埋掉。"""
    s = _src(srv, "auto")
    monkeypatch.setattr(Source, "_get_urllib", lambda *a, **k: (_ for _ in ()).throw(
        RangeNotSupported("不支持 Range 请求")))
    with pytest.raises(RangeNotSupported):
        s.read("m.safetensors", 0, 100)
    assert s.transport == "auto", "不该切"


def test_explicit_urllib_never_switches(srv, monkeypatch) -> None:
    """显式指定了传输就该照办 —— 自动切换会让排查变成猜谜。"""
    s = _src(srv, "urllib")
    monkeypatch.setattr(Source, "_get_urllib", lambda *a, **k: (_ for _ in ()).throw(
        urllib.error.URLError("x")))
    with pytest.raises(urllib.error.URLError):
        s.read("m.safetensors", 0, 10)


# --------------------------------------------------------------------------- #
# 3. curl 也要认出「对端无视 Range」
# --------------------------------------------------------------------------- #
def test_curl_detects_a_server_that_ignores_range(tmp_path) -> None:
    """否则「省下载」没发生却无人知晓，而按区间长度切整文件会切出错位的权重。"""
    import http.server
    import socketserver
    import threading

    (tmp_path / "m.safetensors").write_bytes(BLOB)
    h = http.server.SimpleHTTPRequestHandler        # 实测：不支持 Range
    srv = socketserver.TCPServer(("127.0.0.1", 0),
                                 lambda *a: h(*a, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = Source(base_url=f"http://127.0.0.1:{srv.server_address[1]}",
                   transport="curl", timeout=20)
        with pytest.raises(RangeNotSupported):
            s.read("m.safetensors", 0, 100)
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------- #
# 4. 两条传输的异常类型必须一致
# --------------------------------------------------------------------------- #
def test_curl_raises_httperror_on_404(srv) -> None:
    """**两条路必须抛同一种异常。**

    调用方靠 `HTTPError.code` 区分「仓库里没有这个文件」（404，可选文件正常
    缺席，比如 special_tokens_map.json）和「取失败了」。curl 这边抛 OSError
    的话，一个可有可无的文件就能让整轮下载失败 —— 真踩过。
    """
    import urllib.error as ue

    with pytest.raises(ue.HTTPError) as e:
        _src(srv, "curl").read("nope.json")
    assert e.value.code == 404


def test_both_transports_raise_the_same_type_for_404(srv) -> None:
    import urllib.error as ue

    kinds = []
    for t in ("urllib", "curl"):
        try:
            _src(srv, t).read("nope.json")
        except ue.HTTPError as e:
            kinds.append(("HTTPError", e.code))
        except Exception as e:
            kinds.append((type(e).__name__, None))
    assert kinds[0] == kinds[1] == ("HTTPError", 404), kinds


def test_an_optional_file_missing_does_not_kill_the_run(tmp_path, srv) -> None:
    """整轮下载不该被一个可选文件绊倒。"""
    import json as _json
    import subprocess as _sp
    import sys as _sys

    from p2pmoe.sim.fake_checkpoint import (
        TINY_QWEN3_NEXT,
        write_fake_next_checkpoint,
    )

    cfg = dict(TINY_QWEN3_NEXT)
    d = Path(write_fake_next_checkpoint(str(tmp_path / "ck"), cfg, seed=0))
    assert not (d / "special_tokens_map.json").exists(), "前提：这个文件本来就没有"

    s2 = WeightServer(d, "127.0.0.1", 0).start()
    try:
        L, E = cfg["num_hidden_layers"], cfg["num_experts"]
        l0 = L // 3
        man = {"l0": l0, "model": {}, "segments": {"B0": {
            "role": "back:u", "task": "u", "nodes": ["nb"],
            "splits": [[l0 + 1, L]], "head": "nb", "tail": "nb", "hops": 0,
            "compute_ms": 1., "hop_ms": 0., "delay_ms": 1.}},
            "nodes": [{"node": "nb", "role": "back:u", "segment": "B0",
                       "position": 0, "is_head": True, "is_tail": True,
                       "layer_range": [l0 + 1, L], "weight_gb": .1, "kv_gb": 0.,
                       "total_gb": .1,
                       "layers": [{"layer": l, "experts": list(range(0, E, 3)),
                                   "weight_gb": .01, "kv_gb": 0.}
                                  for l in range(l0 + 1, L + 1)]}]}
        plan = tmp_path / "plan.json"
        plan.write_text(_json.dumps(man), encoding="utf-8")
        out = tmp_path / "w"
        r = _sp.run([_sys.executable, "-m", "p2pmoe.deploy.fetch",
                     "--plan", str(plan), "--node", "nb",
                     "--base-url", f"http://127.0.0.1:{s2.port}",
                     "--out", str(out), "--transport", "curl"],
                    capture_output=True, text=True, timeout=180,
                    cwd=Path(__file__).resolve().parent.parent)
        assert r.returncode == 0, r.stdout + r.stderr
        assert list(out.glob("*.safetensors"))
    finally:
        s2.stop()


# --------------------------------------------------------------------------- #
# 5. 只用老 curl 也有的选项
# --------------------------------------------------------------------------- #
def test_no_options_that_need_a_recent_curl() -> None:
    """真踩过：`--fail-with-body` 要 curl ≥ 7.76，集群里的常常更老。

    症状很难看 —— 每次请求先等一遍 urllib 超时，再被 curl 以
    `option --fail-with-body: is unknown` 顶回来，如此往复。

    只看**真正拼进命令行的那些**，注释里提到它们是好事（说明为什么不用）。
    """
    import ast

    src = (Path(__file__).resolve().parent.parent
           / "p2pmoe" / "deploy" / "fetch.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    opts: set[str] = set()
    for node in ast.walk(tree):
        # 找 `cmd = ["curl", ...]` 与后续 `cmd += [...]`
        if isinstance(node, (ast.List, ast.Tuple)):
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if any(v == "curl" for v in vals) or any(v.startswith("-") for v in vals):
                opts |= {v for v in vals if v.startswith("-")}
    assert opts, "找不到 curl 的命令行 —— 拼法变了？"
    RECENT = {"--fail-with-body", "--retry-all-errors", "--json",
              "--parallel", "--remove-on-error", "--no-clobber"}
    assert not (opts & RECENT), f"用了较新 curl 才有的选项：{opts & RECENT}"


def test_a_404_body_is_not_swallowed(srv) -> None:
    """不用 --fail 的另一个理由：错误响应体有时才是真正的信息
    （镜像的提示页、限流说明）。"""
    import urllib.error as ue

    with pytest.raises(ue.HTTPError):
        _src(srv, "curl").read("nope.json")


# --------------------------------------------------------------------------- #
# 6. 切换必须在「决定切」时粘住
# --------------------------------------------------------------------------- #
def test_the_switch_sticks_even_if_curl_then_fails(srv, monkeypatch, caplog) -> None:
    """**真踩过。**

    原来是 curl 成功之后才把 transport 置为 curl。curl 本身有问题时，
    每一次请求都会重走一遍 urllib（同样的超时、同样的等待），打同一句话，
    而真正的 curl 报错被淹没在重复里 —— 日志里那句话出现了五遍。
    """
    s = _src(srv, "auto")
    n = {"u": 0, "c": 0}

    def bad_urllib(*a, **k):
        n["u"] += 1
        raise urllib.error.URLError("handshake timed out")

    def bad_curl(*a, **k):
        n["c"] += 1
        raise OSError("curl: option --whatever: is unknown")

    monkeypatch.setattr(Source, "_get_urllib", bad_urllib)
    monkeypatch.setattr(Source, "_get_curl", bad_curl)
    monkeypatch.setattr("p2pmoe.deploy.fetch.time.sleep", lambda *_: None)
    with caplog.at_level("WARNING", logger="p2pmoe.fetch"):
        with pytest.raises(OSError):
            s.read("m.safetensors", 0, 100)

    assert n["u"] == 1, f"urllib 被重试了 {n['u']} 次 —— 切换没粘住"
    assert caplog.text.count("改用 curl") == 1, "同一句话打了多遍"
    assert s.transport == "curl"


def test_curl_availability_is_actually_probed(monkeypatch, tmp_path) -> None:
    """PATH 里有还不够 —— 装了但跑不起来的情况是有的，
    那时候切过去只是把一个错换成另一个错。"""
    s = Source(base_url="http://127.0.0.1:1", transport="auto")
    monkeypatch.setattr("p2pmoe.deploy.fetch.shutil.which", lambda _: "/usr/bin/curl")
    monkeypatch.setattr("p2pmoe.deploy.fetch.subprocess.run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert s._have_curl() is False
