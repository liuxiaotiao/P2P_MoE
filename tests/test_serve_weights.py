"""本地权重源：必须真的支持 Range。

这个文件存在的理由是一句实测结论：`python -m http.server` **不支持 Range** ——
它无视 Range 头，返回 200 加整个文件。而「只拉自己那份张量」完全靠 Range，
所以那条看似顺手的路是死的。

上游拉不动时的备用路线是「下一次全量 → 局域网切片」，而它成立的前提就是
这台服务器把 Range 做对。做错的后果不是慢，是**切出来的权重文件是错的**
（调用方按区间长度去切一个整文件）。
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.serve_weights import WeightServer

BLOB = bytes(range(256)) * 400          # 102400 字节，内容可按位置校验


@pytest.fixture
def srv(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(BLOB)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.safetensors").write_bytes(b"nested")
    s = WeightServer(tmp_path, "127.0.0.1", 0).start()
    yield s
    s.stop()


def get(s: WeightServer, path: str, rng: str | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{s.port}/{path}")
    if rng:
        req.add_header("Range", rng)
    return urllib.request.urlopen(req, timeout=10)


# --------------------------------------------------------------------------- #
# 1. Range —— 这个文件的全部理由
# --------------------------------------------------------------------------- #
def test_a_range_request_gets_206_and_exactly_those_bytes(srv) -> None:
    r = get(srv, "model.safetensors", "bytes=100-199")
    assert r.status == 206
    body = r.read()
    assert len(body) == 100
    assert body == BLOB[100:200], "取到的不是那一段 —— 拼出来的权重会是错的"


def test_the_content_range_header_is_right(srv) -> None:
    """调用方按这个头核对自己拿到的是不是要的那段。"""
    r = get(srv, "model.safetensors", "bytes=0-9")
    assert r.headers["Content-Range"] == f"bytes 0-9/{len(BLOB)}"


def test_an_open_ended_range_runs_to_the_end(srv) -> None:
    r = get(srv, "model.safetensors", f"bytes={len(BLOB)-10}-")
    assert r.read() == BLOB[-10:]


def test_a_suffix_range_takes_the_last_n_bytes(srv) -> None:
    r = get(srv, "model.safetensors", "bytes=-16")
    assert r.read() == BLOB[-16:]


def test_a_range_past_the_end_is_416_not_200(srv) -> None:
    """**不能退回 200。** 退回去的话调用方会把整个文件当成它要的那一小段 ——
    没有报错，只有一份内容错位的权重。"""
    with pytest.raises(urllib.error.HTTPError) as e:
        get(srv, "model.safetensors", f"bytes={len(BLOB)+50}-{len(BLOB)+99}")
    assert e.value.code == 416


def test_a_malformed_range_is_416_not_200(srv) -> None:
    with pytest.raises(urllib.error.HTTPError) as e:
        get(srv, "model.safetensors", "bytes=abc")
    assert e.value.code == 416


def test_no_range_returns_the_whole_file(srv) -> None:
    r = get(srv, "model.safetensors")
    assert r.status == 200 and r.read() == BLOB


def test_accept_ranges_is_advertised(srv) -> None:
    assert get(srv, "model.safetensors").headers["Accept-Ranges"] == "bytes"


def test_many_ranges_over_one_server_stay_correct(srv) -> None:
    """15 台会同时来，各要各的区间 —— 串了就是静默的数据损坏。"""
    for lo in range(0, 4000, 337):
        assert get(srv, "model.safetensors", f"bytes={lo}-{lo+99}").read() \
            == BLOB[lo:lo + 100]


# --------------------------------------------------------------------------- #
# 2. 只读、不越界
# --------------------------------------------------------------------------- #
def test_a_nested_file_is_reachable(srv) -> None:
    assert get(srv, "sub/b.safetensors").read() == b"nested"


def test_climbing_out_of_the_root_is_refused(srv) -> None:
    with pytest.raises(urllib.error.HTTPError) as e:
        get(srv, "../../etc/passwd")
    assert e.value.code == 404


def test_a_missing_file_is_404(srv) -> None:
    with pytest.raises(urllib.error.HTTPError) as e:
        get(srv, "nope.safetensors")
    assert e.value.code == 404


def test_a_directory_is_not_listed(srv) -> None:
    """列目录对权重源没用，只多一个面。"""
    with pytest.raises(urllib.error.HTTPError):
        get(srv, "sub")


# --------------------------------------------------------------------------- #
# 3. 和 fetch 真的接得上
# --------------------------------------------------------------------------- #
def test_fetch_slices_through_this_server(srv, tmp_path) -> None:
    """端到端：Source 按区间读回来的内容要和原文件逐字节一致。"""
    from p2pmoe.deploy.fetch import Source

    s = Source(base_url=f"http://127.0.0.1:{srv.port}")
    assert s.read("model.safetensors", 1000, 1064) == BLOB[1000:1064]
    assert s.read("model.safetensors") == BLOB


def test_a_server_without_range_would_be_caught(tmp_path) -> None:
    """反证：不支持 Range 的源必须被 fetch 抓出来，而不是将就。

    这正是 `python -m http.server` 的行为 —— 实测返回 200 加整个文件。
    """
    import http.server
    import socketserver
    import threading

    from p2pmoe.deploy.fetch import RangeNotSupported, Source

    (tmp_path / "m.safetensors").write_bytes(BLOB)
    h = http.server.SimpleHTTPRequestHandler
    srv = socketserver.TCPServer(("127.0.0.1", 0), 
                                 lambda *a: h(*a, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        s = Source(base_url=f"http://127.0.0.1:{srv.server_address[1]}")
        with pytest.raises(RangeNotSupported):
            s.read("m.safetensors", 0, 10)
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------- #
# 4. 端到端：一台发，多台切
# --------------------------------------------------------------------------- #
def test_two_nodes_slice_different_parts_of_one_checkpoint(tmp_path) -> None:
    """备用路线的整条链：一份全量在一台机器上，其余节点各切各的。

    切出来的必须是**合法的 safetensors**，不是「下了一堆字节」—— 节点接着就要
    用它加载模型，格式不对的话错在几步之后才发作。
    """
    import json
    import subprocess
    import sys as _sys

    from safetensors import safe_open

    from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_NEXT, write_fake_next_checkpoint

    cfg = dict(TINY_QWEN3_NEXT)
    d = Path(write_fake_next_checkpoint(str(tmp_path / "full"), cfg, seed=0))
    full = sum(f.stat().st_size for f in d.glob("*.safetensors"))

    L, E = cfg["num_hidden_layers"], cfg["num_experts"]
    l0 = L // 3

    def lay(a, b, ex):
        return [{"layer": l, "experts": ex, "weight_gb": .01, "kv_gb": 0.}
                for l in range(a, b + 1)]

    def seg(nodes, a, b):
        return {"nodes": nodes, "splits": [[a, b]], "head": nodes[0],
                "tail": nodes[-1], "hops": 0, "compute_ms": 1.,
                "hop_ms": 0., "delay_ms": 1.}

    man = {"l0": l0, "model": {}, "segments": {
        "F0": {"role": "front", "task": None, **seg(["nf"], 1, l0)},
        "B0": {"role": "back:u", "task": "u", **seg(["nb"], l0 + 1, L)}},
        "nodes": [
            {"node": "nf", "role": "front", "segment": "F0", "position": 0,
             "is_head": True, "is_tail": True, "layer_range": [1, l0],
             "weight_gb": .1, "kv_gb": 0., "total_gb": .1,
             "layers": lay(1, l0, list(range(E)))},
            {"node": "nb", "role": "back:u", "segment": "B0", "position": 0,
             "is_head": True, "is_tail": True, "layer_range": [l0 + 1, L],
             "weight_gb": .1, "kv_gb": 0., "total_gb": .1,
             "layers": lay(l0 + 1, L, list(range(0, E, 3)))}]}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(man), encoding="utf-8")

    s = WeightServer(d, "127.0.0.1", 0).start()
    try:
        sizes = {}
        for node in ("nf", "nb"):
            out = tmp_path / f"w-{node}"
            r = subprocess.run(
                [_sys.executable, "-m", "p2pmoe.deploy.fetch", "--plan", str(plan),
                 "--node", node, "--base-url", f"http://127.0.0.1:{s.port}",
                 "--out", str(out)],
                capture_output=True, text=True, timeout=180,
                cwd=Path(__file__).resolve().parent.parent)
            assert r.returncode == 0, r.stdout + r.stderr
            files = sorted(out.glob("*.safetensors"))
            assert files, "什么都没写出来"
            keys = [k for f in files
                    for k in safe_open(str(f), framework="pt").keys()]
            assert keys, "切出来的文件里没有张量"
            sizes[node] = sum(f.stat().st_size for f in files)
    finally:
        s.stop()

    assert all(v < full for v in sizes.values()), "有节点拉了全量 —— 切片没生效"
    assert s.stats["reqs"] > 0, "服务端没收到请求"


def test_slicing_needs_no_network_when_the_source_is_a_local_dir(tmp_path) -> None:
    """有共享盘就连服务都不用起 —— Source 直接 seek 本地文件。"""
    from p2pmoe.deploy.fetch import Source

    (tmp_path / "m.safetensors").write_bytes(BLOB)
    s = Source(local=str(tmp_path))
    assert s.read("m.safetensors", 500, 540) == BLOB[500:540]
