"""源站诊断：小文件与大文件**不在同一个域名上**。

「小文件过得去、大文件过不去」有两种解释，修法完全不同：

1. 链路把长传输掐断（代理、MTU、CDN 抖动）→ 调网络
2. **它们根本是两个域名** → 让 IT 放行第二个

HF 把 config.json 这类小文件由 huggingface.co 直接发，而 LFS 文件
（tokenizer.json、全部 safetensors）**302 到 CDN**。企业白名单常常只放行了
前者。这时候调 MTU、换镜像都是白费力气 —— 要做的只是把 CDN 域名报给 IT。

这个文件盯着「诊断能不能把该报的那个域名找出来」。
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from p2pmoe.deploy.fetch import Source, diagnose
from p2pmoe.deploy.serve_weights import WeightServer

DEAD = "http://127.0.0.1:1/blocked-cdn/tokenizer.json"   # 端口 1，必然连不上


def _redirecting_server(tmp_path: Path):
    """小文件直接发，tokenizer.json 302 到一个连不上的地方 —— 复刻 HF 的形状。"""
    (tmp_path / "config.json").write_text('{"model_type":"x"}', encoding="utf-8")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _go(self):
            if self.path.endswith("tokenizer.json"):
                self.send_response(302)
                self.send_header("Location", DEAD)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = (tmp_path / self.path.lstrip("/")).read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _go
        do_HEAD = _go

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_it_names_the_cdn_host_to_allowlist(tmp_path, caplog) -> None:
    """**这条是重点。** 输出里必须出现那个要报给 IT 的域名。"""
    srv = _redirecting_server(tmp_path)
    try:
        s = Source(base_url=f"http://127.0.0.1:{srv.server_address[1]}", timeout=5)
        with caplog.at_level("INFO", logger="p2pmoe.fetch"):
            diagnose(s)
    finally:
        srv.shutdown()
        srv.server_close()
    out = caplog.text
    assert "127.0.0.1:1" in out, "没点名重定向到哪个域名"
    assert "放行" in out, "没告诉人要做什么"


def test_it_distinguishes_reachable_main_domain(tmp_path, caplog) -> None:
    """主域名通、CDN 不通 —— 这个对比正是结论的依据，不能只报一个失败。"""
    srv = _redirecting_server(tmp_path)
    try:
        s = Source(base_url=f"http://127.0.0.1:{srv.server_address[1]}", timeout=5)
        with caplog.at_level("INFO", logger="p2pmoe.fetch"):
            diagnose(s)
    finally:
        srv.shutdown()
        srv.server_close()
    assert "可达" in caplog.text and "连不上" in caplog.text


def test_same_domain_sends_you_back_to_the_network(tmp_path, caplog) -> None:
    """没有重定向就不是域名问题 —— 别让人白跑一趟 IT。"""
    from p2pmoe.sim.fake_checkpoint import TINY_QWEN3_NEXT, write_fake_next_checkpoint

    d = Path(write_fake_next_checkpoint(str(tmp_path / "ck"),
                                        dict(TINY_QWEN3_NEXT), seed=0))
    s = WeightServer(d, "127.0.0.1", 0).start()
    try:
        with caplog.at_level("INFO", logger="p2pmoe.fetch"):
            diagnose(Source(base_url=f"http://127.0.0.1:{s.port}", timeout=5))
    finally:
        s.stop()
    assert "同一个域名" in caplog.text
    assert "MTU" in caplog.text or "代理" in caplog.text


def test_an_unreachable_main_domain_stops_early(caplog) -> None:
    """主域名都不通的话，后面的重定向诊断没有意义 —— 别输出误导性的结论。"""
    with caplog.at_level("INFO", logger="p2pmoe.fetch"):
        diagnose(Source(base_url="http://127.0.0.1:1", timeout=2))
    assert "别往下看了" in caplog.text
    assert "放行" not in caplog.text


def test_a_local_source_needs_no_diagnosis(tmp_path, caplog) -> None:
    with caplog.at_level("INFO", logger="p2pmoe.fetch"):
        diagnose(Source(local=str(tmp_path)))
    assert "无需诊断" in caplog.text
